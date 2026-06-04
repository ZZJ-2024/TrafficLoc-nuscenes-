import os
import copy
import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F

from .backbone import build_backbone, PointTransformer, ImageEncoder, ImageUpSample, OneImageUpSampleDummy, ImageUpSampleNearest, OneImageUpSample, DepthEncoder
from .backbone.resnet_fpn import BasicBlock
from .backbone.dust_backbone import DustEncoder
from .transformer_module.cofi_att import LocalFeatureTransformer, PositionEmbeddingCoordsSineCoFiI2P
from .transformer_module.biatt_decoder import DecoderBlock, DecoderBlockQKPos, PositionGetter, PositionEmbeddingCoordsSine, PointPE
from .transformer_module.pimae import TransformerEncoder, TransformerEncoderLayer, loadPretrain
from .pointnet2.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation, PointNetSetAbstractionRetIdx
from .loss_utils import fm_desc_loss, fine_circle_loss
from .kpconv.kp_backbone import KPConvFPN

from einops import rearrange

def MLP(channels: list, do_bn=True, ac_fn='relu', norm_fn='bn'):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if norm_fn == 'in':
                layers.append(nn.InstanceNorm1d(channels[i], eps=1e-3))
            elif norm_fn == 'bn':
                layers.append(nn.BatchNorm1d(channels[i], eps=1e-3))
            elif norm_fn == 'ln':
                layers.append(nn.LayerNorm(channels[i]))
            if ac_fn == 'relu':
                layers.append(nn.ReLU())
            elif ac_fn == 'gelu':
                layers.append(nn.GELU())
            elif ac_fn == 'lrelu':
                layers.append(nn.LeakyReLU(negative_slope=0.1))
            
    return nn.Sequential(*layers)

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, n_dim: int = 1, d_model: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.mlp(xyz)

class TrafficLocFeatureMatch(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        
        # 1. Image Encoder
        if self.config['feature_match']['dust_backbone']:
            self.enc_dim = self.config['feature_match']['image_enc_dim']
            self.image_backbone = DustEncoder()
            self.encoder_embed_img=MLP([1024, self.enc_dim, self.enc_dim], 
                                    do_bn=True,
                                    ac_fn='relu',
                                    norm_fn='bn'
                                    )
        else:
            self.image_backbone = ImageEncoder()
            self.enc_dim = self.config['feature_match']['image_enc_dim']
            align = self.config['feature_match']['align_corner']
            self.up_conv1=ImageUpSample(256+128,128, align_c=align)
            self.up_conv2=ImageUpSample(128+64,64, align_c=align)
            self.up_conv3=ImageUpSample(64+64,64, align_c=align)
        
        if self.config['feature_match']['fine_resnet'] and not self.config['feature_match']['fine_resnet_only']:
            self.image_backbone_res = ImageEncoder()
            self.enc_dim = self.config['feature_match']['image_enc_dim']
            align = self.config['feature_match']['align_corner']
            self.up_conv1=ImageUpSample(256+128,128, align_c=align)
            self.up_conv2=ImageUpSample(128+64,64, align_c=align)
            self.up_conv3=ImageUpSample(64+64,64, align_c=align)
        
        if self.config['feature_match']['dust_with_resnet'] and self.config['feature_match']['dust_backbone']:
            self.image_backbone_res = ImageEncoder()
            self.enc_dim = self.config['feature_match']['image_enc_dim']
            align = self.config['feature_match']['align_corner']
            self.up_conv1=OneImageUpSampleDummy(256,128, align_c=align)
            self.up_conv2=ImageUpSample(128+64,64, align_c=align)
            self.up_conv3=ImageUpSample(64+64,64, align_c=align)
            self.patch_enc = MLP([self.enc_dim, 1024], 
                                    do_bn=True,
                                    ac_fn='relu',
                                    norm_fn='bn'
                                    )
            
            
        # 2. Point Encoder
        if self.config['feature_match']['point_backbone']=='pt':
            self.point_backbone = PointTransformer(config=self.config)
        elif self.config['feature_match']['point_backbone'] == 'kpconv':
            self.point_backbone = KPConvFPN(input_dim=4, output_dim=64, init_dim=64, kernel_size=15, init_radius=4.25*0.1, init_sigma=2*0.1, norm = 'gn', group_norm=32)
        elif self.config['feature_match']['point_backbone']=='pimae':
            if self.config['feature_match']['kitti']:
                in_channel = 7
            else:
                in_channel = 3
            self.point_backbone = PointNetSetAbstractionRetIdx(npoint=self.config['feature_match']['num_cluster'], 
                                                        radius=self.config['feature_match']['radius'], 
                                                        nsample=self.config['feature_match']['nsample'], 
                                                        in_channel=in_channel, 
                                                        mlp=[64, 128, 256], group_all=False, 
                                                        use_knn=self.config['feature_match']['use_knn'])
            
            self.point_backbone_fp = PointNetFeaturePropagation(256, [256, 256, 256])
                    
            encoder_layer = TransformerEncoderLayer(
                d_model=256,
                nhead=4,
                dim_feedforward=128,
                dropout=0.1,
                activation='relu',
            )
            self.pimae_encoder = TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=6
            )
            self.pimae_pos_encoding = PositionEmbeddingLearned(3, 256)
        else:
            raise ValueError

        # 3. Cross-Modal Feature Fusion
        if self.config['feature_match']['cofii2p_pipe']:
            self.img_pos_encoding = PositionEmbeddingCoordsSineCoFiI2P(2, self.enc_dim) # dim=128 in cofii2p
            self.pc_pos_encoding = PositionEmbeddingCoordsSineCoFiI2P(3, self.enc_dim)
            self.transformer = LocalFeatureTransformer(D_MODEL=self.enc_dim, NHEAD=4, LAYER_NAMES=['self', 'cross'] * self.config['feature_match']['cofi_num_layer'], ATTENTION = 'full')
            self.img_pos_generator = PositionGetter()
        elif self.config['feature_match']['cofi_fusion']:
            self.img_pos_encoding = PositionEmbeddingCoordsSineCoFiI2P(2, self.enc_dim) # dim=128 in cofii2p
            self.pc_pos_encoding = PositionEmbeddingCoordsSineCoFiI2P(3, self.enc_dim)
            self.fusion_att = LocalFeatureTransformer(D_MODEL=self.enc_dim, NHEAD=4, LAYER_NAMES=['self', 'cross'] * self.config['feature_match']['cofi_num_layer'], ATTENTION = 'full')
            self.img_pos_generator = PositionGetter()
            
        # 4. Feature processing layer
        if self.config['feature_match']['cofii2p_pipe']:
            if self.config['feature_match']['point_backbone']=='kpconv':
                self.pc_feature_layer=nn.Sequential(nn.Linear(2048,1024,bias=False),
                                                    nn.LayerNorm(1024),nn.ReLU(),
                                                    nn.Linear(1024,512,bias=False),
                                                    nn.LayerNorm(512),
                                                    nn.ReLU(),
                                                    nn.Linear(512,128,bias=False))
            else:
                self.pc_feature_layer=nn.Sequential(nn.Linear(self.enc_dim,self.enc_dim,bias=False),
                                                    nn.LayerNorm(self.enc_dim),nn.ReLU(),
                                                    nn.Linear(self.enc_dim,self.enc_dim,bias=False),
                                                    nn.LayerNorm(self.enc_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(self.enc_dim,self.enc_dim,bias=False))

        if self.config['feature_match']['coarse_match_type'] in ['bce','bce_dense']:
            self.coarse_pc_score_mlp = MLP([self.enc_dim, self.enc_dim//2, 1], 
                                    do_bn=True,
                                    ac_fn='relu',
                                    norm_fn='bn'
                                    )
        
        if self.config['feature_match']['upsample']:
            self.image_feature_up1 = OneImageUpSample(in_channel=256, output_channel=256)
            self.image_feature_up2 = OneImageUpSample(in_channel=256, output_channel=256)
            
        if self.config['feature_match']['fine_match']:
            if self.config['feature_match']['cofii2p_pipe']:
                self.node_fuse = MLP([2*self.enc_dim, self.enc_dim, self.enc_dim], 
                                    do_bn=True,
                                    ac_fn='relu',
                                    norm_fn='bn'
                                    )
                self.fine_point = MLP([2*self.enc_dim, 128, 64], 
                                    do_bn=True,
                                    ac_fn='relu',
                                    norm_fn='bn'
                                    )
            else:## 不是哦用cofii2p,就通过上采样来获得fine image feature
                if not self.config['feature_match']['fine_resnet']:
                    self.image_feature_up1 = OneImageUpSample(in_channel=256, output_channel=128)
                    self.image_feature_up2 = OneImageUpSample(in_channel=128, output_channel=64)
                    self.image_feature_up3 = OneImageUpSample(in_channel=64, output_channel=64)
                if self.config['feature_match']['fine_resnet_up_nofuse'] and not self.config['feature_match']['fine_up_point']:
                    self.fine_point = MLP([2*self.enc_dim, 128, 64], 
                                        do_bn=True,
                                        ac_fn='relu',
                                        norm_fn='bn'
                                        )
                else:
                    self.node_fuse = MLP([2*self.enc_dim, self.enc_dim, self.enc_dim], 
                                        do_bn=True,
                                        ac_fn='relu',
                                        norm_fn='bn'
                                        )
                    self.fine_point = MLP([2*self.enc_dim, 128, 64], 
                                        do_bn=True,
                                        ac_fn='relu',
                                        norm_fn='bn'
                                        )
        elif not self.config['feature_match']['fine_match']:
            None
        
        # 5. initialization
        for m_name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 6. load pretrained module
        ## 加载DUSt3R
        if self.config['feature_match']['dust_backbone']:
            model_path = "model_release/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            ckpt = torch.load(model_path, map_location='cpu')
            self.image_backbone.load_state_dict(ckpt['model'], strict=False)
            del ckpt
        ## 加载pimae
        if self.config['feature_match']['point_backbone']=='pimae' and self.config['feature_match']['pimae']:
            pretrain_path = "model_release/pimae.pth"
            print(f"load pimae from {pretrain_path}")
            try:
                loadPretrain(self.pimae_encoder, pretrain_path, 3, 3, prefix='', is_distributed=True)
            except:
                loadPretrain(self.pimae_encoder, pretrain_path, 3, 3, prefix='', is_distributed=False)

        # 7. others
        self.xy_grid_dict = {}  
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        self.clip_temperature = torch.nn.Parameter(torch.tensor(1.))
        self.euc_temperature = torch.nn.Parameter(torch.tensor(1.))
        
        if self.config['feature_match']['cofii2p_pipe']:
            self.bin_emb = nn.Parameter(torch.randn(1, self.enc_dim))
            self.temperature = nn.Parameter(torch.tensor(1.))
            self.dense_temperature = nn.Parameter(torch.tensor(1.))
            self.fine_temperature = nn.Parameter(torch.tensor(1.))
            self.fine_dense_temperature = nn.Parameter(torch.tensor(1.))
        else:
            if self.config['feature_match']['coarse_match']:
                self.bin_emb = nn.Parameter(torch.randn(1, self.enc_dim))
                self.temperature = nn.Parameter(torch.tensor(1.))
            
            if self.config['feature_match']['dense_temp']:
                self.dense_temperature = nn.Parameter(torch.tensor(1.))
            
            if self.config['feature_match']['fine_match'] and (self.config['feature_match']['fine_softmax'] or self.config['feature_match']['fine_loss_type'] == 'fine_softmax'):
                self.fine_temperature = nn.Parameter(torch.tensor(1.))
                
            if self.config['feature_match']['fine_match'] and self.config['feature_match']['fine_loss_type'] == 'dense':
                self.fine_dense_temperature = nn.Parameter(torch.tensor(1.))
                
    
    def fine_matching_softmax(self, fine_patch_feature,fine_pt_feat, corr_flatten_id, valid_mask):
        if self.config['feature_match']['fine_match_norm'] == "dim_norm":
            # dim normalize
            fine_patch_feature, fine_pt_feat = map(lambda feature: feature / feature.shape[-1]**.5,
                                    [fine_patch_feature, fine_pt_feat])
        elif self.config['feature_match']['fine_match_norm'] == "l2_norm":
            # cos similarity normalize
            fine_patch_feature = F.normalize(fine_patch_feature, p=2, dim=-1) 
            fine_pt_feat = F.normalize(fine_pt_feat, p=2, dim=-1) #
        elif self.config['feature_match']['fine_match_norm'] == "no_norm":
            fine_patch_feature = fine_patch_feature
            fine_pt_feat = fine_pt_feat
        else:
            raise ValueError
        
        fine_sim_matrix = torch.einsum("nc,nlc->nl", fine_pt_feat, fine_patch_feature)
        fine_sim_matrix = fine_sim_matrix * self.fine_temperature
        
        loss = F.cross_entropy(fine_sim_matrix, corr_flatten_id, reduction="none")
        
        final_loss = (loss * valid_mask).sum() / valid_mask.sum()
        
        return final_loss
    
    def coarse_matching_softmax(self, feat, voxel_feature, conf_gt, data, sel_proj_patch_idx):
        batch_size, num_point_group, feat_dim = voxel_feature.shape
                
        if self.config['feature_match']['coarse_match_norm'] == "dim_norm":
            # dim normalize
            feat, voxel_feature = map(lambda feature: feature / feature.shape[-1]**.5,
                                    [feat, voxel_feature])
        elif self.config['feature_match']['coarse_match_norm'] == "l2_norm":
            # cos similarity normalize
            feat = F.normalize(feat, p=2, dim=-1) # b n c
            voxel_feature = F.normalize(voxel_feature, p=2, dim=-1) # b n c
        elif self.config['feature_match']['coarse_match_norm'] == "no_norm":
            feat = feat
            voxel_feature = voxel_feature
        else:
            raise ValueError
        
        sim_matrix = torch.einsum("nlc,nsc->nls", voxel_feature, feat)
        sim_matrix = sim_matrix * self.temperature

        loss = F.cross_entropy(sim_matrix.reshape(-1, sim_matrix.shape[-1]), 
                                sel_proj_patch_idx.reshape(-1), 
                                reduction="mean")
        
        data['coarse_matching_loss'] = loss
    
    def coarse_matching_dense(self, feat, voxel_feature, data, sel_proj_patch_idx, fps_idx, n_H, n_W):
        batch_size, num_point_group, feat_dim = voxel_feature.shape
        num_patches = feat.shape[1]
        
        proj_xy_int = data['proj_xy_int'] # Bx2x20480
        sel_proj_xy_int = torch.gather(proj_xy_int, dim=-1, index=fps_idx.unsqueeze(1).repeat(1,2,1)) # Bx2x512
        proj_xy = data['proj_xy'] # Bx2x20480
        sel_proj_xy = torch.gather(proj_xy, dim=-1, index=fps_idx.unsqueeze(1).repeat(1,2,1)) # Bx2x512
        proj_patch = data['proj_patch']
        sel_proj_patch = torch.gather(proj_patch, dim=1, index=fps_idx.unsqueeze(-1).repeat(1,1,2)) # Bx2x512
        
        if self.config['feature_match']['coarse_match_norm'] == "dim_norm":
            # dim normalize
            feat, voxel_feature = map(lambda feature: feature / feature.shape[-1]**.5,
                                    [feat, voxel_feature])
        elif self.config['feature_match']['coarse_match_norm'] == "l2_norm":
            # cos similarity normalize
            feat = F.normalize(feat, p=2, dim=-1) # b n c
            voxel_feature = F.normalize(voxel_feature, p=2, dim=-1) # b n c
        elif self.config['feature_match']['coarse_match_norm'] == "no_norm":
            feat = feat
            voxel_feature = voxel_feature
        else:
            raise ValueError
        
        sim_matrix = torch.einsum("nlc,nsc->nls", voxel_feature, feat)
        
        # add bin
        if self.config['feature_match']['dense_temp']:
            sim_matrix = sim_matrix * self.dense_temperature 
            sim_matrix = torch.softmax(sim_matrix, dim=-1) # 最终softmax之后的matrix
            sim_matrix = sim_matrix.reshape(batch_size, num_point_group, n_H, n_W)
        else:
            ## 通过bin_score来学习一个bin的相似度分数, 然后把这个分数拼接到sim_matrix的最后一维, 作为一个额外的bin, 代表不匹配到任何patch的情况
            ## 从而排除有些不能匹配的图像和点云，关注能匹配的点
            bins0 = self.bin_score.expand(batch_size, num_point_group, 1) 
            sim_matrix = torch.concat([sim_matrix, bins0], dim=-1)
            sim_matrix = sim_matrix * self.temperature 
            sim_matrix = torch.softmax(sim_matrix, dim=-1) # 最终softmax之后的matrix
            sim_matrix = sim_matrix[:,:,:-1] # B x L x n_H*n_W
            sim_matrix = sim_matrix.reshape(batch_size, num_point_group, n_H, n_W)
        
        
        if self.config['feature_match']['coarse_dense_window'] > 0:
            if self.config['feature_match']['coarse_window_gt']:
                max_index_x = sel_proj_patch[:, :, 0].reshape(-1)
                max_index_y = sel_proj_patch[:, :, 1].reshape(-1)
                sim_matrix = sim_matrix.reshape(-1, n_H, n_W) # [L, nH, nW]
            else:
                sim_matrix = sim_matrix.reshape(batch_size, num_point_group, n_H*n_W)
                max_index_flatten = torch.argmax(sim_matrix, dim=-1).reshape(-1)
                max_index_x = max_index_flatten % n_W
                max_index_y = max_index_flatten // n_W
                sim_matrix = sim_matrix.reshape(-1, n_H, n_W) # [L, nH, nW]
            
            window_mask = torch.zeros_like(sim_matrix)   
            window_size = self.config['feature_match']['coarse_dense_window']
            for i in range(-window_size, window_size+1):
                for j in range(-window_size, window_size+1):
                    # make sure the index is within the range
                    clamped_y = torch.clamp(max_index_y+i, 0, n_H-1)   # (H_s * W_s, )
                    clamped_x = torch.clamp(max_index_x+j, 0, n_W-1)   # (H_s * W_s, )
                    window_mask[torch.arange(sim_matrix.shape[0]), clamped_y, clamped_x] = 1
            
            sim_matrix = sim_matrix * window_mask
            sim_matrix = sim_matrix.reshape(batch_size, num_point_group, n_H, n_W)
        else:
            None
                    
        # take soft argmax
        x_grid = torch.arange(0, n_W).view(1,1,1,-1).cuda()
        y_grid = torch.arange(0, n_H).view(1,1,-1,1).cuda()
        
        soft_x = torch.sum(x_grid * sim_matrix, dim=[-2,-1]) / torch.sum(sim_matrix, dim=[-2,-1])
        soft_y = torch.sum(y_grid * sim_matrix, dim=[-2,-1]) / torch.sum(sim_matrix, dim=[-2,-1])
        
        matched_xy = torch.stack([soft_x, soft_y], dim=-1)
        
        coarse_patch_size = self.config['feature_match']['coarse_patch_size']
        gt_keypoint = sel_proj_xy_int.permute(0,2,1).reshape(-1,2)
        gt_keypoint = (gt_keypoint + 0.5) / coarse_patch_size - 0.5

        
        pred_keypoint = matched_xy.reshape(-1,2)
        
        valid_mask = (sel_proj_patch_idx != num_patches).reshape(-1)
        
        std = self.config['feature_match']['dense_std']  # TODO: ablation
        noise = torch.randn_like(gt_keypoint, dtype=torch.float32) * std
        gt_keypoint = gt_keypoint + noise
        
        
        # dense loss
        dense_loss = torch.norm(pred_keypoint - gt_keypoint, dim=-1) * valid_mask
        dense_loss = dense_loss.sum() / valid_mask.sum()
        
        data['coarse_dense_loss'] = dense_loss

        return
    
    def extract_patch(self, feature_map, center_points ,size=16):
        '''
        Extract patch on feature_map through center points
        :param center_points: coarse points project to fine feature map in shape [B, N, 2] in xy format not hw
        :param feature_map: fine feature map in shape [B, C, H, W]
        :param size: size of patch in type int
        '''
        batch_size, N, _ = center_points.shape
        _, _, H, W = feature_map.shape
        center_points = center_points.to(torch.int64)
        
        half_size = int(size/2)
        x = torch.arange(-half_size,half_size, device=center_points.device)
        y = torch.arange(-half_size,half_size, device=center_points.device)
        
        x_grid, y_grid = torch.meshgrid(x, y)
        xy_grid = torch.stack([y_grid, x_grid], dim=-1)
        
        center_points = center_points.unsqueeze(2).unsqueeze(2) # B x N x 1 x 1 x 2
        xy_grid = xy_grid.unsqueeze(0).unsqueeze(0) # 1 x 1 x S x S x 2

        xy_ids = center_points + xy_grid # B x N x S x S x 2
        xy_ids = xy_ids.reshape(batch_size, N, -1, 2) # B x N x SS x 2
        x_ids = torch.clamp(xy_ids[:,:,:,0], min=0, max=W-1) # B x N x SS
        y_ids = torch.clamp(xy_ids[:,:,:,1], min=0, max=H-1) # B x N x SS
        
        b_ids = torch.arange(0, batch_size).unsqueeze(-1).unsqueeze(-1) # B x 1 x 1
        b_ids = b_ids.expand_as(x_ids) # B x N x SS

        patch_feat = feature_map[b_ids,:,y_ids,x_ids] # B x N x SS x C
        patch_feat = rearrange(patch_feat, 'b n (s1 s2) c -> b n c s1 s2', s1=size) # B x N x C x S x S

        return patch_feat


    def train_forward(self, data):
        batch_size, _, H, W=data['image'].shape
        coarse_patch_size = self.config['feature_match']['coarse_patch_size']
        
        n_H = H//coarse_patch_size
        n_W = W//coarse_patch_size
        f_H = H//coarse_patch_size
        f_W = W//coarse_patch_size
            
        # 1. get image feature #
        if 'feat' not in data:
            if self.config['feature_match']['dust_backbone']:
                if self.config['feature_match']['dust_with_resnet']:
                    feat_set=self.image_backbone_res(data['image']) # 1/8 resolution
                    feat_s16=feat_set[-3] #256
                    feat_s8=feat_set[-4]  #128
                    feat_s4=feat_set[-5]  #64
                    feat_s2=feat_set[-6]  #64
                    patch_embedding = self.patch_enc(rearrange(feat_s16,'b c h w -> b c (h w)'))
                    patch_embedding = rearrange(patch_embedding,'b c n -> b n c')
                else:
                    patch_embedding = None
                if self.config['feature_match']['frozen_dust']:
                    self.image_backbone.eval()
                    feat=self.image_backbone(data['image'], patch_embedding) # BHWC
                else:
                    feat=self.image_backbone(data['image'], patch_embedding) # BHWC
                feat=rearrange(feat,'b h w c -> b c (h w)')
                feat=feat.contiguous()
                feat=self.encoder_embed_img(feat) # BC(HW)
                if self.config['feature_match']['cofii2p_pipe']:
                    feat = F.normalize(feat, dim=1, p=2)
            else:
                feat_set=self.image_backbone(data['image']) # 1/8 resolution
                feat_global=feat_set[-1]  #512
                feat_s32=feat_set[-2] #512
                feat_s16=feat_set[-3] #256
                if self.config['feature_match']['cofii2p_pipe'] and coarse_patch_size==16:
                    feat_s16 = F.normalize(feat_s16, dim=1, p=2)
                feat_s8=feat_set[-4]  #128
                if self.config['feature_match']['cofii2p_pipe'] and coarse_patch_size==8:
                    feat_s8 = F.normalize(feat_s8, dim=1, p=2)
                feat_s4=feat_set[-5]  #64
                feat_s2=feat_set[-6]  #64
                if coarse_patch_size == 16:
                    feat=rearrange(feat_s16,'b c h w -> b c (h w)')
                elif coarse_patch_size == 8:
                    feat=rearrange(feat_s8,'b c h w -> b c (h w)')
        else:
            if self.config['feature_match']['dust_backbone']:
                feat=data['feat']
                if self.config['feature_match']['dust_with_resnet']:
                    feat_s8 = data['feat_s8']
                    feat_s4 = data['feat_s4']
                    feat_s2 = data['feat_s2']
                else:
                    None
                if self.config['feature_match']['cofii2p_pipe']:
                    feat = F.normalize(feat, dim=1, p=2)
            else:
                feat_set=data['feat']
                feat_global=feat_set[-1]  #512
                feat_s32=feat_set[-2] #512
                feat_s16=feat_set[-3] #256
                if self.config['feature_match']['cofii2p_pipe'] and coarse_patch_size==16:
                    feat_s16 = F.normalize(feat_s16, dim=1, p=2)
                feat_s8=feat_set[-4]  #128
                if self.config['feature_match']['cofii2p_pipe'] and coarse_patch_size==8:
                    feat_s8 = F.normalize(feat_s8, dim=1, p=2)
                feat_s4=feat_set[-5]  #64
                feat_s2=feat_set[-6]  #64
                if coarse_patch_size == 16:
                    feat=rearrange(feat_s16,'b c h w -> b c (h w)')
                elif coarse_patch_size == 8:
                    feat=rearrange(feat_s8,'b c h w -> b c (h w)')
                
        if self.config['feature_match']['fine_resnet'] and not self.config['feature_match']['fine_resnet_only']:
            feat_set=self.image_backbone_res(data['image']) # 1/8 resolution
            feat_global=feat_set[-1]  #512
            feat_s32=feat_set[-2] #512
            feat_s16=feat_set[-3] #256
            feat_s8=feat_set[-4]  #128
            feat_s4=feat_set[-5]  #64
            feat_s2=feat_set[-6]  #64

            
        # 2. get voxel feature #
        batch_size, num_point, _ = data['voxel_points'].shape
        voxel_points = data['voxel_points'].permute(0,2,1) # batch_size, 3, num_point
        if self.config['feature_match']['point_backbone']=='pimae':
            if self.config['feature_match']['kitti']:
                voxel_attribute = data['voxel_feats']
                center_points, voxel_feature, fps_idx = self.point_backbone(voxel_points, voxel_attribute)
            else:
                center_points, voxel_feature, fps_idx = self.point_backbone(voxel_points, None)

            voxel_feature = voxel_feature.permute(0,2,1) # batch_size, num_groups, feature_dim
            center_points = center_points.permute(0,2,1) # batch_size, num_groups, 3
                
            if self.config['feature_match']['pimae']:
                voxel_feature = voxel_feature.transpose(0, 1) # B,M,C -> M,B,C 
                pimae_pos = self.pimae_pos_encoding(center_points)
                pimae_pos = pimae_pos.transpose(0, 1) # B,M,C -> M,B,C
                voxel_feature = self.pimae_encoder(src=voxel_feature, 
                                                pos=pimae_pos,
                                                return_attn_weights=False)[1].transpose(0, 1) # M,B,C -> B,M,C
            else:
                None
        elif self.config['feature_match']['point_backbone']=='pt':
            voxel_nodes = data['voxel_nodes']
            point2node_idx = data['point2node']
            num_proxy = self.config['feature_match']['pt_num_proxy']
            fps_idx = data['node2point'][:, :num_proxy]
            
            voxel_feature, node_proxy_idx, pt_feat, node_feat = self.point_backbone(voxel_points, voxel_nodes, point2node_idx)
            
            if self.config['feature_match']['cofii2p_pipe']:
                voxel_feature = self.pc_feature_layer(voxel_feature)
                voxel_feature = F.normalize(voxel_feature, dim=2, p=2)
            center_points = voxel_nodes[:,:,:num_proxy]
            center_points = center_points.permute(0,2,1).contiguous()
        elif self.config['feature_match']['point_backbone']=='kpconv':
            kpconv_dict=data['kpconv_dict']
            for key in kpconv_dict:
                for j in range(len(kpconv_dict[key])):
                    kpconv_dict[key][j] = torch.squeeze(kpconv_dict[key][j])
            kpconv_dict['feats'] = torch.squeeze(kpconv_dict['feats'])
            pc_feature_set = self.point_backbone(kpconv_dict)
            
            pc_encode_feature_map = pc_feature_set[-1]  # [1280, 2048] N x C
            pc_decode_1 = pc_feature_set[-2]  # [2560, 1024]
            pc_decode_2 = pc_feature_set[-3]  # [5120, 512]
            pc_decode_3 = F.normalize(pc_feature_set[-4], dim=1, p=2)  # [10240, 64] fine matching features
            pc_feature_middle = F.normalize(self.pc_feature_layer(pc_encode_feature_map), dim=1, p=2) # coarse matching features

            fine_pt_feat = pc_decode_3.unsqueeze(0) # 1 x 10240 x 64
            voxel_feature = pc_feature_middle.unsqueeze(0) # 1 x 1280 x 128
            fps_idx = data['fps_idx'] # 1 x 1280
            center_points = kpconv_dict['points'][-1].unsqueeze(0)
        else:
            raise ValueError
            
            

        # 2.5. get gt coarse matrix
        if self.training and self.config['feature_match']['coarse_match']:
            proj_patch_idx = data['proj_patch_idx'] # B x #points
            num_node = fps_idx.shape[1]
            num_patch = n_H * n_W
            sel_proj_patch_idx = torch.gather(proj_patch_idx, dim=1, index=fps_idx)
            neg_mask = (sel_proj_patch_idx == -1)
            sel_proj_patch_idx[neg_mask] = num_patch # bin
            
            gt_conf = torch.zeros((batch_size, num_node, num_patch+1), dtype=torch.bool).cuda()
            indices = sel_proj_patch_idx.unsqueeze(-1)  
            gt_conf.scatter_(-1, indices, True)
            # gt_conf = gt_conf[:,:,:-1]


        # before fusion
        ori_feat = feat.clone().permute(0,2,1) # b n c
        ori_voxel_feature = voxel_feature.clone() # b n c
        
        
        # 3. cross-modal feature fusion #
        if self.config['feature_match']['cofii2p_pipe']:
            feat = rearrange(feat,'b c n -> b n c')
            img_xy = self.img_pos_generator(batch_size, n_H, n_W, feat.device)
            img_pos = self.img_pos_encoding(img_xy)  # b x hw x c
            if self.config['feature_match']['cofi_no_pcd_pos']:
                pc_pos = torch.zeros_like(voxel_feature)
            else:
                pc_pos = self.pc_pos_encoding(center_points)
            
            feat = feat + img_pos
            voxel_feature = voxel_feature + pc_pos
            feat, voxel_feature, raw_att_i2p, raw_att_p2i = self.transformer(feat, voxel_feature)
                
            crs_feat = feat.clone()
            crs_voxel_feature = voxel_feature.clone()
            
            feat = F.normalize(feat, dim=2, p=2)
            voxel_feature = F.normalize(voxel_feature, dim=2, p=2)
            
        elif self.config['feature_match']['cofi_fusion']:
            if self.config['feature_match']['cofi_fusion_norm']:
                feat = F.normalize(feat, dim=1, p=2) # b c (h w)
                voxel_feature = F.normalize(voxel_feature, dim=2, p=2) # b n c
            
            feat = rearrange(feat,'b c n -> b n c')
            
            img_xy = self.img_pos_generator(batch_size, n_H, n_W, feat.device)
            img_pos = self.img_pos_encoding(img_xy)  # b x hw x c
            if self.config['feature_match']['cofi_no_pcd_pos']:
                pc_pos = torch.zeros_like(voxel_feature)
            else:
                pc_pos = self.pc_pos_encoding(center_points)
            
            # input feature
            feat = feat + img_pos
            voxel_feature = voxel_feature + pc_pos
            
            # do bi-direction cross-attention
            feat, voxel_feature, raw_att_i2p, raw_att_p2i = self.fusion_att(feat, voxel_feature, self.config['feature_match']['att_loss_layer'])
            # data['patch_to_node_rad_mask']

                
            crs_feat = feat.clone()
            crs_voxel_feature = voxel_feature.clone()
        else:
            raise ValueError("Must choose fusion type")

        # attention map loss
        if self.config['feature_match']['att_loss'] and self.training:
            if self.config['feature_match']['att_loss_layer'] in ['first', 'last']:
                pred_att_p2i = torch.sigmoid(raw_att_p2i).mean(dim=-1)
                pred_att_i2p = torch.sigmoid(raw_att_i2p).mean(dim=-1)
            elif self.config['feature_match']['att_loss_layer'] == 'all':
                raw_att_p2i = torch.concat(raw_att_p2i, dim=0)
                raw_att_i2p = torch.concat(raw_att_i2p, dim=0)
                pred_att_p2i = torch.sigmoid(raw_att_p2i).mean(dim=-1)
                pred_att_i2p = torch.sigmoid(raw_att_i2p).mean(dim=-1)
                pred_att_p2i = pred_att_p2i.mean(dim=0)
                pred_att_i2p = pred_att_i2p.mean(dim=0)
            else:
                raise ValueError
            
            pos_p2i_loss = (1-pred_att_p2i) * data['node_to_patch_dist_mask']
            neg_p2i_loss = (pred_att_p2i) * data['node_to_patch_dist_mask_neg']
            pos_i2p_loss = (1-pred_att_i2p) * data['patch_to_node_rad_mask']
            neg_i2p_loss = (pred_att_i2p) * data['patch_to_node_rad_mask_neg']
            
            total_loss = pos_p2i_loss.sum()+neg_p2i_loss.sum()+pos_i2p_loss.sum()+neg_i2p_loss.sum()
            total_sample = data['node_to_patch_dist_mask'].sum()+data['node_to_patch_dist_mask_neg'].sum()+ \
                            data['patch_to_node_rad_mask'].sum()+data['patch_to_node_rad_mask_neg'].sum()
            att_loss = total_loss / total_sample

            data['att_loss'] = att_loss
                
        if self.config['feature_match']['upsample']:
            f_H *= 4
            f_W *= 4
        
        if self.config['feature_match']['fine_match'] and self.config['feature_match']['point_backbone'] == 'kpconv':
            # 若kpconv，则处理fine feature逻辑不同
            # image fine feature, feat_s2
            if coarse_patch_size == 16:
                up_feat_s8 = self.up_conv1(rearrange(ori_feat,'b (h w) c -> b c h w', h=n_H), feat_s8)
                up_feat_s4 = self.up_conv2(up_feat_s8, feat_s4)
                up_feat_s2 = self.up_conv3(up_feat_s4, feat_s2)
            elif coarse_patch_size == 8:
                up_feat_s4 = self.up_conv2(rearrange(ori_feat,'b (h w) c -> b c h w', h=n_H), feat_s4)
                up_feat_s2 = self.up_conv3(up_feat_s4, feat_s2)
            feat_s2 = F.normalize(up_feat_s2, dim=1, p=2)
            # point fine feature
            c2f_indices = data['c2f_indices']
            fine_pt_feat = fine_pt_feat.permute(0,2,1) # b c 10240
            fine_pt_feat = torch.gather(fine_pt_feat, dim=-1, index=c2f_indices.unsqueeze(1).expand(-1, fine_pt_feat.shape[1], -1)) # b c 1280
        elif self.config['feature_match']['fine_match']:
            # upsample feature
            if self.config['feature_match']['cofii2p_pipe']:
                if coarse_patch_size == 16:
                    up_feat_s8 = self.up_conv1(rearrange(ori_feat,'b (h w) c -> b c h w', h=n_H), feat_s8)
                    up_feat_s4 = self.up_conv2(up_feat_s8, feat_s4)
                    up_feat_s2 = self.up_conv3(up_feat_s4, feat_s2)
                elif coarse_patch_size == 8:
                    up_feat_s4 = self.up_conv2(rearrange(ori_feat,'b (h w) c -> b c h w', h=n_H), feat_s4)
                    up_feat_s2 = self.up_conv3(up_feat_s4, feat_s2)
            
                scattered_node_proxy_feat = torch.gather(ori_voxel_feature, index=node_proxy_idx.expand(-1, -1, self.enc_dim), dim=1)
                scattered_node_proxy_feat = scattered_node_proxy_feat.permute(0,2,1).contiguous()
                fused_node_feat = torch.cat([node_feat, scattered_node_proxy_feat], dim=1)
                fused_node_feat = self.node_fuse(fused_node_feat)
            
                f = fused_node_feat.shape[1]
                scattered_pt_node_feat = torch.gather(fused_node_feat, index=point2node_idx.unsqueeze(1).expand(-1, f, -1), dim=2)
                fused_pt_feat = torch.cat([pt_feat, scattered_pt_node_feat], dim=1)
                fine_pt_feat = self.fine_point(fused_pt_feat)
                
                feat_s2 = F.normalize(up_feat_s2, dim=1, p=2)
                fine_pt_feat = F.normalize(fine_pt_feat, dim=1, p=2)
                            
            elif self.config['feature_match']['fine_resnet']:
                if self.config['feature_match']['fine_resnet_up']:
                    if self.config['feature_match']['fine_resnet_up_nofuse']:
                        if coarse_patch_size == 16:
                            up_feat_s8 = self.up_conv1(rearrange(ori_feat,'b (h w) c -> b c h w', h=n_H), feat_s8)
                            up_feat_s4 = self.up_conv2(up_feat_s8, feat_s4)
                            up_feat_s2 = self.up_conv3(up_feat_s4, feat_s2)
                        elif coarse_patch_size == 8:
                            up_feat_s4 = self.up_conv2(rearrange(ori_feat,'b (h w) c -> b c h w', h=n_H), feat_s4)
                            up_feat_s2 = self.up_conv3(up_feat_s4, feat_s2)
                        feat_s2 = up_feat_s2
                        if self.config['feature_match']['fine_up_point']:
                            # 融合点云部分feat：用ori_voxel_feature
                            scattered_node_proxy_feat = torch.gather(ori_voxel_feature, index=node_proxy_idx.expand(-1, -1, self.enc_dim), dim=1)
                            scattered_node_proxy_feat = scattered_node_proxy_feat.permute(0,2,1).contiguous()
                            fused_node_feat = torch.cat([node_feat, scattered_node_proxy_feat], dim=1)
                            fused_node_feat = self.node_fuse(fused_node_feat)
                            
                            f = fused_node_feat.shape[1]
                            scattered_pt_node_feat = torch.gather(fused_node_feat, index=point2node_idx.unsqueeze(1).expand(-1, f, -1), dim=2)
                            fused_pt_feat = torch.cat([pt_feat, scattered_pt_node_feat], dim=1)
                            fine_pt_feat = self.fine_point(fused_pt_feat)
                        else:
                            fine_pt_feat = self.fine_point(pt_feat)
                    else:
                        if coarse_patch_size == 16:
                            up_feat_s8 = self.up_conv1(rearrange(feat,'b (h w) c -> b c h w', h=n_H), feat_s8)
                            up_feat_s4 = self.up_conv2(up_feat_s8, feat_s4)
                            up_feat_s2 = self.up_conv3(up_feat_s4, feat_s2)
                        elif coarse_patch_size == 8:
                            up_feat_s4 = self.up_conv2(rearrange(feat,'b (h w) c -> b c h w', h=n_H), feat_s4)
                            up_feat_s2 = self.up_conv3(up_feat_s4, feat_s2)
                        feat_s2 = up_feat_s2
                        # 融合点云部分feat：用cross att之后的voxel_feature
                        scattered_node_proxy_feat = torch.gather(voxel_feature, index=node_proxy_idx.expand(-1, -1, self.enc_dim), dim=1)
                        scattered_node_proxy_feat = scattered_node_proxy_feat.permute(0,2,1).contiguous()
                        fused_node_feat = torch.cat([node_feat, scattered_node_proxy_feat], dim=1)
                        fused_node_feat = self.node_fuse(fused_node_feat)
                        
                        f = fused_node_feat.shape[1]
                        scattered_pt_node_feat = torch.gather(fused_node_feat, index=point2node_idx.unsqueeze(1).expand(-1, f, -1), dim=2)
                        fused_pt_feat = torch.cat([pt_feat, scattered_pt_node_feat], dim=1)
                        fine_pt_feat = self.fine_point(fused_pt_feat)
                        
                else:
                    feat_s2 = feat_s2
                    fine_pt_feat = self.fine_point(pt_feat)
            else:
                assert coarse_patch_size == 16
                feat_s8 = self.image_feature_up1(rearrange(feat, 'b (h w) c -> b c h w', h=n_H, w=n_W))
                feat_s4 = self.image_feature_up2(feat_s8)
                feat_s2 = self.image_feature_up3(feat_s4)
                fine_pt_feat = self.fine_point(pt_feat)
               
            fine_pt_feat = torch.gather(fine_pt_feat, dim=-1, index=fps_idx.unsqueeze(1).expand(-1, fine_pt_feat.shape[1], -1)) # b x c x n
        else:
            None # no fine feature
        
        # 3.5 coarse matching 
        if self.training and self.config['feature_match']['coarse_match']:
            if self.config['feature_match']['coarse_match_type'] == 'bce_dense':
                coarse_indices = data['coarse_indices']
                coarse_pc_idx = data['coarse_pc_idx']
                coarse_patch_idx_flatten = torch.gather(sel_proj_patch_idx, dim=1, index=coarse_indices)
                coarse_out_indices = data['coarse_out_indices']
                coarse_pc_out_idx = data['coarse_pc_out_idx']
                
                # bce loss
                coarse_pc_score = self.coarse_pc_score_mlp(rearrange(voxel_feature, 'b n c -> b c n'))
                coarse_pc_score = torch.sigmoid(coarse_pc_score).squeeze(1).clamp(min=1e-7, max=1-1e-7)
                if self.config['feature_match']['bce_sample']:
                    pos_pc_score = torch.gather(coarse_pc_score, dim=1, index=coarse_indices)
                    neg_pc_score = torch.gather(coarse_pc_score, dim=1, index=coarse_out_indices)
                    gt_pos_pc_score = torch.ones_like(pos_pc_score)
                    gt_neg_pc_score = torch.zeros_like(neg_pc_score)
                    pred_pc_score = torch.concat([pos_pc_score, neg_pc_score], dim=1)
                    gt_coarse_pc_score = torch.concat([gt_pos_pc_score, gt_neg_pc_score], dim=1)
                    bce_loss = nn.BCELoss()
                    coarse_pc_loss = bce_loss(pred_pc_score, gt_coarse_pc_score)
                    data['coarse_bce_loss'] = coarse_pc_loss
                else:
                    num_patches = feat.shape[1]
                    gt_coarse_pc_score = (sel_proj_patch_idx != num_patches).to(torch.float32)
                    bce_loss = nn.BCELoss()
                    coarse_pc_loss = bce_loss(coarse_pc_score, gt_coarse_pc_score)
                    data['coarse_bce_loss'] = coarse_pc_loss
                
                # sparse coarse
                if self.config['feature_match']['bce_sparse_softmax']: # point与所有patch做softmax
                    pos_pc_feat = torch.gather(voxel_feature, dim=1, index=coarse_indices.unsqueeze(-1).expand(-1,-1,self.enc_dim))
                    self.coarse_matching_softmax(feat, pos_pc_feat, gt_conf, data, coarse_patch_idx_flatten)
                else:
                    pos_pc_feat = torch.gather(voxel_feature, dim=1, index=coarse_indices.unsqueeze(-1).expand(-1,-1,self.enc_dim))
                    pos_img_feat = torch.gather(feat, dim=1, index=coarse_patch_idx_flatten.unsqueeze(-1).expand(-1,-1,self.enc_dim))
                    
                    proj_xy = data['proj_xy'].permute(0,2,1) # B x #points
                    in_voxel_xy = torch.gather(proj_xy, dim=1, index=coarse_pc_idx.unsqueeze(-1).expand(-1,-1,2))
                    if self.config['feature_match']['coarse_intra']:
                        cat_in_voxel_xy = torch.concat([in_voxel_xy, in_voxel_xy], dim=1) 
                        dist = torch.norm(cat_in_voxel_xy.unsqueeze(1)-cat_in_voxel_xy.unsqueeze(2), dim=-1)
                        corr_mask = (dist <= self.config['feature_match']['tol_thres'] * coarse_patch_size).float()
                        descriptor_loss, _ = fm_desc_loss(pos_img_feat, pos_pc_feat, corr_mask,
                                            pos_margin=self.config['feature_match']['pos_margin'], 
                                            neg_margin=self.config['feature_match']['neg_margin'], 
                                            log_scale=self.config['feature_match']['log_scale'], 
                                            bin_score = self.bin_score, 
                                            mode="log_contrastive_intra",
                                            temperature=self.clip_temperature)
                    else:
                        dist = torch.norm(in_voxel_xy.unsqueeze(1)-in_voxel_xy.unsqueeze(2), dim=-1)
                        corr_mask = (dist <= self.config['feature_match']['tol_thres'] * coarse_patch_size).float()
                        descriptor_loss, _ = fm_desc_loss(pos_img_feat, pos_pc_feat, corr_mask,
                                            pos_margin=self.config['feature_match']['pos_margin'], 
                                            neg_margin=self.config['feature_match']['neg_margin'], 
                                            log_scale=self.config['feature_match']['log_scale'], 
                                            bin_score = self.bin_score, 
                                            mode="log_contrastive",
                                            temperature=self.clip_temperature)
                    data['coarse_matching_loss'] = descriptor_loss
                
                # dense coarse
                if self.config['feature_match']['bce_sample']:
                    sample_fps_idx = coarse_pc_idx
                    self.coarse_matching_dense(feat, pos_pc_feat, data, coarse_patch_idx_flatten, sample_fps_idx, n_H=n_H, n_W=n_W)
                else:
                    self.coarse_matching_dense(feat, voxel_feature, data, sel_proj_patch_idx, fps_idx, n_H=n_H, n_W=n_W)
            elif self.config['feature_match']['coarse_match_type'] == 'bce':
                coarse_indices = data['coarse_indices']
                coarse_pc_idx = data['coarse_pc_idx']
                coarse_patch_idx_flatten = torch.gather(sel_proj_patch_idx, dim=1, index=coarse_indices)
                coarse_out_indices = data['coarse_out_indices']
                coarse_pc_out_idx = data['coarse_pc_out_idx']
                
                # bce loss
                coarse_pc_score = self.coarse_pc_score_mlp(rearrange(voxel_feature, 'b n c -> b c n'))
                coarse_pc_score = torch.sigmoid(coarse_pc_score).squeeze(1).clamp(min=1e-7, max=1-1e-7)
                if self.config['feature_match']['bce_sample']:
                    pos_pc_score = torch.gather(coarse_pc_score, dim=1, index=coarse_indices)
                    neg_pc_score = torch.gather(coarse_pc_score, dim=1, index=coarse_out_indices)
                    gt_pos_pc_score = torch.ones_like(pos_pc_score)
                    gt_neg_pc_score = torch.zeros_like(neg_pc_score)
                    pred_pc_score = torch.concat([pos_pc_score, neg_pc_score], dim=1)
                    gt_coarse_pc_score = torch.concat([gt_pos_pc_score, gt_neg_pc_score], dim=1)
                    bce_loss = nn.BCELoss()
                    coarse_pc_loss = bce_loss(pred_pc_score, gt_coarse_pc_score)
                    data['coarse_bce_loss'] = coarse_pc_loss
                else:
                    num_patches = feat.shape[1]
                    gt_coarse_pc_score = (sel_proj_patch_idx != num_patches).to(torch.float32)
                    bce_loss = nn.BCELoss()
                    coarse_pc_loss = bce_loss(coarse_pc_score, gt_coarse_pc_score)
                    data['coarse_bce_loss'] = coarse_pc_loss
                
                # sparse coarse
                pos_pc_feat = torch.gather(voxel_feature, dim=1, index=coarse_indices.unsqueeze(-1).expand(-1,-1,self.enc_dim))
                pos_img_feat = torch.gather(feat, dim=1, index=coarse_patch_idx_flatten.unsqueeze(-1).expand(-1,-1,self.enc_dim))
                
                proj_xy = data['proj_xy'].permute(0,2,1) # B x #points
                in_voxel_xy = torch.gather(proj_xy, dim=1, index=coarse_pc_idx.unsqueeze(-1).expand(-1,-1,2))
                if self.config['feature_match']['coarse_intra']:
                    cat_in_voxel_xy = torch.concat([in_voxel_xy, in_voxel_xy], dim=1) 
                    dist = torch.norm(cat_in_voxel_xy.unsqueeze(1)-cat_in_voxel_xy.unsqueeze(2), dim=-1)
                    corr_mask = (dist <= self.config['feature_match']['tol_thres'] * coarse_patch_size).float()
                    descriptor_loss, _ = fm_desc_loss(pos_img_feat, pos_pc_feat, corr_mask,
                                        pos_margin=self.config['feature_match']['pos_margin'], 
                                        neg_margin=self.config['feature_match']['neg_margin'], 
                                        log_scale=self.config['feature_match']['log_scale'], 
                                        bin_score = self.bin_score, 
                                        mode="log_contrastive_intra",
                                        temperature=self.clip_temperature)
                else:
                    dist = torch.norm(in_voxel_xy.unsqueeze(1)-in_voxel_xy.unsqueeze(2), dim=-1)
                    corr_mask = (dist <= self.config['feature_match']['tol_thres'] * coarse_patch_size).float()
                    descriptor_loss, _ = fm_desc_loss(pos_img_feat, pos_pc_feat, corr_mask,
                                        pos_margin=self.config['feature_match']['pos_margin'], 
                                        neg_margin=self.config['feature_match']['neg_margin'], 
                                        log_scale=self.config['feature_match']['log_scale'], 
                                        bin_score = self.bin_score, 
                                        mode="log_contrastive",
                                        temperature=self.clip_temperature)
                data['coarse_matching_loss'] = descriptor_loss
            else:
                raise ValueError
            
            # coarse euclidean
            if self.config['feature_match']['coarse_euc'] and self.config['feature_match']['coarse_match_type'] in ['bce', 'bce_dense']:
                euc_pc_feat = torch.gather(crs_voxel_feature, dim=1, index=coarse_indices.unsqueeze(-1).expand(-1,-1,self.enc_dim))
                euc_img_feat = torch.gather(crs_feat, dim=1, index=coarse_patch_idx_flatten.unsqueeze(-1).expand(-1,-1,self.enc_dim))
                
                feat_dim = euc_pc_feat.shape[-1]
                euc_cat_features = torch.concat([euc_pc_feat, euc_img_feat], dim=1)
                
                euc_dist = torch.cdist(euc_cat_features, euc_cat_features, p=2) / (feat_dim ** 0.5)
                euc_dist = -euc_dist
                
                batch_size, num_sample, _ = euc_dist.shape
                eye_mask =  torch.eye(num_sample, dtype=torch.bool).cuda()
                eye_mask = eye_mask.unsqueeze(0).expand(batch_size, -1, -1)
                
                half_sample = num_sample//2
                labels = torch.concat([(torch.arange(half_sample)+half_sample), torch.arange(half_sample)], dim=0).cuda()
                labels = labels.unsqueeze(0).repeat(batch_size,1)
        
                euc_dist[eye_mask] = -float('1e5')
                euc_dist = euc_dist * torch.exp(self.euc_temperature)
                
                euc_loss_row = F.cross_entropy(euc_dist.reshape(-1, num_sample), 
                                        labels.reshape(-1), 
                                        reduction="mean")
                
                data['coarse_euc_loss'] = euc_loss_row
            
            # fine match loss
            if self.config['feature_match']['fine_match']:
                if self.config['feature_match']['coarse_match_type'] in ['dense', 'bce_dense']:
                    num_patches = feat.shape[1]
                    proj_xy_int = data['proj_xy_int']
                    proj_xy_int_s2 = proj_xy_int // 2
                    
                    if self.config['feature_match']['bce_sample']:
                        valid_mask = (coarse_patch_idx_flatten != num_patches)
                        sel_proj_xy_int_s2 = torch.gather(proj_xy_int_s2, dim=2, index=coarse_pc_idx.unsqueeze(1).repeat(1,2,1))
                        sel_proj_xy_int_s2 = sel_proj_xy_int_s2.permute(0,2,1)
                        fine_pt_feat = torch.gather(fine_pt_feat, dim=-1, index=coarse_indices.unsqueeze(1).expand(-1, fine_pt_feat.shape[1], -1)) # b c n
                    else:
                        valid_mask = (sel_proj_patch_idx != num_patches)
                        sel_proj_xy_int_s2 = torch.gather(proj_xy_int_s2, dim=2, index=fps_idx.unsqueeze(1).repeat(1,2,1))
                        sel_proj_xy_int_s2 = sel_proj_xy_int_s2.permute(0,2,1)
                    
                    patch_size = coarse_patch_size // 2
                    half_patch = patch_size // 2
                    if self.config['feature_match']['fine_patch_aug']:
                        patch_center = sel_proj_xy_int_s2
                        patch_center = patch_center + torch.randint_like(patch_center, low=-(half_patch-1), high=(half_patch+1))
                        patch_center = patch_center
                    else:
                        patch_center = sel_proj_xy_int_s2
                        patch_center = patch_center + torch.zeros_like(patch_center)
                    
                    patch_center[:,:,0] = torch.clamp(patch_center[:,:,0], min=half_patch, max=f_W*patch_size-half_patch)
                    patch_center[:,:,1] = torch.clamp(patch_center[:,:,1], min=half_patch, max=f_H*patch_size-half_patch)
                    
                    
                    fine_patch_feature = self.extract_patch(feat_s2, patch_center, size=patch_size)
                    
                    # proj_xy = data['proj_xy'] # Bx2x20480
                    # sel_proj_xy = torch.gather(proj_xy, dim=-1, index=fps_idx.unsqueeze(1).repeat(1,2,1)) # Bx2x512
                    # sel_proj_xy_s2 = sel_proj_xy / 2
                    
                    relative_coords =  sel_proj_xy_int_s2 - patch_center
                    gt_relative = relative_coords + patch_size // 2
                    corr_flatten_id = gt_relative[:,:,1] * patch_size + gt_relative[:,:,0]
                    
                    debug_rel_coords = relative_coords * valid_mask.unsqueeze(-1)
                    corr_flatten_id = corr_flatten_id * valid_mask
                    corr_flatten_id = corr_flatten_id.to(torch.int64)
                    
                    valid_mask = rearrange(valid_mask, 'b n -> (b n)')
                    gt_relative = rearrange(gt_relative, 'b n c -> (b n) c')
                    corr_flatten_id = rearrange(corr_flatten_id, 'b n -> (b n)')
                    fine_pt_feat = rearrange(fine_pt_feat, 'b c n -> (b n) c')
                    fine_patch_feature = rearrange(fine_patch_feature, 'b n c h w -> (b n) (h w) c')
                    
                    fine_pt_feat, fine_patch_feature
                    
                    if self.config['feature_match']['fine_softmax']:
                        fine_softmax_loss = self.fine_matching_softmax(fine_patch_feature,fine_pt_feat, corr_flatten_id, valid_mask)
                        
                        data['fine_softmax_loss'] = fine_softmax_loss
                    
                    if self.config['feature_match']['fine_loss_type'] in ['log_contrastive', 'circle']:
                        fine_loss =  fine_circle_loss(fine_patch_feature,fine_pt_feat, corr_flatten_id, valid_mask, 
                                                patch_size=patch_size, 
                                                fine_norm_type=self.config['feature_match']['fine_match_norm'],
                                                mode=self.config['feature_match']['fine_loss_type'])
                        data['fine_matching_loss'] = fine_loss
                        
                    elif self.config['feature_match']['fine_loss_type'] == 'dense':
                        if self.config['feature_match']['fine_match_norm'] == "dim_norm":
                            # dim normalize
                            fine_patch_feature, fine_pt_feat = map(lambda feature: feature / feature.shape[-1]**.5,
                                                    [fine_patch_feature, fine_pt_feat])
                        elif self.config['feature_match']['fine_match_norm'] == "l2_norm":
                            # cos similarity normalize
                            fine_patch_feature = F.normalize(fine_patch_feature, p=2, dim=-1) 
                            fine_pt_feat = F.normalize(fine_pt_feat, p=2, dim=-1) #
                        elif self.config['feature_match']['fine_match_norm'] == "no_norm":
                            fine_patch_feature = fine_patch_feature
                            fine_pt_feat = fine_pt_feat
                        else:
                            raise ValueError
        
                        fine_sim_matrix = torch.einsum("nc,nlc->nl", fine_pt_feat, fine_patch_feature)
                        fine_sim_matrix = fine_sim_matrix * self.fine_dense_temperature 
                        fine_sim_matrix = torch.softmax(fine_sim_matrix, dim=-1) # 最终softmax之后的matrix
                        fine_sim_matrix = fine_sim_matrix.reshape(-1, patch_size, patch_size)
                        
                        # take soft argmax
                        x_grid = torch.arange(0, patch_size).view(1,1,-1).cuda()
                        y_grid = torch.arange(0, patch_size).view(1,-1,1).cuda()
                        
                        soft_x = torch.sum(x_grid * fine_sim_matrix, dim=[-2,-1]) / torch.sum(fine_sim_matrix, dim=[-2,-1])
                        soft_y = torch.sum(y_grid * fine_sim_matrix, dim=[-2,-1]) / torch.sum(fine_sim_matrix, dim=[-2,-1])
                        
                        pred_relative = torch.stack([soft_x, soft_y], dim=-1)
                        gt_relative
                        valid_mask
                        
                        fine_loss = torch.norm(pred_relative - gt_relative, dim=-1) * valid_mask
                        fine_loss = fine_loss.sum() / valid_mask.sum()
                        
                        data['fine_matching_loss'] = fine_loss
                    elif self.config['feature_match']['fine_loss_type'] == 'fine_softmax':
                        fine_loss = self.fine_matching_softmax(fine_patch_feature,fine_pt_feat, corr_flatten_id, valid_mask)
                        
                        data['fine_matching_loss'] = fine_loss
                    else:
                        raise ValueError
                    
                elif self.config['feature_match']['coarse_match_type'] in ['bce']:
                    num_patches = feat.shape[1]

                    proj_xy_int = data['proj_xy_int']
                    proj_xy_int_s2 = proj_xy_int // 2
                    proj_patch = data['proj_patch'] # B x #points
                    
                    if self.config['feature_match']['bce_sample']:
                        valid_mask = (coarse_patch_idx_flatten != num_patches)
                        sel_proj_xy_int_s2 = torch.gather(proj_xy_int_s2, dim=2, index=coarse_pc_idx.unsqueeze(1).repeat(1,2,1))
                        sel_proj_xy_int_s2 = sel_proj_xy_int_s2.permute(0,2,1)
                        sel_proj_patch = torch.gather(proj_patch, dim=1, index=coarse_pc_idx.unsqueeze(-1).repeat(1,1,2))
                        fine_pt_feat = torch.gather(fine_pt_feat, dim=-1, index=coarse_indices.unsqueeze(1).expand(-1, fine_pt_feat.shape[1], -1)) # b c n
                    else:
                        valid_mask = (sel_proj_patch_idx != num_patches)
                        sel_proj_xy_int_s2 = torch.gather(proj_xy_int_s2, dim=2, index=fps_idx.unsqueeze(1).repeat(1,2,1))
                        sel_proj_xy_int_s2 = sel_proj_xy_int_s2.permute(0,2,1)
                        sel_proj_patch = torch.gather(proj_patch, dim=1, index=fps_idx.unsqueeze(-1).repeat(1,1,2))
                        
                    patch_size = coarse_patch_size // 2
                    patch_center = sel_proj_patch * patch_size + patch_size//2
                    fine_patch_feature = self.extract_patch(feat_s2, patch_center, size=patch_size)
                    
                    relative_coords =  sel_proj_xy_int_s2 - patch_center
                    corr_flatten_id = relative_coords + patch_size // 2
                    corr_flatten_id = corr_flatten_id[:,:,1] * patch_size + corr_flatten_id[:,:,0]
                    
                    debug_rel_coords = relative_coords * valid_mask.unsqueeze(-1)
                    corr_flatten_id = corr_flatten_id * valid_mask
                    
                    valid_mask = rearrange(valid_mask, 'b n -> (b n)')
                    corr_flatten_id = rearrange(corr_flatten_id, 'b n -> (b n)')
                    fine_pt_feat = rearrange(fine_pt_feat, 'b c n -> (b n) c')
                    fine_patch_feature = rearrange(fine_patch_feature, 'b n c h w -> (b n) (h w) c')
                    
                    fine_pt_feat, fine_patch_feature
                    
                    fine_loss =  fine_circle_loss(fine_patch_feature,fine_pt_feat, corr_flatten_id, valid_mask, 
                                                patch_size=patch_size, 
                                                fine_norm_type=self.config['feature_match']['fine_match_norm'],
                                                mode=self.config['feature_match']['fine_loss_type'])
                    
                    data['fine_matching_loss'] = fine_loss

                
        if not self.training and self.config['feature_match']['coarse_match']:
            proj_patch_idx = data['proj_patch_idx'] # B x #points
            proj_patch = data['proj_patch'] # B x #points
            num_node = fps_idx.shape[1]
            num_patch = n_H * n_W
            sel_proj_patch_idx = torch.gather(proj_patch_idx, dim=1, index=fps_idx)
            sel_proj_patch = torch.gather(proj_patch, dim=1, index=fps_idx.unsqueeze(-1).repeat(1,1,2))
            
            
            N, L, S, C = voxel_feature.size(0), voxel_feature.size(1), feat.size(1), voxel_feature.size(2)
            
            if self.config['feature_match']['coarse_match_type'] in ['bce', 'bce_dense']:
                coarse_pc_score = self.coarse_pc_score_mlp(rearrange(voxel_feature, 'b n c -> b c n'))
                coarse_pc_score = torch.sigmoid(coarse_pc_score).squeeze(1).clamp(min=1e-7, max=1-1e-7)
                filter_mask = coarse_pc_score > 0.9 # 0.9
                if filter_mask.sum() < 20:
                    topk_values, topk_indices = torch.topk(coarse_pc_score, 20)
                    filter_mask = torch.zeros_like(coarse_pc_score, dtype=torch.bool)
                    filter_mask[0, topk_indices] = True
                
            if self.config['feature_match']['coarse_match_norm'] == "dim_norm":
                # dim normalize
                feat, voxel_feature = map(lambda feature: feature / feature.shape[-1]**.5,
                                        [feat, voxel_feature])
            elif self.config['feature_match']['coarse_match_norm'] == "l2_norm":
                # cos similarity normalize
                feat = F.normalize(feat, p=2, dim=-1) # b n c
                voxel_feature = F.normalize(voxel_feature, p=2, dim=-1) # b n c
                        
                    
            elif self.config['feature_match']['coarse_match_norm'] == "no_norm":
                feat = feat
                voxel_feature = voxel_feature
            else:
                raise ValueError
            
            if self.config['feature_match']['coarse_match_type'] in ['dense', 'bce', 'bce_dense']:
                sim_matrix = torch.einsum("nlc,nsc->nls", voxel_feature, feat)
                
                if self.config['feature_match']['coarse_match_type'] == 'dense':
                    # add image dustbin embedding
                    bins0 = self.bin_score.expand(batch_size, L, 1)
                    sim_matrix = torch.concat([sim_matrix, bins0], dim=-1)
                    
                    filter_matrix = torch.softmax(sim_matrix * self.temperature, dim=-1)
                    
                    filter0 = (filter_matrix.max(dim=2)[1] != S)  # [N, L]
                    filter1 = (filter_matrix.max(dim=2)[0] > 0.2)  # [N, L]
                    filter_mask = filter0 * filter1
                    
                    if self.config['feature_match']['dense_temp']:
                        sim_matrix = sim_matrix[filter_mask][:,:-1]
                        sim_matrix = sim_matrix * self.dense_temperature
                        sim_matrix = torch.softmax(sim_matrix, dim=-1) # [L, nH*nW]
                    else:
                        sim_matrix = sim_matrix * self.temperature
                        sim_matrix = torch.softmax(sim_matrix, dim=-1)
                        sim_matrix = sim_matrix[filter_mask][:,:-1] # [L, nH*nW]
                        
                elif self.config['feature_match']['coarse_match_type'] == 'bce_dense':
                    if self.config['feature_match']['dense_temp']:
                        sim_matrix = sim_matrix[filter_mask]
                        sim_matrix = sim_matrix * self.dense_temperature
                        sim_matrix = torch.softmax(sim_matrix, dim=-1) # [L, nH*nW]
                    else:
                        sim_matrix = sim_matrix * self.temperature
                        sim_matrix = torch.softmax(sim_matrix, dim=-1)
                        sim_matrix = sim_matrix[filter_mask] # [L, nH*nW]
                elif self.config['feature_match']['coarse_match_type'] == 'bce':
                    # score_filter = torch.max(sim_matrix, dim=-1)[0] > 0.2
                    # filter_mask = filter_mask * score_filter raw_att_p2i
                    sim_matrix = sim_matrix[filter_mask]
                    # raw_att_p2i = raw_att_p2i[filter_mask]
                else:
                    None
                
                matched_3d = center_points[filter_mask]
                # raw_att_p2i = raw_att_p2i[filter_mask]
                # raw_att_i2p = raw_att_i2p[0]
 
                if self.config['feature_match']['coarse_match_type'] == 'bce':
                    max_index_flatten = torch.argmax(sim_matrix, dim=-1)
                    max_index_x = max_index_flatten % n_W
                    max_index_y = max_index_flatten // n_W
                    matched_patch = torch.stack([max_index_x, max_index_y], dim=1)
                elif not self.config['feature_match']['coarse_match_type'] == 'bce':
                    # zero out the corr_map outside the window
                    max_index_flatten = torch.argmax(sim_matrix, dim=-1)
                    max_index_x = max_index_flatten % n_W
                    max_index_y = max_index_flatten // n_W
                    sim_matrix = sim_matrix.reshape(-1, n_H, n_W) # [L, nH, nW]
                    
                    window_mask = torch.zeros_like(sim_matrix)   
                    window_size = 2
                    for i in range(-window_size, window_size+1):
                        for j in range(-window_size, window_size+1):
                            # make sure the index is within the range
                            clamped_y = torch.clamp(max_index_y+i, 0, n_H-1)   # (H_s * W_s, )
                            clamped_x = torch.clamp(max_index_x+j, 0, n_W-1)   # (H_s * W_s, )
                            window_mask[torch.arange(sim_matrix.shape[0]), clamped_y, clamped_x] = 1
                    
                    sim_matrix = sim_matrix * window_mask

                    
                    # take soft argmax
                    x_grid = torch.arange(0, n_W).view(1,1,-1).cuda()
                    y_grid = torch.arange(0, n_H).view(1,-1,1).cuda()
                    
                    soft_x = torch.sum(x_grid * sim_matrix, dim=[1,2]) / torch.sum(sim_matrix, dim=[1,2])
                    soft_y = torch.sum(y_grid * sim_matrix, dim=[1,2]) / torch.sum(sim_matrix, dim=[1,2])
                    
                    matched_patch_sparse = torch.stack([max_index_x, max_index_y], dim=1)
                    matched_patch = torch.stack([soft_x, soft_y], dim=1)
                
                # dense_fine_match = False
                if self.config['feature_match']['fine_match'] and matched_patch.shape[0] > 0:
                    patch_size = coarse_patch_size // 2
                    patch_center = matched_patch * patch_size + patch_size//2
                    patch_center = patch_center.to(torch.int32)
                    fine_patch_feature = self.extract_patch(feat_s2, patch_center.unsqueeze(0), 
                                                            size=patch_size)
                    fine_patch_feature = rearrange(fine_patch_feature, 'b n c h w -> (b n) (h w) c')
                    fine_point_feature = fine_pt_feat.permute(0,2,1)[filter_mask]
                    
                    if self.config['feature_match']['fine_match_norm'] == "dim_norm":
                        # dim normalize
                        fine_patch_feature, fine_point_feature = map(lambda feature: feature / feature.shape[-1]**.5,
                                                [fine_patch_feature, fine_point_feature])
                    elif self.config['feature_match']['fine_match_norm'] == "l2_norm":
                        # cos similarity normalize
                        fine_patch_feature = F.normalize(fine_patch_feature, p=2, dim=-1) 
                        fine_point_feature = F.normalize(fine_point_feature, p=2, dim=-1) #
                    elif self.config['feature_match']['fine_match_norm'] == "no_norm":
                        fine_patch_feature = fine_patch_feature
                        fine_point_feature = fine_point_feature
                    else:
                        raise ValueError
                    
                    if self.config['feature_match']['fine_loss_type'] in ['log_contrastive', 'fine_softmax', 'circle']:
                        fine_sim_matrix = torch.einsum("nc,nlc->nl", fine_point_feature, fine_patch_feature) # N x size^2
                        delta_xy = fine_sim_matrix.argmax(dim=-1) 
                        delta_xy = torch.stack(
                                [delta_xy % patch_size, delta_xy // patch_size],
                                dim=1)
                        delta_xy = delta_xy - patch_size//2
                        
                        # pred xy
                        matched_xy = (patch_center + delta_xy) * 2 + 0.5
                        # gt xy
                        # proj_xy_int = data['proj_xy_int']
                        # sel_proj_xy_int = torch.gather(proj_xy_int, dim=2, index=fps_idx.unsqueeze(1).repeat(1,2,1))
                        # sel_proj_xy_int = sel_proj_xy_int[0].T
                        # sel_proj_xy_int = sel_proj_xy_int[i_ids]
                        coords = matched_3d.unsqueeze(0) * data['voxel_scalar'][:,None,None] + data['voxel_mean'][:,None,:]


                    elif self.config['feature_match']['fine_loss_type'] == 'dense':
                        fine_sim_matrix = torch.einsum("nc,nlc->nl", fine_point_feature, fine_patch_feature) # N x size^2
                        fine_sim_matrix = fine_sim_matrix * self.fine_dense_temperature # 只取最大值，因此乘哪个temp对取值没影响，直接用dense的temp，后面要计算soft argmax
                        fine_sim_matrix = torch.softmax(fine_sim_matrix, dim=-1) # 最终softmax之后的matrix
                        
                        # zero out the corr_map outside the window
                        max_index_flatten = torch.argmax(fine_sim_matrix, dim=-1) 
                        max_index_x = max_index_flatten % patch_size
                        max_index_y = max_index_flatten // patch_size
                        fine_sim_matrix = fine_sim_matrix.reshape(-1, patch_size, patch_size) # [L, nH, nW]
                        
                        # TODO: whether USE FINE WINDOW
                        window_mask = torch.zeros_like(fine_sim_matrix)   
                        window_size = 2
                        for i in range(-window_size, window_size+1):
                            for j in range(-window_size, window_size+1):
                                # make sure the index is within the range
                                clamped_y = torch.clamp(max_index_y+i, 0, patch_size-1)   # (H_s * W_s, )
                                clamped_x = torch.clamp(max_index_x+j, 0, patch_size-1)   # (H_s * W_s, )
                                window_mask[torch.arange(fine_sim_matrix.shape[0]), clamped_y, clamped_x] = 1
                        fine_sim_matrix = fine_sim_matrix * window_mask


                        # take soft argmax
                        x_grid = torch.arange(0, patch_size).view(1,1,-1).cuda()
                        y_grid = torch.arange(0, patch_size).view(1,-1,1).cuda()
                        
                        soft_x = torch.sum(x_grid * fine_sim_matrix, dim=[-2,-1]) / torch.sum(fine_sim_matrix, dim=[-2,-1])
                        soft_y = torch.sum(y_grid * fine_sim_matrix, dim=[-2,-1]) / torch.sum(fine_sim_matrix, dim=[-2,-1])
                        
                        pred_relative = torch.stack([soft_x, soft_y], dim=-1)
                        pred_relative = pred_relative - patch_size//2
                        
                        matched_xy = patch_center + pred_relative
                        matched_xy = matched_xy * 2 + 0.5
                        coords = matched_3d.unsqueeze(0) * data['voxel_scalar'][:,None,None] + data['voxel_mean'][:,None,:]
                    
                    
                else:
                    matched_xy = (matched_patch + 0.5) * coarse_patch_size - 0.5
                    coords = matched_3d.unsqueeze(0) * data['voxel_scalar'][:,None,None] + data['voxel_mean'][:,None,:]
                
    
                data['xy_sample']=matched_xy.unsqueeze(0) # B x M x 2
                data['coords'] = coords # B x M x 3
                data['coords_gt_sample'] = coords
                data['scores_logit']=torch.ones_like(coords)[:,:,0]
                data['mask_sample']=torch.ones_like(coords)[:,:,0].to(torch.bool)
                
            else:
                None

    def forward(self, data):
        
        if self.training:
            self.train_forward(data)
        else:
            coords_list=[]
            scores_list=[]
            xy_sample_list=[]
            mask_list=[]
            coords_gt=[]
            with torch.no_grad(): # extract image features
                if self.config['feature_match']['dust_backbone']:
                    if self.config['feature_match']['dust_with_resnet']:
                        feat_set=self.image_backbone_res(data['image']) # 1/8 resolution
                        feat_s16=feat_set[-3] #256
                        feat_s8=feat_set[-4]  #128
                        feat_s4=feat_set[-5]  #64
                        feat_s2=feat_set[-6]  #64
                        patch_embedding = self.patch_enc(rearrange(feat_s16,'b c h w -> b c (h w)'))
                        patch_embedding = rearrange(patch_embedding,'b c n -> b n c')
                    else:
                        patch_embedding = None
                    feat=self.image_backbone(data['image'],patch_embedding) # BHWC
                    feat=rearrange(feat,'b h w c -> b c (h w)')
                    feat=feat.contiguous()
                    feat=self.encoder_embed_img(feat) # BC(HW)
                else:
                    feat=self.image_backbone(data['image']) # 1/8 resolution
                
                # predict coordiantes voxel by voxel
                for idx, voxel_id in enumerate(data['voxel_id']): 
                    data_copy=copy.deepcopy(data)
                    data_copy['feat']=feat
                    
                    if self.config['feature_match']['new_pipeline']:
                        None
                    else:
                        data_copy['voxel_id']=voxel_id
                        data_copy['voxel_points']=data['voxel_points'][idx]
                        data_copy['voxel_scalar']=data['voxel_scalar'][idx]
                        data_copy['voxel_mean']=data['voxel_mean'][idx]
                        
                        data_copy['proj_patch_idx']=data['proj_patch_idx'][idx]
                        data_copy['proj_patch']=data['proj_patch'][idx]
                        data_copy['proj_xy_int']=data['proj_xy_int'][idx]
                        
                        data_copy['voxel_nodes'] = data['voxel_nodes'][idx]
                        data_copy['point2node'] = data['point2node'][idx]
                        data_copy['node2point'] = data['node2point'][idx]
                    
                    if self.config['feature_match']['dust_with_resnet'] and self.config['feature_match']['dust_backbone']:
                        data_copy['feat_s8'] = feat_s8
                        data_copy['feat_s4'] = feat_s4
                        data_copy['feat_s2'] = feat_s2
                    
                    
                    self.train_forward(data_copy)
                    coords_list.append(data_copy['coords'])
                    scores_list.append(data_copy['scores_logit'])
                    xy_sample_list.append(data_copy['xy_sample'])
                    mask_list.append(data_copy['mask_sample'])
                    coords_gt.append(data_copy['coords_gt_sample'])
                    
            data['coords_list']=coords_list
            data['scores_list']=[torch.sigmoid(scores) for scores in scores_list]
            data['xy_list']=xy_sample_list
            data['mask_sample_list'] = mask_list
            data['coords_gt_list'] = coords_gt
            
            data['coords']=torch.cat(coords_list, dim=1) # coordinate prediction
            data['scores_logit']=torch.cat(scores_list, dim=1) # score prediction
            data['xy_sample']=torch.cat(xy_sample_list, dim=1) # pixel coordinate 
            data['mask_sample']=torch.cat(mask_list, dim=1)
            data['coords_gt_sample']=torch.cat(coords_gt, dim=1)
            
            data['scores']=torch.sigmoid(data['scores_logit'])
        