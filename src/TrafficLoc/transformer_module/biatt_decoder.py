# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Main encoder/decoder blocks
# --------------------------------------------------------
# References: 
# timm
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/helpers.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/mlp.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py


import torch
import torch.nn as nn 

from itertools import repeat
import collections.abc

from copy import deepcopy
import numpy as np
import math

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class RoPE2D(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq 
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D,seq_len,device,dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos() # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D,seq_len,device,dtype] = (cos,sin)
        return self.cache[D,seq_len,device,dtype]
        
    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim==2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)
        
    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        assert tokens.size(-1)%2==0, "number of dimensions should be a multiple of two"
        D = tokens.size(-1) // 2
        assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2
        cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
        # split features into two along the feature dimension, and apply rope1d on each half
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:,:,0], cos, sin)
        x = self.apply_rope1d(x, positions[:,:,1], cos, sin)
        tokens = torch.cat((y, x), dim=-1)
        return tokens

class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=3,
        gauss_scale=1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        # if self.normalize:
        #     xyz = shift_scale_points(xyz, src_range=input_range)

        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert (
            ndim % 2 == 0
        ), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack(
                (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
            ).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        # if self.normalize:
        #     xyz = shift_scale_points(xyz, src_range=input_range)

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # original: return batch x d_pos x npoints embedding
        # final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        
        # modified: return batch x npoints x d_pos, e.g. 4x64x256
        final_embeds = torch.cat(final_embeds, dim=2)
        
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                return self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                return self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown {self.pos_type}")

    def extra_repr(self):
        st = f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}"
        if hasattr(self, "gauss_B"):
            st += (
                f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
            )
        return st
    
class PointPE(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        self.encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(128, self.out_dim),
        )

    def forward(self, pos):
        # pos is [B, N, 3]
        # output pe is [B, N, self.out_dim]
        return self.encoder(pos)
    
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1,3)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        # q,k,v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)
               
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, query, key, value):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]
        
        q = self.projq(query).reshape(B,Nq,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B,Nk,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B,Nv,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
            
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True):
        super().__init__()
        
        # pe for img and pcd
        self.pe_img = RoPE2D(freq=100, F0=1)
        self.pe_pcd = PointPE(out_dim=dim)
        # self.pe_pcd = PositionEmbeddingCoordsSine(
        #     d_pos=dim, pos_type="fourier", normalize=False
        # )
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, xpos, ypos, mode=None, new_pe=False):
        # mode is used to specify the self-att modality
        assert mode == 'img' or mode == 'pcd'
        
        # self-attention
        x = self.norm1(x)
        if new_pe:
            # x = x + xpos
            if mode == 'img':
                if xpos.shape[-1] == x.shape[-1]:
                    x = x + xpos
                else:
                    x = self.pe_img(x, xpos)
            elif mode == 'pcd':
                x = x + xpos
        else:
            if mode == 'img':
                x = self.pe_img(x, xpos)
            elif mode == 'pcd':
                x = x + self.pe_pcd(xpos)
        x = x + self.drop_path(self.attn(x))
        
        # cross-attention
        x = self.norm2(x)
        y_ = self.norm_y(y)
        if new_pe:
            # x = x + xpos
            # y_ = y_ + ypos
            if mode == 'img':
                if xpos.shape[-1] == x.shape[-1]:
                    x = x + xpos
                else:
                    x = self.pe_img(x, xpos)
                y_ = y_ + ypos
            elif mode == 'pcd':
                x = x + xpos
                if ypos.shape[-1] == y_.shape[-1]:
                    y_ = y_ + ypos
                else:
                    y_ = self.pe_img(y_, ypos)
        else:
            if mode == 'img':
                x = self.pe_img(x, xpos)
                y_ = y_ + self.pe_pcd(ypos)
            elif mode == 'pcd':
                x = x + self.pe_pcd(xpos)
                y_ = self.pe_img(y_, ypos)
        x = x + self.drop_path(self.cross_attn(x, y_, y_))
        
        # mlp
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        
        return x, y

class DecoderBlockQKPos(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True):
        super().__init__()
        
        # pe for img and pcd
        self.pe_img = RoPE2D(freq=100, F0=1)
        self.pe_pcd = PointPE(out_dim=dim)
        # self.pe_pcd = PositionEmbeddingCoordsSine(
        #     d_pos=dim, pos_type="fourier", normalize=False
        # )
        
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, xpos, ypos, mode=None, new_pe=False):
        # self-attention
        x2 = self.norm1(x)
        x = x + self.drop_path(self.attn(x2+xpos, x2+xpos, x2))
        
        # cross-attention
        x2 = self.norm2(x)
        y2 = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(x2+xpos, y2+ypos, y2))
        
        # mlp
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        
        return x, y
    
class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos
    
if __name__ == "__main__":
    dec_embed_dim = 256
    dec_num_heads = 4
    dec_depth = 6
    img_pos_generator = PositionGetter()
    
    dec_blocks = nn.ModuleList([
            DecoderBlock(dim=dec_embed_dim,
                             num_heads=dec_num_heads,
                             mlp_ratio=4,
                             qkv_bias=True,
                             drop=0,
                             attn_drop=0,
                             drop_path=0,
                             act_layer=nn.ReLU,
                             norm_layer=nn.LayerNorm,
                             norm_mem=True)
            for i in range(dec_depth)])
    dec_blocks2 = deepcopy(dec_blocks)
    dec_norm = nn.LayerNorm(dec_embed_dim)
    
    img_token = torch.randn(4,576,256)
    n_H = 18
    n_W = 32
    img_pos = img_pos_generator(4, n_H, n_W, img_token.device)
    # img_pos = torch.randint(0, 20, (4,576,2))
    
    pcd_token = torch.randn(4,64,256)
    pcd_pos = torch.randn(4,64,3)
    
    final_output = [(img_token, pcd_token)]
    
    for blk1, blk2 in zip(dec_blocks, dec_blocks2): # dec_block2和dec_blocks结构相同，但不share weight
        # img1 side
        f1, _ = blk1(*final_output[-1][::+1], img_pos, pcd_pos, mode='img')
        # img2 side
        f2, _ = blk2(*final_output[-1][::-1], pcd_pos, img_pos, mode='pcd')
        # store the result
        final_output.append((f1, f2))
    
    final_output[-1] = tuple(map(dec_norm, final_output[-1]))
    print(final_output)

