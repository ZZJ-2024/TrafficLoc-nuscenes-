import torch
import numpy as np
import torch.nn.functional as F
import math

def fine_circle_loss(fine_img_feature,fine_pc_feature, relative_index, valid_mask, patch_size, fine_norm_type, mode='circle'):
    if mode == 'log_contrastive':
        log_scale = 10
        pos_margin = 0.2
        neg_margin = 1.8
        num_kpt = fine_img_feature.shape[0]
        
        if fine_norm_type == "dim_norm":
            # dim normalize
            fine_img_feature, fine_pc_feature = map(lambda feature: feature / feature.shape[-1]**.5,
                                    [fine_img_feature, fine_pc_feature])
        elif fine_norm_type == "l2_norm":
            # cos similarity normalize
            fine_img_feature = F.normalize(fine_img_feature, p=2, dim=-1) 
            fine_pc_feature = F.normalize(fine_pc_feature, p=2, dim=-1) #
        elif fine_norm_type == "no_norm":
            fine_img_feature = fine_img_feature
            fine_pc_feature = fine_pc_feature
        else:
            raise ValueError
        
        sim_matrix = torch.einsum("nc,nlc->nl", fine_pc_feature, fine_img_feature)
        dists = 1-sim_matrix
        
        label = torch.zeros(num_kpt, patch_size*patch_size).cuda()
        index = torch.arange(0, num_kpt, 1).cuda()
        label[index, relative_index] = 1
        
        pos_mask=label
        neg_mask=1-label
        
        pos=dists-1e5*neg_mask
        pos_weight=(pos-pos_margin).detach()
        pos_weight=torch.max(torch.zeros_like(pos_weight),pos_weight)
        lse_positive_row=torch.logsumexp(log_scale*(pos-pos_margin)*pos_weight,dim=-1)
        # lse_positive_col=torch.logsumexp(log_scale*(pos-pos_margin)*pos_weight,dim=-2)

        neg=dists+1e5*pos_mask
        neg_weight=(neg_margin-neg).detach()
        neg_weight=torch.max(torch.zeros_like(neg_weight),neg_weight)
        lse_negative_row=torch.logsumexp(log_scale*(neg_margin-neg)*neg_weight,dim=-1)
        # lse_negative_col=torch.logsumexp(log_scale*(neg_margin-neg)*neg_weight,dim=-2)

        loss_col=F.softplus(lse_positive_row+lse_negative_row)/log_scale
        # loss_row=F.softplus(lse_positive_col+lse_negative_col)/log_scale
        # loss=loss_col+loss_row
        loss = loss_col
        
        return torch.mean(loss)
    
    elif mode == 'circle':
        m = 0.2
        gamma = 5
        num_kpt = fine_img_feature.shape[0]

        if fine_norm_type == "dim_norm":
            # dim normalize
            fine_img_feature, fine_pc_feature = map(lambda feature: feature / feature.shape[-1]**.5,
                                    [fine_img_feature, fine_pc_feature])
        elif fine_norm_type == "l2_norm":
            # cos similarity normalize
            fine_img_feature = F.normalize(fine_img_feature, p=2, dim=-1) 
            fine_pc_feature = F.normalize(fine_pc_feature, p=2, dim=-1) #
        elif fine_norm_type == "no_norm":
            fine_img_feature = fine_img_feature
            fine_pc_feature = fine_pc_feature
        else:
            raise ValueError
        
        sim_matrix = torch.einsum("nc,nlc->nl", fine_pc_feature, fine_img_feature)
        # dist = 1-sim_matrix
        dist = sim_matrix
        
        label = torch.zeros(num_kpt, patch_size*patch_size).cuda()
        index = torch.arange(0, num_kpt, 1).cuda()
        label[index, relative_index] = 1
        
        dist = torch.squeeze(dist)
        pos = label
        neg = 1 - label
        sp = dist * pos
        sn = dist * neg
        ap = torch.relu(-sp.detach() + pos + pos * m)
        an = torch.relu(sn.detach() + neg * m)
        delta_p = 1 - m
        delta_n = m

        logit_p = - ap * (sp - pos * delta_p) * gamma
        logit_n = an * (sn - neg * delta_n) * gamma

        loss_p = torch.sum(torch.exp(logit_p) * pos, dim=-1)
        loss_n = torch.sum(torch.exp(logit_n) * neg, dim=-1)
        
        loss_pn = torch.log(1 + loss_n * loss_p) * valid_mask
        loss = loss_pn.sum() / valid_mask.sum()

        return loss
        
    
def fm_desc_loss(img_features,pc_features,mask,pos_margin=0.2,neg_margin=1.8,log_scale=10,
                 bin_score=None,mode='contrastive',temperature=None):
    if mode == "clip":
        pc_features = F.normalize(pc_features, p=2, dim=-1) # B x #kpt x feat_dim
        img_features = F.normalize(img_features, p=2, dim=-1) # B x #kpt x feat_dim  
    
        cos_sim = torch.bmm(pc_features, img_features.transpose(1,2))
        
        batch_size, num_sample, _ = cos_sim.shape

        labels = torch.arange(num_sample).cuda()
        labels = labels.unsqueeze(0).repeat(batch_size,1)
 

        cos_sim = cos_sim * torch.exp(temperature)
        
        loss_row = F.cross_entropy(cos_sim.reshape(-1, num_sample), 
                                   labels.reshape(-1), 
                                   reduction="mean")
        
        loss_col = F.cross_entropy(cos_sim.permute(0,2,1).reshape(-1, num_sample), 
                                   labels.reshape(-1), 
                                   reduction="mean")
        
        loss = (loss_row + loss_col) / 2
        return loss, _
    
    elif mode == 'contrastive':
        batch_size, num_kpt, des_dim = img_features.shape
            
        # similarity score (point to pixel)
        # cos_sim = F.cosine_similarity(pc_features.unsqueeze(2), img_features.unsqueeze(1), dim=-1)
        pc_features = F.normalize(pc_features, p=2, dim=-1) # B x #kpt x feat_dim
        img_features = F.normalize(img_features, p=2, dim=-1) # B x #kpt x feat_dim
        cos_sim = torch.bmm(pc_features, img_features.transpose(1,2)) 
        dists = 1-cos_sim
        
        # positive pair mask
        positive_mask = torch.eye(num_kpt).unsqueeze(0).repeat(batch_size,1,1).cuda()
        
        # negative pair mask
        dists_prime = dists.detach() + mask * 1e5
        values, indices = (dists_prime).topk(1, dim=2, largest=False, sorted=True) # 找距离最小的negative anchor
        negative_mask = torch.zeros_like(dists_prime, dtype=torch.bool)

        # negative pair mask
        negative_mask.scatter_(2, indices, True)
        
        # positive pair loss
        positive_loss = (dists - pos_margin) * positive_mask
        positive_loss = positive_loss.sum() / (positive_mask.sum() + 1e-9)
        
        # negative pair loss
        negative_loss = (neg_margin - dists) * negative_mask
        negative_loss = negative_loss.sum() / (negative_mask.sum() + 1e-9)
        
        # print(f"positive loss: {positive_loss}")
        # print(f"negative loss: {negative_loss}")
        
        return torch.mean(positive_loss + negative_loss), dists
    elif mode == 'log_contrastive':
        pos_mask=mask
        neg_mask=1-mask
        
        # similarity score (point to pixel)
        # cos_sim = F.cosine_similarity(pc_features.unsqueeze(2), img_features.unsqueeze(1), dim=-1)
        pc_features = F.normalize(pc_features, p=2, dim=-1) # B x #kpt x feat_dim
        img_features = F.normalize(img_features, p=2, dim=-1) # B x #kpt x feat_dim
        cos_sim = torch.bmm(pc_features, img_features.transpose(1,2)) 
        dists = 1-cos_sim
        
        pos=dists-1e5*neg_mask
        pos_weight=(pos-pos_margin).detach()
        pos_weight=torch.max(torch.zeros_like(pos_weight),pos_weight)

        lse_positive_row=torch.logsumexp(log_scale*(pos-pos_margin)*pos_weight,dim=-1)
        lse_positive_col=torch.logsumexp(log_scale*(pos-pos_margin)*pos_weight,dim=-2)

        neg=dists+1e5*pos_mask
        neg_weight=(neg_margin-neg).detach()
        neg_weight=torch.max(torch.zeros_like(neg_weight),neg_weight)

        lse_negative_row=torch.logsumexp(log_scale*(neg_margin-neg)*neg_weight,dim=-1)
        lse_negative_col=torch.logsumexp(log_scale*(neg_margin-neg)*neg_weight,dim=-2)

        loss_col=F.softplus(lse_positive_row+lse_negative_row)/log_scale
        loss_row=F.softplus(lse_positive_col+lse_negative_col)/log_scale
        loss=loss_col+loss_row
        
        return torch.mean(loss),dists
    elif mode == 'log_contrastive_intra':
        pos_mask=mask
        neg_mask=1-mask
        
        # similarity score (point to pixel)
        # cos_sim = F.cosine_similarity(pc_features.unsqueeze(2), img_features.unsqueeze(1), dim=-1)
        pc_features = F.normalize(pc_features, p=2, dim=-1) # B x #kpt x feat_dim
        img_features = F.normalize(img_features, p=2, dim=-1) # B x #kpt x feat_dim
        
        cat_features = torch.concat([pc_features, img_features], dim=1)
        
        cos_sim = torch.bmm(cat_features, cat_features.transpose(1,2)) 
        dists = 1-cos_sim
        
        pos=dists-1e5*neg_mask
        pos_weight=(pos-pos_margin).detach()
        pos_weight=torch.max(torch.zeros_like(pos_weight),pos_weight)

        lse_positive_row=torch.logsumexp(log_scale*(pos-pos_margin)*pos_weight,dim=-1)
        lse_positive_col=torch.logsumexp(log_scale*(pos-pos_margin)*pos_weight,dim=-2)

        neg=dists+1e5*pos_mask
        neg_weight=(neg_margin-neg).detach()
        neg_weight=torch.max(torch.zeros_like(neg_weight),neg_weight)
        
        lse_negative_row=torch.logsumexp(log_scale*(neg_margin-neg)*neg_weight,dim=-1)
        # lse_negative_col=torch.logsumexp(log_scale*(neg_margin-neg)*neg_weight,dim=-2)

        loss_col=F.softplus(lse_positive_row+lse_negative_row)/log_scale
        # loss_row=F.softplus(lse_positive_col+lse_negative_col)/log_scale
        # loss=loss_col+loss_row
        
        return torch.mean(loss_col),dists
    