import imp
from multiprocessing.spawn import import_main_path
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config['loss']  # config under the global namespace
        self.all_config = config
     
    def forward(self, data):
        
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        
        loss = torch.tensor(0.).cuda()
   
        if 'coarse_matching_loss' in data:
            loss = loss + data['coarse_matching_loss']
            loss_scalars.update({"coarse_matching_loss": data['coarse_matching_loss'].clone().detach().cpu()})
        
        if 'coarse_dense_loss' in data:
            loss = loss + data['coarse_dense_loss']
            loss_scalars.update({"coarse_dense_loss": data['coarse_dense_loss'].clone().detach().cpu()})
            
        if 'fine_matching_loss' in data:
            loss = loss + data['fine_matching_loss']
            loss_scalars.update({"fine_matching_loss": data['fine_matching_loss'].clone().detach().cpu()})
        
        if 'fine_softmax_loss' in data:
            loss = loss + data['fine_softmax_loss']
            loss_scalars.update({"fine_softmax_loss": data['fine_softmax_loss'].clone().detach().cpu()})
            
        if 'coarse_bce_loss' in data:
            loss = loss + data['coarse_bce_loss']
            loss_scalars.update({"coarse_bce_loss": data['coarse_bce_loss'].clone().detach().cpu()})
            
        if 'coarse_euc_loss' in data:
            loss = loss + data['coarse_euc_loss']
            loss_scalars.update({"coarse_euc_loss": data['coarse_euc_loss'].clone().detach().cpu()})
        
        if 'att_loss' in data:
            loss = loss + data['att_loss']
            loss_scalars.update({"att_loss": data['att_loss'].clone().detach().cpu()})
        
        
        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})

