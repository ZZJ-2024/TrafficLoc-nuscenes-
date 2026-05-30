import torch
import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR, StepLR


def build_optimizer(model, config):
    name = config['trainer']['optimizer']
    lr = config['trainer']['true_lr']
    backbone_lr=config['trainer']['backbone_lr']

    if name == "adamw":
        others=[]
        backbones=[]
        pimae=[]
        for k, v in model.named_parameters():
            if (('pimae' in k) or ('image_backbone.' in k)) and config['feature_match']['frozen_dust']: # pimae and dust3R
                pimae.append(v)
                print(f"(params group pimae) {k}: {config['exp']['pimae_lr']}")
            elif (('pimae' in k) or ('image_backbone' in k)) and not config['feature_match']['frozen_dust']: # pimae and dust3R
                pimae.append(v)
                print(f"(params group pimae) {k}: {config['exp']['pimae_lr']}")
            elif 'backbone' in k: # backbone parameters
                backbones.append(v)
                print(f"(params group backbone) {k}: {backbone_lr}")
            else: # other parameters
                others.append(v)
                print(f"(params group others) {k}: {lr}")

        params=[] 
        
        if len(pimae) > 0 and not config['feature_match']['frozen_dust']:
            params.append({
                        'params': pimae,
                        'lr': config['exp']['pimae_lr'],
                        'weight_decay':config['trainer']['adamw_decay']
                    }
                )
        
        params.append({
                'params': backbones,
                'lr': backbone_lr,
                'weight_decay':config['trainer']['adamw_decay']
            }
        )
        
        params.append({
                    'params': others,
                    'lr': lr,
                    'weight_decay':config['trainer']['adamw_decay']
                }
            ) 
        
        return torch.optim.AdamW(params)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {'interval': config['trainer']['scheduler_interval']}
    name = config['trainer']['scheduler']

    if name == 'MultiStepLR':
        scheduler.update(
            {'scheduler': MultiStepLR(optimizer, config['trainer']['mslr_milestones'], gamma=config['trainer']['mslr_gamma'])})
    else:
        raise NotImplementedError()

    return scheduler
