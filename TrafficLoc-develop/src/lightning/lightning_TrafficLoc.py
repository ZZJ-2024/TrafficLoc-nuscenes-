
from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path
import os
import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from src.TrafficLoc import TrafficLocFeatureMatch
from src.losses.loss import Loss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_pose_errors,
    aggregate_metrics
)
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_TrafficLoc(pl.LightningModule):
    def __init__(self, config, backbone_pretrained_ckpt=None, neumap_pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        

        self.config = lower_config(config)  # full config
        self.save_hyperparameters(self.config)

        self.profiler = profiler or PassThroughProfiler()
        self.dump_dir = dump_dir

        self.model = TrafficLocFeatureMatch(config=self.config)
      
        self.loss = Loss(self.config)

        if backbone_pretrained_ckpt is not None: # load backbone weights if there exists
            if os.path.exists(backbone_pretrained_ckpt):
                self.load_backbone_weight(backbone_pretrained_ckpt)
            else:
                raise Exception("backbone weights not exists: {}".format(backbone_pretrained_ckpt))
        
        if neumap_pretrained_ckpt is not None: # load neumap weights, used in code prune and code finetune
            if os.path.exists(neumap_pretrained_ckpt):
                self.load_neumap_weights(neumap_pretrained_ckpt)
            else:
                raise Exception("neumap weights not exists: {}".format(neumap_pretrained_ckpt))

        if self.config['exp']['pred_focal']:
            from dust3r.inference import load_model

            model_path = "model_release/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            device = 'cuda'
            self.dust3r = load_model(model_path, device)

    def load_backbone_weight(self, backbone_pretrained_ckpt):

        state_dict = torch.load(backbone_pretrained_ckpt, map_location='cpu')['state_dict']
        pretrained_dict=self.model.state_dict()
        model_dict={}
        for k,v in state_dict.items():
            if k in pretrained_dict and v.shape==pretrained_dict[k].shape:
                model_dict[k]=v
                print('hit', k)
            elif k[:8]=='backbone':
                print('miss', k)
        self.model.load_state_dict(model_dict, strict=False)
        logger.info(f"Load \'{backbone_pretrained_ckpt}\' as pretrained checkpoint")
    
    def load_neumap_weights(self, neumap_pretrained_ckpt):
        try:
            state_dict=torch.load(neumap_pretrained_ckpt, map_location='cpu')['state_dict']
        except:
            state_dict=torch.load(neumap_pretrained_ckpt, map_location='cpu')
        state_dict={k: v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)
        
    def configure_optimizers(self):
        optimizer1 = build_optimizer(self, self.config)
        scheduler1 = build_scheduler(self.config, optimizer1)

        return [optimizer1], [scheduler1]
    
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
       
        warmup_step = self.config['trainer']['warmup_step']
        if self.trainer.global_step <= warmup_step:
            if self.config['trainer']['warmup_type'] == 'linear':
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['initial_lr'] * (self.trainer.global_step / warmup_step)
            else:
                raise ValueError('Unknown lr warm-up strategy: {}'.format(self.config['trainer']['warmup_type']))
        
        # update params
        optimizer.step(closure=optimizer_closure)

        optimizer.zero_grad()

        
    def _trainval_inference(self, batch):    
        with self.profiler.profile("NeuMap"):
            self.model(batch)
    
        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def _compute_metrics(self, batch):
        unseen = 0
        for voxel_name in batch['voxel_id']:
            if 'no_img' in voxel_name[0]:
                unseen = 1
        with self.profiler.profile("Copmute metrics"):
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair
            
            image_name = list(zip(*batch['image_name']))
            bs = batch['image'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(image_name[b]) for b in range(bs)],
                'inliers': batch['inliers'],
                'R_errs_abs': batch['R_errs_abs'],
                't_errs_abs': batch['t_errs_abs'],
                'qvec': batch['qvec'],
                'tvec': batch['tvec'],
                'name': batch['image_name'],
                'unseen': [unseen]}
            ret_dict = {'metrics': metrics}
        return ret_dict
    
    def training_step(self, batch, batch_idx):
        if self.global_step == 10 or self.global_step % 5000 == 0:
            os.system('nvidia-smi')
        batch['epoch'] = self.current_epoch
        self._trainval_inference(batch)
        
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.log(f'train/{k}', v, self.global_step)
        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.log(
                'train/avg_loss_on_epoch', avg_loss)
            self.logger.experiment.add_scalar('train/avg_loss_on_epoch_new', avg_loss, self.current_epoch + 1)
    
    def on_validation_epoch_start(self) -> None:
        self.r_error_list = []
        self.t_error_list = []
        
    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        batch['epoch'] = self.current_epoch
        self._trainval_inference(batch)
        ret_dict = self._compute_metrics(batch)
        
        self.r_error_list.append(ret_dict['metrics']['R_errs_abs'][0])
        self.t_error_list.append(ret_dict['metrics']['t_errs_abs'][0])
        
        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
        }
        
    def validation_epoch_end(self, outputs):
        if self.config['feature_match']['kitti']:
            s_rate, r_rate, t_rate, r_mean, r_med, t_mean, t_med = self.cal_error_kitti()
            self.logger.experiment.add_scalar(f'kitti/success_rate', s_rate, self.current_epoch + 1)
            self.logger.experiment.add_scalar(f'kitti/r<10_rate', r_rate, self.current_epoch + 1)
            self.logger.experiment.add_scalar(f'kitti/t<5_rate', t_rate, self.current_epoch + 1)
            self.logger.experiment.add_scalar(f'kitti/r_mean', r_mean, self.current_epoch + 1)
            self.logger.experiment.add_scalar(f'kitti/r_med', r_med, self.current_epoch + 1)
            self.logger.experiment.add_scalar(f'kitti/t_mean', t_mean, self.current_epoch + 1)
            self.logger.experiment.add_scalar(f'kitti/t_med', t_med, self.current_epoch + 1)
            
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0 
            val_metrics_4tb = aggregate_metrics(metrics)
        
            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.log(f'val_{valset_idx}/avg_{k}', mean_v)
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}_epoch', mean_v, self.current_epoch + 1)

                for k, v in val_metrics_4tb.items():
                    self.log(f"metrics_{valset_idx}/{k}", v)
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}_epoch", v, self.current_epoch + 1)
            plt.close('all')
            
            if valset_idx == 0:
                self.log(f'r_median', torch.tensor(val_metrics_4tb['r_median']))
                self.log(f't_median', torch.tensor(val_metrics_4tb['t_median']))
                self.log(f'acc1', torch.tensor(val_metrics_4tb['acc1']))
                self.log(f'acc2', torch.tensor(val_metrics_4tb['acc2']))
                self.log(f'acc3', torch.tensor(val_metrics_4tb['acc3']))
                self.logger.experiment.add_scalar(f'eval/r_median', torch.tensor(val_metrics_4tb['r_median']), self.current_epoch + 1)
                self.logger.experiment.add_scalar(f'eval/t_median', torch.tensor(val_metrics_4tb['t_median']), self.current_epoch + 1)
                self.logger.experiment.add_scalar(f'eval/acc1', torch.tensor(val_metrics_4tb['acc1']), self.current_epoch + 1)
                self.logger.experiment.add_scalar(f'eval/acc2', torch.tensor(val_metrics_4tb['acc2']), self.current_epoch + 1)
                self.logger.experiment.add_scalar(f'eval/acc3', torch.tensor(val_metrics_4tb['acc3']), self.current_epoch + 1)
            else:
                self.logger.experiment.add_scalar(f'eval/r_median_{valset_idx}', torch.tensor(val_metrics_4tb['r_median']), self.current_epoch + 1)
                self.logger.experiment.add_scalar(f'eval/t_median_{valset_idx}', torch.tensor(val_metrics_4tb['t_median']), self.current_epoch + 1)
                self.logger.experiment.add_scalar(f'eval/acc1_{valset_idx}', torch.tensor(val_metrics_4tb['acc1']), self.current_epoch + 1)
                self.logger.experiment.add_scalar(f'eval/acc2_{valset_idx}', torch.tensor(val_metrics_4tb['acc2']), self.current_epoch + 1)
                self.logger.experiment.add_scalar(f'eval/acc3_{valset_idx}', torch.tensor(val_metrics_4tb['acc3']), self.current_epoch + 1)

        # only save pth file, not ckpt file
        save_path = os.path.join(self.logger.log_dir, "checkpoints", f"epoch_{self.current_epoch}.pth")
        if (self.current_epoch + 1) % 5 == 0:
            try:
                torch.save(self.state_dict(), save_path)
                print(f"Model weights saved at {save_path}")
            except:
                None
        else:
            None
            
        
    def on_test_epoch_start(self) -> None:
        self.pred_error = 0
        self.final_error = 0
        self.final_error_num = 0
        
        
        self.r_error_list = []
        self.t_error_list = []

    def test_step(self, batch, batch_idx): 
        batch['epoch'] = 0
        print("processing " + batch['image_name'][0])
        with self.profiler.profile("LoFTR"):
            self.model(batch)
          
        with self.profiler.profile("Compute losses"):
            self.loss(batch)
                

        gt_focal = batch['camera_params'][0][0].clone()
        if self.config['exp']['pred_focal']:
            from dust3r.inference import inference, load_model
            from dust3r.utils.image import load_images
            from dust3r.image_pairs import make_pairs
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
            import PIL## to read image size
            device = 'cuda'
            image_path = batch['image_path'][0]
            images = load_images([image_path, image_path], size=512)
            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True) # pair with itself
            output = inference(pairs, model=self.dust3r, device=device, batch_size=1)

            # at this stage, you have the raw dust3r predictions
            view1, pred1 = output['view1'], output['pred1'] # pair (im1, im2)
            view2, pred2 = output['view2'], output['pred2'] # pair (im2, im1)

            scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)

            width, height = PIL.Image.open(image_path).size
            long_side = max(width, height)
            scale_factor = long_side/512

            focals = scene.get_focals()

            print(f"after scaling with scale factor {long_side}/512={scale_factor}")
            print(focals * scale_factor)

            pred_focal = focals * scale_factor
            pred_focal = pred_focal[0]

            # substitute GT camera_params
            print(f"GT type is {batch['camera_type']}")
            print(f"GT intriansic is {batch['camera_params']}")
            gt_focal = batch['camera_params'][0][0].clone()


            # SIMPLE_PINHOLE
            batch['camera_type'] = ['SIMPLE_PINHOLE']
            batch['camera_params'][0][0] = pred_focal
            batch['camera_params'] = batch['camera_params'][:, :3]
            print(f"pred type is {batch['camera_type']}")
            print(f"pred intriansic is {batch['camera_params']}")

        ret_dict = self._compute_metrics(batch)
        print(f"r_error:{ret_dict['metrics']['R_errs_abs']}")
        print(f"t_error:{ret_dict['metrics']['t_errs_abs']}")
        
        self.r_error_list.append(ret_dict['metrics']['R_errs_abs'][0])
        self.t_error_list.append(ret_dict['metrics']['t_errs_abs'][0])
        
        if self.config['exp']['pred_focal']:
            print(gt_focal)
            print(pred_focal)
            self.pred_error += abs((pred_focal - gt_focal) / gt_focal)
            print(self.pred_error)
        
        if 'final_camera_params' in batch:
            print(f"final refined intrinsic is {batch['final_camera_params']}")
            final_focal = batch['final_camera_params'][0]
            print(final_focal)

            if abs((final_focal - gt_focal) / gt_focal) < 0.2:
                self.final_error += abs((final_focal - gt_focal) / gt_focal)
                self.final_error_num += 1
                print(self.final_error)
                print(self.final_error_num)
            
            


        
        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                image_name = list(zip(*batch['image_name']))
                bs = batch['image'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    item['image_name'] = image_name[b_id]
                    dumps.append(item)
                ret_dict['dumps'] = dumps
        
        return ret_dict

    def cal_error_kitti(self):
        r_error_arr = np.array(self.r_error_list)
        t_error_arr = np.array(self.t_error_list)
        
        r_mask = r_error_arr < 10
        t_mask = t_error_arr < 5
        
        valid_mask = r_mask * t_mask
        
        success_rate = valid_mask.sum() / valid_mask.shape[0]
        r_rate = r_mask.sum() / r_mask.shape[0]
        t_rate = t_mask.sum() / t_mask.shape[0]
        
        r_error_arr = r_error_arr[valid_mask]
        t_error_arr = t_error_arr[valid_mask]
        
        print(f"success rate is: {success_rate}")
        print(f"r<10 success rate is: {r_rate}")
        print(f"t<5 success rate is: {t_rate}")
        print(f"r_error: {np.mean(r_error_arr)}, {np.median(r_error_arr)}")
        print(f"t_error: {np.mean(t_error_arr)}, {np.median(t_error_arr)}")
        
        return success_rate, r_rate, t_rate, np.mean(r_error_arr), np.median(r_error_arr), np.mean(t_error_arr), np.median(t_error_arr)
        
        
    def test_epoch_end(self, outputs):
        # for kitti baseline compare
        if self.config['feature_match']['kitti']:
            a,b,c,d,e,f,g = self.cal_error_kitti()
         
        if self.config['exp']['pred_focal']:
            self.pred_error = self.pred_error / len(outputs)
            print(f"pred focal error avearege in %: {self.pred_error * 100}")
            
        self.final_error = self.final_error / self.final_error_num
        print(f"final focal error avearege in %: {self.final_error * 100}")
            
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
           
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'NSM_pred_eval', dumps)
            names_set=set()
            
            with open(os.path.join(self.dump_dir, 'pose.txt'), 'w') as f:
                for i, name in enumerate(metrics['name']):
                    if name in names_set:
                        continue
                    names_set.add(name)
                    qvec=metrics['qvec'][i*4:i*4+4]
                    tvec=metrics['tvec'][i*3:i*3+3]
                    qvec = ' '.join(map(str, qvec))
                    tvec = ' '.join(map(str, tvec))
                    f.write(f'{name} {qvec} {tvec}\n')
                   