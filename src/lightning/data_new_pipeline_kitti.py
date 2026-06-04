import os
from collections import abc
from loguru import logger
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    RandomSampler,
    dataloader
)


from src.utils.dataloader import get_local_split
from src.datasets.kapture_feature_match import KaptureDatasetFeatureMatch
from src.lightning.kitti import kitti_pc_img_dataset
from src.datasets.sampler import RandomConcatSampler
from kapture.io.csv import kapture_from_dir

from src.utils.misc import lower_config

class KITTIModule(pl.LightningDataModule):
    """ 
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """
    def __init__(self, args, config):
        super().__init__()

        from src.lightning.options_kitti import Options_KITTI, Options_Nuscenes
        if config['FEATURE_MATCH']['NUSCENES']:
            self.opt = Options_Nuscenes()
        else:
            self.opt = Options_KITTI()
        
        
        self.config = lower_config(config)  # full config
        # 1. data config
        # Train and Val should from the same data source
        self.trainval_data_source = config.DATASET.TRAINVAL_DATA_SOURCE
        self.test_data_source = config.DATASET.TEST_DATA_SOURCE
        # training and validating
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.train_list_path = config.DATASET.TRAIN_LIST_PATH
        self.train_subdir=config.DATASET.TRAIN_SUBDIR
        self.val_data_root = config.DATASET.VAL_DATA_ROOT
        self.val_list_path = config.DATASET.VAL_LIST_PATH
        self.val_subdir=config.DATASET.VAL_SUBDIR
        # testing
        self.test_data_root = config.DATASET.TEST_DATA_ROOT
        self.test_list_path = config.DATASET.TEST_LIST_PATH
        self.test_subdir=config.DATASET.TEST_SUBDIR
        self.resolution=config.DATASET.RESOLUTION
        self.max_n_points=config.DATASET.MAX_N_POINTS
        self.random_crop=config.DATASET.RANDOM_CROP
        self.aspect_ratio=config.DATASET.ASPECT_RATIO
        self.rgb=config.MODEL.RGB

        # 2. dataset config
        # general options
       
        # 3.loader parameters
        self.batch_size=args.batch_size
        
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }

        self.val_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': True
        }
        
        # 4. sampler
        self.data_sampler = config.TRAINER.DATA_SAMPLER
        self.n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET

        # misc configurations
        self.parallel_load_data = getattr(args, 'parallel_load_data', False)
        self.seed = config.TRAINER.SEED  # 66
      

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'test'], "stage must be either fit or test"

        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except:
            self.world_size = 1
            self.rank = 0
            logger.warning(" (set wolrd_size=1 and rank=0)")

        if stage == 'fit':
            self.train_dataset = kitti_pc_img_dataset(self.opt, 
                                                    'train', 
                                                    self.config)
            self.val_dataset = kitti_pc_img_dataset(self.opt, 
                                                    'val', 
                                                    self.config)
            logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')
        else:  # stage == 'test
            self.test_dataset = kitti_pc_img_dataset(self.opt, 
                                                    'val', 
                                                    self.config)
            logger.info(f'[rank:{self.rank}]: Test Dataset loaded!')
    
    def train_dataloader(self):
        """ Build training dataloader for ScanNet / MegaDepth. """
        #assert self.data_sampler in ['scene_balance']
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
        
        if self.world_size > 1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            dataloader = DataLoader(self.train_dataset, sampler=sampler, **self.train_loader_params)
        else:
            dataloader = DataLoader(self.train_dataset, shuffle=True, **self.train_loader_params)
        
        return dataloader
    
    def val_dataloader(self):
        """ Build validation dataloader for ScanNet / MegaDepth. """
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        if not isinstance(self.val_dataset, abc.Sequence):
            if self.world_size > 1:
                sampler = DistributedSampler(
                    self.val_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False
                )
                return DataLoader(self.val_dataset, sampler=sampler, **self.val_loader_params)
            else:
                return DataLoader(self.val_dataset, **self.val_loader_params)
        else:
            dataloaders = []
            for dataset in self.val_dataset:
                if self.world_size > 1:
                    sampler = DistributedSampler(
                        dataset,
                        num_replicas=self.world_size,
                        rank=self.rank,
                        shuffle=False
                    )
                    dataloaders.append(DataLoader(dataset, sampler=sampler, **self.val_loader_params))
                else:
                    dataloaders.append(DataLoader(dataset, **self.val_loader_params))
            return dataloaders

    def test_dataloader(self, *args, **kwargs):
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Test Sampler and DataLoader re-init.')
        if self.world_size > 1:
            sampler = DistributedSampler(
                self.test_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
            return DataLoader(self.test_dataset, sampler=sampler, **self.test_loader_params)
        else:
            return DataLoader(self.test_dataset, **self.test_loader_params)
