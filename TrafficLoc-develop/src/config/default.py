from yacs.config import CfgNode as CN
_CN = CN()

##############  ↓  LoFTR Pipeline  ↓  ##############
_CN.MODEL = CN()
_CN.MODEL.RGB=False #Use RGB image or Grey image

# 1. LoFTR-backbone (local feature CNN) config
_CN.MODEL.RESNETFPN = CN()
_CN.MODEL.RESNETFPN.INITIAL_DIM = 128 #Dimensions after the first convolutional layer 
_CN.MODEL.RESNETFPN.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3
_CN.MODEL.RESNETFPN.STRIDES = [1, 2, 2]
_CN.MODEL.NHEAD = 4 #Transformer heads
_CN.MODEL.FFDIM_FACTOR = 4 #MLP middle dimensions
_CN.MODEL.TRANS_BLOCK_NUM=1 #Number of transformer blocks
_CN.MODEL.CODE_NUM = 100 #Code number per transformer block
_CN.MODEL.D_MODEL = 100 #
_CN.MODEL.N_SAMPLE_POINTS = 5000 #Sample N points for each image in the training stage
_CN.MODEL.N_SAMPLE_IN_VOXEL_POINTS = 4000 #Sample M in-voxels points for each image

##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = "kapture" 
_CN.DATASET.TRAIN_DATA_ROOT = None # Root folder
_CN.DATASET.TRAIN_LIST_PATH = None # Training reegion list path
_CN.DATASET.TRAIN_SUBDIR='mapping' # 'mapping'
_CN.DATASET.VAL_DATA_ROOT = None # 'validation', 'testing'
_CN.DATASET.VAL_LIST_PATH = None   # Validation reegion list path
_CN.DATASET.VAL_SUBDIR='mapping' # 'validation', 'testing'
# testing
_CN.DATASET.TEST_DATA_SOURCE = "kapture" 
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None   # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None
_CN.DATASET.RESOLUTION=640
_CN.DATASET.TEST_SUBDIR='query'
_CN.DATASET.MAX_N_POINTS=20000 # Sample maximum N points in dataloader
_CN.DATASET.RANDOM_CROP=True 
_CN.DATASET.ASPECT_RATIO=None 
# 2. dataset config



##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adamw]
_CN.TRAINER.TRUE_LR = None  # Initial learning rate
_CN.TRAINER.ADAMW_DECAY = 0.

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 2000
_CN.TRAINER.PRUNE_THRESH=0.1 #Code pruning thresh

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5


# geometric metrics and pose solver
_CN.TRAINER.SCORE_THRESH=0.5 #Keep predictions whose 
_CN.TRAINER.RANSAC_THRESH=48 #RANSAC reporjection error threshold

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 256 # Sample n images for each voxel


# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5
_CN.TRAINER.LRSTEPS=700
_CN.TRAINER.BACKBONE_LR=2e-3 # Backbone initial learning rate
_CN.TRAINER.UNION_COORDS=False
# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed values might affect the final training result, since not all data items
# are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
_CN.TRAINER.SEED = 66

_CN.LOSS = CN()
_CN.LOSS.COORD_LOSS_THRESH=25 #L2 loss if loss<COORD_LOSS_THRESH, else square root loss
_CN.LOSS.SCALE=1.0 #
_CN.LOSS.CE_SCALE=1.0 #BCE loss scale

_CN.EXP = CN()
_CN.EXP.PRED_FOCAL=False # when True, using dust3r to estimate the focal length during evaluation

_CN.EXP.PIMAE_LR = 1e-5

########################
# feature match config #
########################
_CN.FEATURE_MATCH = CN()
_CN.FEATURE_MATCH.NEW_PIPELINE = False
_CN.FEATURE_MATCH.KITTI = False
_CN.FEATURE_MATCH.NUSCENES = False

# dataset 
_CN.FEATURE_MATCH.PCD_SIZE = 20480
_CN.FEATURE_MATCH.CUBE_NORM = False
_CN.FEATURE_MATCH.CENTER_CROP = False
_CN.FEATURE_MATCH.SAMPLE_POINT = 512
_CN.FEATURE_MATCH.PCD_AUG = False
_CN.FEATURE_MATCH.PCD_FILE_SUFFIX = "train_down"
_CN.FEATURE_MATCH.RGB = False

_CN.FEATURE_MATCH.IN_IMAGE_BOARD = 0.0

# image backbone
_CN.FEATURE_MATCH.DUST_BACKBONE = False
_CN.FEATURE_MATCH.IMAGE_ENC_DIM = 256 # map dust encoder output from 1024 dim to X dim
_CN.FEATURE_MATCH.FROZEN_DUST = False
_CN.FEATURE_MATCH.DUST_WITH_RESNET = False

# point backbone
_CN.FEATURE_MATCH.POINT_BACKBONE = 'pt'
## pt
_CN.FEATURE_MATCH.PT_RAND = 8
_CN.FEATURE_MATCH.PT_NUM_NODE = 1280
_CN.FEATURE_MATCH.PT_NUM_PROXY = 512
_CN.FEATURE_MATCH.PT_NUM_SA_LAYER = 3
_CN.FEATURE_MATCH.PT_NUM_HEAD = 8
_CN.FEATURE_MATCH.PT_POINT_FEAT_DIM = 3
_CN.FEATURE_MATCH.PT_EMBED_DIM = 256
_CN.FEATURE_MATCH.PT_MLP_DIM = 1024
_CN.FEATURE_MATCH.PT_ATTENTION_DROPOUT = 0.1
_CN.FEATURE_MATCH.PT_MLP_DROPOUT = 0.1


_CN.FEATURE_MATCH.RADIUS = 0.2
_CN.FEATURE_MATCH.NUM_CLUSTER = 512
_CN.FEATURE_MATCH.NSAMPLE = 64
_CN.FEATURE_MATCH.PIMAE = False
_CN.FEATURE_MATCH.USE_KNN = False

# feature fusion
_CN.FEATURE_MATCH.NO_FUSION = False
_CN.FEATURE_MATCH.COFI_FUSION = False
_CN.FEATURE_MATCH.COFI_FUSION_NORM = False
_CN.FEATURE_MATCH.COFI_NO_PCD_POS = False
_CN.FEATURE_MATCH.COFI_NUM_LAYER = 4

# coarse match
_CN.FEATURE_MATCH.COARSE_PATCH_SIZE = 16
_CN.FEATURE_MATCH.COARSE_MATCH = False
_CN.FEATURE_MATCH.COARSE_MATCH_TYPE = 'bce_dense'
_CN.FEATURE_MATCH.COARSE_MATCH_NORM = 'none'
_CN.FEATURE_MATCH.DENSE_STD = 0.25
_CN.FEATURE_MATCH.UPSAMPLE = False
_CN.FEATURE_MATCH.DENSE_TEMP = False
_CN.FEATURE_MATCH.BCE_SAMPLE = False
_CN.FEATURE_MATCH.BCE_SPARSE_SOFTMAX = False
_CN.FEATURE_MATCH.COARSE_DENSE_WINDOW = 0
_CN.FEATURE_MATCH.COARSE_WINDOW_GT = False

_CN.FEATURE_MATCH.COARSE_INTRA = False

_CN.FEATURE_MATCH.COARSE_EUC = False

_CN.FEATURE_MATCH.ATT_LOSS = False
_CN.FEATURE_MATCH.ATT_LOSS_I2P_RAD_LOW = 10
_CN.FEATURE_MATCH.ATT_LOSS_I2P_RAD_UP = 20
_CN.FEATURE_MATCH.ATT_LOSS_P2I_DIST_LOW = 3
_CN.FEATURE_MATCH.ATT_LOSS_P2I_DIST_UP = 5
_CN.FEATURE_MATCH.ATT_LOSS_LAYER = 'last'

# fine match
_CN.FEATURE_MATCH.FINE_MATCH = False
_CN.FEATURE_MATCH.FINE_MATCH_NORM = 'l2_norm'
_CN.FEATURE_MATCH.FINE_RESNET = False
_CN.FEATURE_MATCH.FINE_RESNET_ONLY = False
_CN.FEATURE_MATCH.FINE_RESNET_UP = False
_CN.FEATURE_MATCH.FINE_RESNET_UP_NOFUSE = False
_CN.FEATURE_MATCH.FINE_LOSS_TYPE = 'circle'
_CN.FEATURE_MATCH.FINE_SOFTMAX = False
_CN.FEATURE_MATCH.ALIGN_CORNER = False
_CN.FEATURE_MATCH.FINE_PATCH_AUG = False
_CN.FEATURE_MATCH.FINE_UP_POINT = False

# cofi pipeline
_CN.FEATURE_MATCH.COFII2P_PIPE = False
_CN.FEATURE_MATCH.COFII2P_NORM = False

# loss
_CN.FEATURE_MATCH.POS_MARGIN = 0.2
_CN.FEATURE_MATCH.NEG_MARGIN = 1.8
_CN.FEATURE_MATCH.LOG_SCALE = 10
_CN.FEATURE_MATCH.TOL_THRES = 1.0

_CN.FEATURE_MATCH.PRED_INT = False

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
