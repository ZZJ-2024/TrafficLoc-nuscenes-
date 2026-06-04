import torch
import pytorch_lightning as pl
import argparse
import pprint
from loguru import logger as loguru_logger

from src.config.default import get_cfg_defaults
from src.utils.profiler import build_profiler

from src.lightning.data_new_pipeline_kitti import KITTIModule
from src.lightning.lightning_TrafficLoc import PL_TrafficLoc


def log_test_architecture(config):
    fm = config.FEATURE_MATCH
    loguru_logger.info(
        "Test model architecture: "
        f"dust_backbone={fm.DUST_BACKBONE}, dust_with_resnet={fm.DUST_WITH_RESNET}, "
        f"point_backbone={fm.POINT_BACKBONE}, image_enc_dim={fm.IMAGE_ENC_DIM}, "
        f"coarse_patch_size={fm.COARSE_PATCH_SIZE}, fine_match_norm={fm.FINE_MATCH_NORM}"
    )


def load_checkpoint(model, ckpt_path, strict=False):
    try:
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
    except (KeyError, TypeError):
        state_dict = torch.load(ckpt_path, map_location='cpu')

    incompatible = model.load_state_dict(state_dict, strict=strict)
    n_missing = len(incompatible.missing_keys)
    n_unexpected = len(incompatible.unexpected_keys)
    loguru_logger.info(
        f"Loaded checkpoint '{ckpt_path}': missing_keys={n_missing}, unexpected_keys={n_unexpected}"
    )
    if n_missing:
        loguru_logger.warning(f"Sample missing keys: {incompatible.missing_keys[:8]}")
    if n_unexpected:
        loguru_logger.warning(f"Sample unexpected keys: {incompatible.unexpected_keys[:8]}")
    if n_missing > 50:
        loguru_logger.error(
            "Too many missing keys — test config likely does not match the checkpoint. "
            "Use the same data_cfg_path / main_cfg_path as training "
            "(configs/data/config_nuscenes.py + configs/neumap/nuscenes.py)."
        )
    return incompatible


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        '--main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--ckpt_path', type=str, default=None, help='path to the checkpoint')
    parser.add_argument(
        '--dump_dir', type=str, default=None, help="if set, the matching results will be dump to dump_dir")
    parser.add_argument(
        '--profiler_name', type=str, default=None, help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=8)
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    pprint.pprint(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    
    loguru_logger.info("Args and config initialized!")
    log_test_architecture(config)

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_TrafficLoc(config, profiler=profiler, dump_dir=args.dump_dir)
    load_checkpoint(model, args.ckpt_path, strict=False)
    model.dump_dir = args.dump_dir

    data_module = KITTIModule(args, config)

    # lightning trainer (PL 1.x: --gpus / --accelerator 由 shell 脚本传入)
    trainer = pl.Trainer.from_argparse_args(args,
                                            replace_sampler_ddp=False, 
                                            logger=False,
                                            limit_test_batches=1.0)

    loguru_logger.info(f"Start testing!")
    trainer.test(model, datamodule=data_module, verbose=False)
