#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/config_nuscenes.py"
main_cfg_path="configs/neumap/nuscenes.py"

n_nodes=1
n_gpus_per_node=1
torch_num_workers=4 # 8
batch_size=4 # 等效 batch=2×4=8
pin_memory=true
exp_name="demo_nuscenes_100"

python3.9 -u ./train_new_pipeline_kitti.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --accumulate_grad_batches=2\
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=50 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=0 \
    --benchmark=True \
    --max_epochs=30 \
    --tensorboard
# 说明：
#   - batch_size=1 + accumulate_grad_batches=8 → 梯度等价 batch=8（论文设置），显存友好。
#   - LR=1e-3，MSLR_MILESTONES=[5,10,15,20,25]（按 epoch，×0.5），到 epoch 25 已衰减到 ~3e-5。
#   - 故 max_epochs 设 30（略超最后一个 milestone）即可，再多基本是低 LR 空转；
#     官方 NuScenes release 权重为 epoch_11，通常 10~25 epoch 即收敛，可据验证集 acc3/r_median 早停。