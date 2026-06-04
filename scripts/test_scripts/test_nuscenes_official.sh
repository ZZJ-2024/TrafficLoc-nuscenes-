#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

# 官方 release 权重对应 KPConv + patch 8×8 + 128 维（与 TrafficLoc-develop-2 一致）
# 若本地 config_nuscenes.py 已改成论文 DUSt3R+PT 版，请临时恢复官方 KPConv 配置，或使用 develop-2 里的 config。
data_cfg_path="configs/data/config_nuscenes_official_kpconv.py"
main_cfg_path="configs/neumap/carla_int.py"
ckpt_path="logs/tb_logs/demo_nuscenes_100/version_6/checkpoints/epoch_29.pth"

n_nodes=1
n_gpus_per_node=1
torch_num_workers=4
batch_size=1
dump_dir="dump/nuscenes_test_official"
profiler_name="inference"

python3.9 test_new_pipeline_kitti.py \
    --data_cfg_path=${data_cfg_path} \
    --main_cfg_path=${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${dump_dir} \
    --profiler_name=${profiler_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}  \
    --benchmark=True
