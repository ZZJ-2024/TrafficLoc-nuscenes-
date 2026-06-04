#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

# 必须与 train_nuscenes.sh 完全一致（论文 Route B：DUSt3R* + ResNet + Point Transformer）
data_cfg_path="configs/data/config_nuscenes.py"
main_cfg_path="configs/neumap/nuscenes.py"

# 改成你自训 checkpoint 的实际路径（不要用官方 trafficloc_nuscenes.pth，那是 KPConv 老结构）
ckpt_path="model_release/trafficloc_nuscenes.pth.DISABLED"

n_nodes=1
n_gpus_per_node=1
torch_num_workers=4
batch_size=4
dump_dir="dump/nuscenes_test"
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
# 说明：
#   - 自训 ckpt 必须用这套 config；若用官方 test 配置（KPConv + patch 8），strict=False 会静默跳过
#     大部分权重，test 结果不可信。
#   - 启动后看日志里的 missing_keys：接近 0 表示加载正常；若 >50 说明 config 与 ckpt 不匹配。
#   - 测官方 release 权重请用 scripts/test_scripts/test_nuscenes_official.sh。
