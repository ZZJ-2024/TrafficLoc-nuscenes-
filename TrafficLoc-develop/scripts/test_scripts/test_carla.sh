#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/config_trafficloc.py"
main_cfg_path="configs/neumap/carla_int.py"
ckpt_path="model_release/carla_epoch_39.pth"

n_nodes=1
n_gpus_per_node=1
torch_num_workers=4
batch_size=1
dump_dir="dump/carla_test"
profiler_name="inference"

python test.py \
    --data_cfg_path=${data_cfg_path} \
    --main_cfg_path=${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${dump_dir} \
    --profiler_name=${profiler_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}  \
    --benchmark=True 
