import os
import shutil
import tqdm

# USE ABSOLUTE PATH !
source_folder = '/home/stud/luyun/storage/user/carla/dataset_large_int' # source dataset folder path
train_folder = '/home/stud/luyun/code/TrafficLoc/demo_dataset' # destination folder path 


int_name_list = ['t1_int1', 't1_int2', 't1_int3', 't1_int4', 't1_int5', 't1_int6', 't1_int7', 't1_int8', 't1_int9', 't1_int10',
                 't2_int1', 't2_int2', 't2_int3', 't2_int4', 't2_int5', 't2_int6', 't2_int7', 't2_int8', 't2_int9',
                 't3_int1', 't3_int2', 't3_int3', 't3_int4', 't3_int5', 't3_int6', 't3_int7', 't3_int8',
                 't4_int1', 't4_int2', 't4_int3', 't4_int4', 't4_int5', 't4_int6', 't4_int7', 't4_int8', 't4_int9', 't4_int10', 't4_int11', 't4_int12',
                 't5_int1', 't5_int2', 't5_int3', 't5_int4', 't5_int5', 't5_int6', 't5_int7', 't5_int8', 't5_int9', 't5_int10', 't5_int11', 't5_int12', 't5_int13', 't5_int14', 't5_int15',
                 't6_int1', 't6_int2', 't6_int3', 't6_int4', 't6_int5', 't6_int6', 't6_int7', 't6_int8', 't6_int9', 't6_int10', 't6_int11', 't6_int12',
                 't7_int1', 't7_int2', 't7_int3', 't7_int4', 't7_int5', 't7_int6', 't7_int7', 't7_int8',
                 't10_int1'
                 ]

int_ply_name_list = ['lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road',
                     'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road',
                     'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road',
                     'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road',
                     'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road',
                     'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road',
                     'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road', 'lidar_road',
                     'lidar_road'
                    ]

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
else:
    print(f"{train_folder} has already existed.")
    input("The training data will be soft linked to the directory. Press 'Enter' to continue...")

for int_name in tqdm.tqdm(int_name_list,total=len(int_name_list)):
    # copy template folder
    template_path = os.path.join(source_folder, 'template')
    dst_path = os.path.join(train_folder, int_name)
    shutil.copytree(template_path, dst_path)
    
for int_name, int_ply_name in zip(int_name_list, int_ply_name_list):
    print(f"processing {int_name}")
    # train ply
    os.chdir(os.path.join(train_folder, int_name, 'mapping'))
    link_target = os.path.join(source_folder, int_name, f'{int_name}_{int_ply_name}', f'agg_{int_name}_{int_ply_name}_down02_crop_100_100_50.ply')
    link_name = f'pcd_{int_name}_train_down.ply'
    os.symlink(link_target, link_name)
    
    # val ply
    os.chdir(os.path.join(train_folder, int_name, 'query'))
    link_target = os.path.join(source_folder, int_name, f'{int_name}_{int_ply_name}', f'agg_{int_name}_{int_ply_name}_down02_crop_100_100_50.ply')
    link_name = f'pcd_{int_name}_train_down.ply'
    os.symlink(link_target, link_name)
    
    
    # train depth
    os.chdir(os.path.join(train_folder, int_name, 'mapping/sensors/depth_data'))
    link_target = os.path.join(source_folder, int_name, f'{int_name}_seq3_depth_png')
    link_name = 'seq3'
    os.symlink(link_target, link_name)

    # train rgb
    os.chdir(os.path.join(train_folder, int_name, 'mapping/sensors/records_data'))
    link_target = os.path.join(source_folder, int_name, f'{int_name}_seq3_rgb_png')
    link_name = 'seq3'
    os.symlink(link_target, link_name)

    # val depth seq4
    os.chdir(os.path.join(train_folder, int_name, 'query/sensors/depth_data'))
    link_target = os.path.join(source_folder, int_name, f'{int_name}_seq4_depth_png')
    link_name = 'seq4'
    os.symlink(link_target, link_name)

    # val rgb seq4
    os.chdir(os.path.join(train_folder, int_name, 'query/sensors/records_data'))
    link_target = os.path.join(source_folder, int_name, f'{int_name}_seq4_rgb_png')
    link_name = 'seq4'
    os.symlink(link_target, link_name)
    
    # val depth & rgb for seq5 (Test T1-T7 hard)
    if 'int1' == int_name[-4:]:
        os.chdir(os.path.join(train_folder, int_name, 'query/sensors/depth_data'))
        link_target = os.path.join(source_folder, int_name, f'{int_name}_seq5_depth_png')
        link_name = 'seq5'
        os.symlink(link_target, link_name)

        os.chdir(os.path.join(train_folder, int_name, 'query/sensors/records_data'))
        link_target = os.path.join(source_folder, int_name, f'{int_name}_seq5_rgb_png')
        link_name = 'seq5'
        os.symlink(link_target, link_name)
    else:
        None

    # train traj
    os.chdir(os.path.join(train_folder, int_name, 'mapping/sensors'))
    link_target = os.path.join(source_folder, int_name, f'{int_name}_train_records_camera.txt')
    link_name = 'records_camera.txt'
    os.symlink(link_target, link_name)

    link_target = os.path.join(source_folder, int_name, f'{int_name}_train_trajectories.txt')
    link_name = 'trajectories.txt'
    os.symlink(link_target, link_name)

    link_target = os.path.join(source_folder, int_name, f'{int_name}_sensors.txt')
    link_name = 'sensors.txt'
    os.symlink(link_target, link_name)


    # val traj
    os.chdir(os.path.join(train_folder, int_name, 'query/sensors'))
    link_target = os.path.join(source_folder, int_name, f'{int_name}_eval_records_camera.txt')
    link_name = 'records_camera.txt'
    os.symlink(link_target, link_name)

    link_target = os.path.join(source_folder, int_name, f'{int_name}_eval_trajectories.txt')
    link_name = 'trajectories.txt'
    os.symlink(link_target, link_name)

    link_target = os.path.join(source_folder, int_name, f'{int_name}_sensors.txt')
    link_name = 'sensors.txt'
    os.symlink(link_target, link_name)