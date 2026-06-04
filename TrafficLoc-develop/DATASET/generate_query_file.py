'''
This file is used to generate train_list_{voxel_size} files from mapping folders
Q: How to get mapping folders for diy scenes.
A: To run custom dataset, please first use r2d2 to generate key-points and triangulate 3D points with COLMAP.
Then, convert the COLMAP output files to the same format as data/kapture/aachen1.0/mapping/points.
'''

import numpy as np
import os
import tqdm
import random
import argparse
from kapture.io.csv import kapture_from_dir
import kapture
import open3d as o3d
import glob
import math
from src.utils.depth_convert import dpt_3d_convert
import quaternion
from PIL import Image

# y should be flipped
scene_center_dict = {
    't1_int1': np.array([92.5, -57.5, 0]),
    't1_int2': np.array([157.5, -55, 0]),
    't1_int3': np.array([205, -57.5, 0]),
    't1_int4': np.array([332.5, -57.5, 0]),
    't1_int5': np.array([92.5, -125, 0]),
    't1_int6': np.array([220, -130, 0]),
    't1_int7': np.array([90, -200, 0]),
    't1_int8': np.array([220, -195, 0]),
    't1_int9': np.array([92.5, -325, 0]),
    't1_int10': np.array([335, -287.5, 0]),
    
    't2_int1': np.array([135, -235, 0]),
    't2_int2': np.array([45, -190, 0]),
    't2_int3': np.array([135, -190, 0]),
    't2_int4': np.array([50, -240, 0]),
    't2_int5': np.array([2.5, -187.5, 0]),
    't2_int6': np.array([45, -302.5, 0]),
    't2_int7': np.array([187.5, -235, 0]),
    't2_int8': np.array([187.5, -192.5, 0]),
    't2_int9': np.array([195, -297.5, 0]),
    
    't3_int1': np.array([-80, 0, 0]),
    't3_int2': np.array([-80, 140, 0]),
    't3_int3': np.array([-80, -135, 0]),
    't3_int4': np.array([0, -130, 0]),
    't3_int5': np.array([2.5, 137.5, 0]),
    't3_int6': np.array([-2.5, -197.5, 0]),
    't3_int7': np.array([170, -62.5, 0]),
    't3_int8': np.array([5, 195, 0]),
    
    't4_int1': np.array([257.5, 250, 0]),
    't4_int2': np.array([257.5, 172.5, 0]),
    't4_int3': np.array([310, 172.5, 0]),
    't4_int4': np.array([312.5, 250, 0]),
    't4_int5': np.array([202.5, 247.5, 0]),
    't4_int6': np.array([202.5, 172.5, 0]),
    't4_int7': np.array([202.5, 307.5, 0]),
    't4_int8': np.array([257.5, 307.5, 0]),
    't4_int9': np.array([350, 170, 0]),
    't4_int10': np.array([312.5, 120, 0]),
    't4_int11': np.array([257.5, 122.5, 0]),
    't4_int12': np.array([130, 175, 0]),
    
    't5_int1': np.array([-125, 0, 0]),
    't5_int2': np.array([-50, 0, 0]),
    't5_int3': np.array([30, 0, 0]),
    't5_int4': np.array([102.5, 0, 0]),
    't5_int5': np.array([-190, 0, 0]),
    't5_int6': np.array([-190, 90, 0]),
    't5_int7': np.array([-125, 90, 0]),
    't5_int8': np.array([-50, 90, 0]),
    't5_int9': np.array([30, 90, 0]),
    't5_int10': np.array([-125, 135, 0]),
    't5_int11': np.array([-190, -90, 0]),
    't5_int12': np.array([-125, -90, 0]),
    't5_int13': np.array([-50, -90, 0]),
    't5_int14': np.array([30, -90, 0]),
    't5_int15': np.array([-125, -145, 0]),
    
    't6_int1': np.array([140, 17.5, 0]),
    't6_int2': np.array([225, 17.5, 0]),
    't6_int3': np.array([330, 17.5, 0]),
    't6_int4': np.array([0, -45, 0]),
    't6_int5': np.array([140, -42.5, 0]),
    't6_int6': np.array([0, -140, 0]),
    't6_int7': np.array([90, -140, 0]),
    't6_int8': np.array([5, -240, 0]),
    't6_int9': np.array([95, -245, 0]),
    't6_int10': np.array([95, -42.5, 0]),
    't6_int11': np.array([230, -47.5, 0]),
    't6_int12': np.array([10, 17.5, 0]),
    
    't7_int1': np.array([-30, 0, 0]),
    't7_int2': np.array([-60, 0, 0]),
    't7_int3': np.array([-25, 60, 0]),
    't7_int4': np.array([-60, 70, 0]),
    't7_int5': np.array([-45, 85, 0]),
    't7_int6': np.array([-30, 105, 0]),
    't7_int7': np.array([-80, 135, 0]),
    't7_int8': np.array([-117.5, 152.5, 0]),
    
    't10_int1': np.array([-50, -20, 0]),
}

scene_center_dict = {
    't3_int4': np.array([0, -130, 0]),
    't10_int1': np.array([-50, -20, 0]),
}

scene_center_dict = {
    't3_int4': np.array([0, -130, 0]),
}

scene_center_dict = {
    't1_int1': np.array([92.5, -57.5, 0]),
    't2_int1': np.array([135, -235, 0]),
    't3_int1': np.array([-80, 0, 0]),
    't4_int1': np.array([257.5, 250, 0]),
    't5_int1': np.array([-125, 0, 0]),
    't6_int1': np.array([140, 17.5, 0]),
    't7_int1': np.array([-30, 0, 0]),
    't10_int1': np.array([-50, -20, 0]),
}



voxel_size = 50
voxel_stride = 25
image_overlap_threshold = 0.3
str_image_overlap = str(image_overlap_threshold).replace(".", "")
voxel_overlap_threshold = 0.25
str_voxel_overlap = str(voxel_overlap_threshold).replace(".", "")

root_path = f"./demo_dataset"

only_max = False
is_seq5 = False

for scene_name, scene_center in scene_center_dict.items():
    input_path = f"{scene_name}/query"

    output_path = f"query_{scene_name}_v{voxel_size}_s{voxel_stride}_io{str_image_overlap}_vo{str_voxel_overlap}"
    if is_seq5:
        output_path = "seq5_" + output_path
    
    if only_max:
        output_path += "_max"
    path=os.path.join(root_path, input_path)
    if not os.path.exists(path):
        continue
    print(f"processing {scene_name} with voxel size={voxel_size}")

    # read ply
    point_cloud_file = os.path.join(root_path, input_path, f"pcd_{scene_name}_train_down.ply")
    point_cloud = o3d.io.read_point_cloud(point_cloud_file)
    scene_points = np.array(point_cloud.points)
    print(scene_points.shape)
    
        
    # read voxel data
    voxel_path = os.path.join(root_path, scene_name, f"train_list_v{voxel_size}_s{voxel_stride}_io{str_image_overlap}_vo{str_voxel_overlap}")
    voxel_file_list = glob.glob(voxel_path + '/*.npy')
    voxel_file_list = [file_name for file_name in voxel_file_list if "query" not in file_name]
    print(voxel_file_list)

    voxel_dict = {}
    for file_name in voxel_file_list:
        key_name = os.path.join(scene_name, f"train_list_v{voxel_size}_s{voxel_stride}_io{str_image_overlap}_vo{str_voxel_overlap}", os.path.basename(file_name))
        voxel_data = np.load(file=file_name, allow_pickle=True).item()
        voxel_dict[key_name]={
            'xyz_max': voxel_data['xyz_max'],
            'xyz_min': voxel_data['xyz_min'],
            'voxel_points': scene_points[np.all((scene_points >= voxel_data['xyz_min']) & (scene_points <= voxel_data['xyz_max']), axis=1)]
        }

    query_dict = {}
    
    # 用GT Pose和Depth，用Projection得到image对应的3D point
    projector = dpt_3d_convert()
    W = 1920
    H = 1080
    fov = 90  # horizontal fov
    f_original = W / (2 * math.tan(fov/2 * math.pi/180))
    resize_w = 512
    resize_h = 288

    f_new = f_original * resize_w/W
    intrinsic = np.array([[f_new,0,resize_w/2],
                    [0,f_new,resize_h/2],
                    [0,0,1]])
    
    kdata=kapture_from_dir(path)
    for timestep, data_dict in kdata.records_camera.items():
        sensor_id, img_name = next(iter(data_dict.items()))
        
        if is_seq5 and 'seq5' not in img_name:
            continue
        
        if not is_seq5 and 'seq5' in img_name:
            continue
        
        pose_world_to_cam = kdata.trajectories[(timestep, sensor_id)]
        pose_world_to_cam_matrix = np.zeros((4, 4), dtype=np.float)
        pose_world_to_cam_matrix[0:3, 0:3] = quaternion.as_rotation_matrix(pose_world_to_cam.r)
        pose_world_to_cam_matrix[0:3, 3] = pose_world_to_cam.t_raw
        pose_world_to_cam_matrix[3, 3] = 1.0
        T_c2w = np.linalg.inv(pose_world_to_cam_matrix)
        T_w2c = pose_world_to_cam_matrix
        
        depth_map_path = os.path.join(path, 'sensors/depth_data', img_name.replace("image", "depth"))
        depth_map = Image.open(depth_map_path) # RGBA
        depth_map = np.array(depth_map)
        
        R = depth_map[:,:,0].astype(np.float32)
        G = depth_map[:,:,1].astype(np.float32)
        B = depth_map[:,:,2].astype(np.float32)
        normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1)
        depth_map = 1000 * normalized
        
        depth_map_image = Image.fromarray(depth_map)
        new_size = (resize_w, resize_h)
        resized_depth_map_nearest = depth_map_image.resize(new_size, Image.NEAREST)
        depth_map = np.array(resized_depth_map_nearest)
        
        x = np.arange(resize_w)
        y = np.arange(resize_h)
        xv, yv = np.meshgrid(x, y)

        keypoints = np.vstack([xv.ravel(), yv.ravel()]).T
        keypoints = keypoints.astype(np.int16)
        
        depths = depth_map.reshape(-1)
        depth_mask = depths < 999
        
        # project into 3D points
        dense_point = projector.proj_2to3(keypoints, depths, intrinsic, T_c2w, depth_unit=1)
        img_points = dense_point[depth_mask]
        num_dense_point = img_points.shape[0]
    
        max_mask_num = 0
        max_mask_voxel_name = None
        print(f'scene {scene_name}, voxel size {voxel_size}: processing {img_name}')
        best_num = 0
        for voxel_name, data in voxel_dict.items():
            voxel_max = data['xyz_max']
            voxel_min = data['xyz_min']
            voxel_points = data['voxel_points']

            # if np.sum(mask) > max_mask_num:
            #     max_mask_num = np.sum(mask)
            #     max_mask_voxel_name = voxel_name
                # query_dict[img_name] = [voxel_name]
                
                    
            # image中重叠部分
            mask = np.all((dense_point >= voxel_min) & (dense_point <= voxel_max), axis=1)
            if np.sum(mask) < num_dense_point * image_overlap_threshold:
                continue
                    
            # voxel中重叠部分
            cam_coord = np.dot(T_w2c[0:3, 0:3], voxel_points.T) + T_w2c[0:3, 3:]  # transform pc to camera coordinate system
            proj_coord = np.dot(intrinsic, cam_coord) # 3 x num_point
            proj_xy = proj_coord[0:2, :] / (proj_coord[2, :] + 1e-9)
            proj_xy_int = proj_xy.astype(np.int32)
            proj_depth = proj_coord[2,:]
            in_frustum_mask = (proj_xy[0, :] >= 0) & (proj_xy[0, :] <= resize_w-1) & \
                (proj_xy[1, :] >= 0) & (proj_xy[1, :] <= resize_h-1) & \
                (proj_depth > 0)  # num_point
            if np.sum(in_frustum_mask) < voxel_points.shape[0] * voxel_overlap_threshold:
                continue
            
            # success align
            if only_max:
                # 1. only align the best voxel
                if np.sum(mask) > best_num:
                    best_num = np.sum(mask)
                    query_dict[img_name] = [voxel_name]
            else:
                # 2. align all good voxels
                if img_name not in query_dict:
                    query_dict[img_name] = [voxel_name]
                else:
                    query_dict[img_name].append(voxel_name)
                
        if img_name not in query_dict:
            print(f"no voxel is associated with {img_name}")
            # print(f"no voxel has more than {points_thresh} associated points with {img_name}")
            # print(f"add {max_mask_voxel_name} with {max_mask_num} points as associated voxel for {img_name}")
            # query_dict[img_name] = [max_mask_voxel_name]
    
    # assert len(query_dict) == len(train_names)
    # print(query_dict)
    print(len(query_dict))
    tot=0
    for k, v in query_dict.items():
        print(k, len(v))
        tot+=len(v)
    print(tot)
    np.save(os.path.join(root_path, scene_name, f"train_list_v{voxel_size}_s{voxel_stride}_io{str_image_overlap}_vo{str_voxel_overlap}", output_path+'.npy'), query_dict) # 保存在场景的train_list_x里
    with open(os.path.join(root_path, 'train_list', output_path+'.txt'), 'w') as f: # 保存在外部的train_list里，query_场景名.txt
        f.write(os.path.join(scene_name, f"train_list_v{voxel_size}_s{voxel_stride}_io{str_image_overlap}_vo{str_voxel_overlap}", output_path+'.npy'))



