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
import quaternion
from PIL import Image
from src.utils.depth_convert import dpt_3d_convert
import math

# Here y is flipped to keep consistant with 3D point cloud map
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



voxel_size_list = [50]
voxel_stride_list = [25]

thres_num_points = 10000 # one voxel should has > 'thres_num_points' points
n_image_thresh = 100 # one voxel should has > 'n_image_thresh' associated images

image_overlap_threshold = 0.3
voxel_overlap_threshold = 0.25
str_image_overlap = str(image_overlap_threshold).replace(".", "")
str_voxel_overlap = str(voxel_overlap_threshold).replace(".", "")

vis_result = False  # save point submap

# root_folder
root_path =f"./demo_dataset"
if not os.path.exists(os.path.join(root_path, 'train_list')):
    os.makedirs(os.path.join(root_path, 'train_list'))

for voxel_size, voxel_stride in zip(voxel_size_list, voxel_stride_list):
    print(f"start processing with voxel_size {voxel_size} and voxel_stride {voxel_stride}")
    for scene_name, scene_center in scene_center_dict.items():
        input_path = f"{scene_name}/mapping"
        output_path = f"train_{scene_name}_v{voxel_size}_s{voxel_stride}_io{str_image_overlap}_vo{str_voxel_overlap}"
        point_cloud_file = os.path.join(root_path, input_path, f"pcd_{scene_name}_train_down.ply")

        path=os.path.join(root_path, input_path)
        if not os.path.exists(path):
            print(f"Scene {scene_name} doesn't exist")
            continue
        print(f"processing {scene_name} with voxel size={voxel_size}, voxel stride={voxel_stride}")

        # added
        point_cloud = o3d.io.read_point_cloud(point_cloud_file)
        scene_points = np.array(point_cloud.points)

        print(scene_points.shape)
        max_voxel = np.max(scene_points, axis=0)
        min_voxel = np.min(scene_points, axis=0)
        print(max_voxel)
        print(min_voxel)
        
        diy_max = scene_center + np.array([50, 50, voxel_size / 2])
        diy_min = scene_center - np.array([50, 50, voxel_size / 2])

        assert (np.all(diy_max >= max_voxel)) and (np.all(diy_min <= min_voxel))
        if diy_max is not None and diy_min is not None:
            max_voxel = diy_max
            min_voxel = diy_min
            
        print(max_voxel)
        print(min_voxel)

        # prepare a dict for scene
        data_dict = {}
        x_size = (max_voxel[0] - min_voxel[0]) // voxel_stride
        y_size = (max_voxel[1] - min_voxel[1]) // voxel_stride
        z_size = (max_voxel[2] - min_voxel[2]) // voxel_stride
        x_size = x_size.astype(np.int32)
        y_size = y_size.astype(np.int32)
        z_size = z_size.astype(np.int32)

        while x_size >=1 and min_voxel[0] + (x_size-1) * voxel_stride + voxel_size >= max_voxel[0]:
            x_size = x_size-1
        while y_size >=1 and min_voxel[1] + (y_size-1) * voxel_stride + voxel_size >= max_voxel[1]:
            y_size = y_size-1
        while z_size >=1 and min_voxel[2] + (z_size-1) * voxel_stride + voxel_size >= max_voxel[2]:
            z_size = z_size-1

        assert min_voxel[0] + x_size * voxel_stride + voxel_size >= max_voxel[0]
        assert min_voxel[1] + y_size * voxel_stride + voxel_size >= max_voxel[1]
        assert min_voxel[2] + z_size * voxel_stride + voxel_size >= max_voxel[2]

        print(x_size, y_size, z_size)

        # 记录每个voxel的信息，min，max
        voxel_data = {}
        for indice_x in range(x_size+1):
            for indice_y in range(y_size+1):
                for indice_z in range(z_size+1):
                    voxel_indice = np.array([indice_x, indice_y, indice_z]).astype(int)
                    voxel_coord = tuple(voxel_indice)
                    lower_bound = min_voxel + voxel_indice * voxel_stride
                    upper_bound = lower_bound + voxel_size
                    voxel_points = scene_points[np.all((scene_points >= lower_bound) & (scene_points <= upper_bound), axis=1)]
                    if voxel_points.shape[0] < thres_num_points:
                        continue
                    voxel_data[voxel_coord] = {
                        'xyz_max': upper_bound,
                        'xyz_min': lower_bound,
                        'points': voxel_points,
                        'xyz_mean': np.mean(voxel_points, axis=0),
                        'xyz_std': np.std(voxel_points, axis=0),
                        'xyz_median': np.median(voxel_points, axis=0)
                    }


        # 用points获得image对应的3D point
        # kdata=kapture_from_dir(path)
        # train_points_list, train_names, max_list, min_list=load_points(kdata, path) 
        
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
            sensor_id, image_name = next(iter(data_dict.items()))
            
            pose_world_to_cam = kdata.trajectories[(timestep, sensor_id)]
            pose_world_to_cam_matrix = np.zeros((4, 4), dtype=np.float)
            pose_world_to_cam_matrix[0:3, 0:3] = quaternion.as_rotation_matrix(pose_world_to_cam.r)
            pose_world_to_cam_matrix[0:3, 3] = pose_world_to_cam.t_raw
            pose_world_to_cam_matrix[3, 3] = 1.0
            T_c2w = np.linalg.inv(pose_world_to_cam_matrix)
            T_w2c = pose_world_to_cam_matrix
            
            depth_map_path = os.path.join(path, 'sensors/depth_data', image_name.replace("image", "depth"))
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
            dense_point = dense_point[depth_mask]
            num_dense_point = dense_point.shape[0]
            
            for voxel_name, data in voxel_data.items():
                voxel_max = data['xyz_max']
                voxel_min = data['xyz_min']
                voxel_points = data['points']
                
                # image overlap
                mask = np.all((dense_point >= voxel_min) & (dense_point <= voxel_max), axis=1)
                if np.sum(mask) < num_dense_point * image_overlap_threshold:
                    continue
                        
                # voxel overlap
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
                print(f"scene {scene_name}, voxel_size {voxel_size}: {image_name} aligned to {voxel_name}")
                if "image_names" not in data:
                    voxel_data[voxel_name]['image_names'] = [image_name]
                else:
                    voxel_data[voxel_name]['image_names'].append(image_name)
                    
        

        train_idx=0
        train_fn_list=[]
        middle_dir=f'train_list_v{voxel_size}_s{voxel_stride}_io{str_image_overlap}_vo{str_voxel_overlap}'
        save_dir=os.path.join(os.path.dirname(path), middle_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save positive voxels
        for voxel_name, data in voxel_data.items():
            if 'image_names' in data and len(data['image_names']) >= n_image_thresh:
                print(f"{voxel_name}: {len(data['image_names'])} images")
                
                if vis_result:
                    voxel_points = data['points']
                    debug_point_cloud = o3d.geometry.PointCloud()
                    debug_point_cloud.points = o3d.utility.Vector3dVector(voxel_points)
                    o3d.io.write_point_cloud(f'z_debug_submap/{scene_name}_v{voxel_size}_{train_idx}.ply', debug_point_cloud)
                del data['points']

                fn='train_all_{}_{}.npy'.format(voxel_size, train_idx)
                path=os.path.join(save_dir, fn)
                np.save(path, data)
                train_fn_list.append(fn)

                train_idx+=1

        # untrained voxels
        for voxel_name, data in voxel_data.items():
            if 'image_names' not in data:
                print(f"{voxel_name}: 0 images")

                if vis_result:
                    voxel_points = data['points']
                    debug_point_cloud = o3d.geometry.PointCloud()
                    debug_point_cloud.points = o3d.utility.Vector3dVector(voxel_points)
                    o3d.io.write_point_cloud(f'z_debug_submap/{scene_name}_v{voxel_size}_{train_idx}_no_img.ply', debug_point_cloud)
                del data['points']

                fn='train_all_no_img_{}_{}.npy'.format(voxel_size, train_idx)
                path=os.path.join(save_dir, fn)
                np.save(path, data)
                train_fn_list.append(fn)

                train_idx+=1

            elif len(data['image_names']) < n_image_thresh:
                print(f"{voxel_name}: {len(data['image_names'])} images")
                del voxel_data[voxel_name]['image_names']

                if vis_result:
                    voxel_points = data['points']
                    debug_point_cloud = o3d.geometry.PointCloud()
                    debug_point_cloud.points = o3d.utility.Vector3dVector(voxel_points)
                    o3d.io.write_point_cloud(f'z_debug_submap/{scene_name}_v{voxel_size}_{train_idx}_no_img.ply', debug_point_cloud)
                del data['points']

                fn='train_all_no_img_{}_{}.npy'.format(voxel_size, train_idx)
                path=os.path.join(save_dir, fn)
                np.save(path, data)
                train_fn_list.append(fn)

                train_idx+=1

        # save train files
        with open(os.path.join(root_path, 'train_list/{}.txt'.format(output_path)),'w') as f:
            for fn in train_fn_list:
                path=os.path.join(os.path.dirname(input_path), middle_dir, fn)
                f.write(path+'\n')
