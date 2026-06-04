from torchvision import transforms
import os
import random
import numpy as np
from torch.utils import data
import pickle as pkl
import cv2
import torch
import quaternion
import random
import open3d as o3d
from PIL import Image
import math
import time
from scipy.spatial import cKDTree

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

class FarthestSampler:
    def __init__(self, dim=3):
        self.dim = dim

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=0)

    def sample(self, pts, k):
        farthest_pts = np.zeros((self.dim, k))
        farthest_pts_idx = np.zeros(k, dtype=np.int)
        init_idx = np.random.randint(len(pts))
        farthest_pts[:, 0] = pts[:, init_idx]
        farthest_pts_idx[0] = init_idx
        distances = self.calc_distances(farthest_pts[:, 0:1], pts)
        for i in range(1, k):
            idx = np.argmax(distances)
            farthest_pts[:, i] = pts[:, idx]
            farthest_pts_idx[i] = idx
            distances = np.minimum(distances, self.calc_distances(farthest_pts[:, i:i+1], pts))
        return farthest_pts, farthest_pts_idx
    
class KaptureDatasetFeatureMatch(data.Dataset):
    def __init__(self, config, root, kapture_data, sensor_dict, train_path, input_path, mode='train',image_size=640, max_n_points=20000, random_crop=True, rgb=False, aspect_ratio=None):
        self.config = config
        self.max_n_points=max_n_points
        self.train_path=train_path
        self.root=root
        self.mode=mode
        self.image_size=image_size
        self.input_path=input_path
        self.random_crop=random_crop
        self.rgb=rgb
        self.aspect_ratio=aspect_ratio
        self.farthest_sampler = FarthestSampler(dim=3)
        
        if self.config['feature_match']['rgb']:
            if self.mode=='train':
                clr_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=[-0.1,0.1])
            else:
                clr_jitter = transforms.ColorJitter(saturation=[1,1])
        else:
            if self.mode=='train':
                clr_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=[0,0])
            else:
                clr_jitter = transforms.ColorJitter(saturation=[0,0])
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            clr_jitter,
            transforms.ToTensor()
            ])
        self.image_transform_dust = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.sensor_dict=sensor_dict
        self.kaptures=kapture_data
        self.pcd_size = self.config['feature_match']['pcd_size']

        self.load_meta()
        
        if self.mode=='train':
            # get voxel points data
            scene_name = self.train_path.split('/')[0]
            pcd_file_suffix = self.config['feature_match']['pcd_file_suffix']
            point_cloud_file = os.path.join(self.input_path, f'pcd_{scene_name}_{pcd_file_suffix}.ply')
            print(f"load pcd file from {point_cloud_file}")
            pcd = o3d.io.read_point_cloud(point_cloud_file)
            pcd_points = np.array(pcd.points)

            voxel_path = os.path.join(self.root, self.voxel_id)
            voxel_info=np.load(voxel_path, allow_pickle=True).item()

            mean = voxel_info['xyz_mean'][:3]
            median = voxel_info['xyz_median'][:3]
            std = voxel_info['xyz_std'][:3]
            voxel_min = voxel_info['xyz_min'][:3].astype(np.float32)
            voxel_max = voxel_info['xyz_max'][:3].astype(np.float32)
            voxel_center = (voxel_min + voxel_max) / 2
            voxel_size = (voxel_max - voxel_min)[0]

            voxel_points = pcd_points[np.all(pcd_points >= voxel_min, axis=1) & np.all(pcd_points <= voxel_max, axis=1)]
            voxel_points = voxel_points.astype(np.float32)
            
            if self.config['feature_match']['cube_norm']:
                voxel_points = (voxel_points - voxel_center) / (voxel_size / 2) # normalize in [-1,1] unit cube
                voxel_mean = voxel_center
                voxel_scalar = voxel_size / 2
            else:
                raise ValueError("you should use cube_norm") 

            self.voxel_points = voxel_points
            self.voxel_mean = voxel_mean.astype(np.float32)
            self.voxel_scalar = voxel_scalar.astype(np.float32)
            self.voxel_min = voxel_min.astype(np.float32)
            self.voxel_max = voxel_max.astype(np.float32)
            None
        else:
            # save pointmap for each voxel
            self.voxel_id_to_voxel_points = {}
            self.voxel_id_to_voxel_mean = {}
            self.voxel_id_to_voxel_scalar = {}
            self.voxel_id_to_voxel_min = {}
            self.voxel_id_to_voxel_max = {}
            scene_name = self.train_path.split('/')[0]
            pcd_file_suffix = self.config['feature_match']['pcd_file_suffix']
            point_cloud_file = os.path.join(self.input_path, f'pcd_{scene_name}_{pcd_file_suffix}.ply')
            print(f"load pcd file from {point_cloud_file}")
            pcd = o3d.io.read_point_cloud(point_cloud_file)
            pcd_points = np.array(pcd.points)

            for voxel_id_list in self.voxel_id:
                for voxel_id in voxel_id_list:
                    if voxel_id not in self.voxel_id_to_voxel_points:
                        voxel_name = os.path.basename(voxel_id)[:-4]
                        voxel_path = os.path.join(self.root, voxel_id)
                        voxel_info=np.load(voxel_path, allow_pickle=True).item()

                        mean = voxel_info['xyz_median'][:3]
                        std = voxel_info['xyz_std'][:3]
                        voxel_min = voxel_info['xyz_min'][:3].astype(np.float32)
                        voxel_max = voxel_info['xyz_max'][:3].astype(np.float32)
                        voxel_center = (voxel_min + voxel_max) / 2
                        voxel_size = (voxel_max - voxel_min)[0]

                        voxel_points = pcd_points[np.all(pcd_points >= voxel_min, axis=1) & np.all(pcd_points <= voxel_max, axis=1)]
                        voxel_points = voxel_points.astype(np.float32)
                        
                        target_size = self.pcd_size
                        if len(voxel_points) >= target_size:
                            indices=np.random.choice(len(voxel_points), size=target_size, replace=False)
                            voxel_points = voxel_points[indices]
                        else:
                            fix_idx = np.asarray(range(len(voxel_points)))
                            while len(voxel_points) + fix_idx.shape[0] < target_size:
                                fix_idx = np.concatenate((fix_idx, np.asarray(range(len(voxel_points)))), axis=0)
                            random_idx = np.random.choice(len(voxel_points), target_size - fix_idx.shape[0], replace=False)
                            indices = np.concatenate((fix_idx, random_idx), axis=0)
                            voxel_points = voxel_points[indices]
                        
                        if self.config['feature_match']['cube_norm']:
                            voxel_points = (voxel_points - voxel_center) / (voxel_size / 2) # normalize in [-1,1] unit cube
                            voxel_mean = voxel_center
                            voxel_scalar = voxel_size / 2
                        else:
                            raise ValueError("you should use cube_norm") 
                        
                        self.voxel_id_to_voxel_points[voxel_id] = voxel_points
                        self.voxel_id_to_voxel_scalar[voxel_id] = voxel_scalar
                        self.voxel_id_to_voxel_mean[voxel_id] = voxel_mean
                        self.voxel_id_to_voxel_min[voxel_id] = voxel_min
                        self.voxel_id_to_voxel_max[voxel_id] = voxel_max

    def load_meta(self):
        if self.mode=='train':
            self.voxel_id=self.train_path
            info=np.load(os.path.join(self.root, self.train_path), allow_pickle=True).item()
            self.idx_list=info['image_names']           
        else:
            self.idx_list=[]
            query_voxel_id_map=np.load(os.path.join(self.root, self.train_path),allow_pickle=True).item()
            self.idx_list=[]
            self.voxel_id=[]
            
            for k, v in query_voxel_id_map.items():
                self.idx_list.append(k)
                self.voxel_id.append(list(v))

    def __len__(self):
        return len(self.idx_list)
    
    def load_pose(self, timestep, sensor_id):
        
        if self.kaptures.trajectories is not None and (timestep, sensor_id) in self.kaptures.trajectories:
            pose_world_to_cam = self.kaptures.trajectories[(timestep, sensor_id)]
            pose_world_to_cam_matrix = np.zeros((4, 4), dtype=np.float)
            pose_world_to_cam_matrix[0:3, 0:3] = quaternion.as_rotation_matrix(pose_world_to_cam.r)
            pose_world_to_cam_matrix[0:3, 3] = pose_world_to_cam.t_raw
            pose_world_to_cam_matrix[3, 3] = 1.0
            T = torch.tensor(pose_world_to_cam_matrix).float()
            gt_pose=T.inverse() # gt_pose为从cam_to_world
        else:
            gt_pose=T=torch.eye(4)
        return gt_pose, pose_world_to_cam

    def __getitem__(self, index):
        image_id = self.idx_list[index]
        timestep, sensor_id=self.sensor_dict[image_id]

        # load image
        if self.rgb:
            # print(os.path.join(self.input_path, 'sensors/records_data', image_id))
            image = cv2.imread(os.path.join(self.input_path, 'sensors/records_data', image_id))
            if image is None:
                print(os.path.join(self.input_path, 'sensors/records_data', image_id))
                return self.__getitem__(random.randint(0, len(self.idx_list)-1))
            image_h, image_w, _=image.shape
        else:
            image = cv2.imread(os.path.join(self.input_path, 'sensors/records_data', image_id), cv2.IMREAD_GRAYSCALE)
            image_h, image_w=image.shape
                    
        scale_factor = self.image_size / min(image_h, image_w)
        resize_h = int(image_h * scale_factor)
        resize_w = int(image_w * scale_factor)
        
        
        # load camera parameters
        camera_params=np.array(self.kaptures.sensors[sensor_id].camera_params[2:])
        camera_type=str(self.kaptures.sensors[sensor_id].camera_type).split('.')[-1]
        gt_pose, _ =self.load_pose(timestep, sensor_id) # camera to world
        
        T_w2c = np.linalg.inv(gt_pose.numpy())  # world to camera
        f_original = camera_params[0]
        
        if self.mode=='train':
            # 1. process point cloud
            new_voxel_mean = self.voxel_mean
            new_voxel_min = self.voxel_min
            new_voxel_max = self.voxel_max
            new_voxel_scalar = self.voxel_scalar
            new_voxel_points = self.voxel_points
            
            # 2. intrinsic, depth map, image center crop ...
            depth_map_path = os.path.join(self.input_path, 'sensors/depth_data', image_id.replace("image", "depth"))
            depth_map = Image.open(depth_map_path) # RGBA
            depth_map = np.array(depth_map)
            
            R = depth_map[:,:,0].astype(np.float32)
            G = depth_map[:,:,1].astype(np.float32)
            B = depth_map[:,:,2].astype(np.float32)
            normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1)
            depth_map = 1000 * normalized
            
            center_crop = self.config['feature_match']['center_crop']
            if center_crop:
                # random centered crop augmentation
                min_crop_scale = 1280 // 16
                max_crop_scale = 1920 // 16
                
                crop_scale = np.random.randint(min_crop_scale, max_crop_scale+1)
                crop_width = 16*crop_scale
                crop_height = 9*crop_scale
                
                # crop depth map
                height, width = depth_map.shape
                start_x = (width - crop_width) // 2
                start_y = (height - crop_height) // 2
                depth_map = depth_map[start_y:start_y + crop_height, start_x:start_x + crop_width]
                
                # crop image
                image = image[start_y:start_y + crop_height, start_x:start_x + crop_width, :]
                
                # change corresponding intrinsic 
                f_new = f_original * resize_w/crop_width
                intrinsic = np.array([[f_new,0,resize_w/2],
                                    [0,f_new,resize_h/2],
                                    [0,0,1]])
            else:
                f_new = f_original * resize_w/image_w
                intrinsic = np.array([[f_new,0,resize_w/2],
                                [0,f_new,resize_h/2],
                                [0,0,1]])
            
            depth_map_image = Image.fromarray(depth_map)
            new_size = (resize_w, resize_h)
            resized_depth_map_nearest = depth_map_image.resize(new_size, Image.NEAREST)
            depth_map = np.array(resized_depth_map_nearest)

            # load point cloud submap
            input_voxel_points_size = self.pcd_size

            if len(new_voxel_points) >= input_voxel_points_size:
                indices=np.random.choice(len(new_voxel_points), size=input_voxel_points_size, replace=False)
                input_voxel_points = new_voxel_points[indices]
            else:
                fix_idx = np.asarray(range(len(new_voxel_points)))
                while len(new_voxel_points) + fix_idx.shape[0] < input_voxel_points_size:
                    fix_idx = np.concatenate((fix_idx, np.asarray(range(len(new_voxel_points)))), axis=0)
                random_idx = np.random.choice(len(new_voxel_points), input_voxel_points_size - fix_idx.shape[0], replace=False)
                indices = np.concatenate((fix_idx, random_idx), axis=0)
                input_voxel_points = new_voxel_points[indices]
                
            # project point cloud in image
            origin_voxel_points = input_voxel_points * new_voxel_scalar + new_voxel_mean # transform pc to origin scale
            cam_coord = np.dot(T_w2c[0:3, 0:3], origin_voxel_points.T) + T_w2c[0:3, 3:]  # transform pc to camera coordinate system
            proj_coord = np.dot(intrinsic, cam_coord) # 3 x num_point
            proj_xy = proj_coord[0:2, :] / (proj_coord[2, :] + 1e-9)
            proj_xy_int = np.floor(proj_xy).astype(np.int32)
            proj_depth = proj_coord[2,:]
            min_depth = depth_map[np.clip(proj_xy_int[1,:], 0, resize_h-1), np.clip(proj_xy_int[0,:], 0, resize_w-1)]
            
            depth_thres = 3
            board = self.config['feature_match']['in_image_board']
            
            # point project insiade camera frustum
            in_frustum_mask = (proj_xy_int[0, :] >= board) & (proj_xy_int[0, :] <= resize_w-1-board) & \
                (proj_xy_int[1, :] >= board) & (proj_xy_int[1, :] <= resize_h-1-board) & \
                (proj_depth > 0)  # num_point
            
            # point project into image and no occlusion
            in_image_mask = (proj_xy_int[0, :] >= board) & (proj_xy_int[0, :] <= resize_w-1-board) & \
                (proj_xy_int[1, :] >= board) & (proj_xy_int[1, :] <= resize_h-1-board) & \
                (proj_depth > 0) & (proj_depth < min_depth + depth_thres) # num_point
            
            
            patch_size = self.config['feature_match']['coarse_patch_size']
            num_patch_row = resize_w / patch_size
            
            proj_patch = (proj_xy_int // patch_size).T
            proj_patch_idx = proj_patch[:, 1] * num_patch_row + proj_patch[:, 0]
            # proj_patch_idx[~in_frustum_mask] = -1
            proj_patch_idx[~in_image_mask] = -1
            # sel_proj_patch = proj_patch[:, in_image_pc_idx].T
            # sel_proj_patch_idx = sel_proj_patch[:, 1] * num_patch_row + sel_proj_patch[:, 0]
            # print(np.max(sel_proj_patch, axis=0))
            # print(sel_proj_patch_idx)
            
                
            # augmentation
            if self.config['feature_match']['pcd_aug']:
                # Rotation along z axis
                rot_angle = (np.random.random()*np.pi*2) - np.pi # -180 ~ +180 degree
                rot_mat_z = rotz(rot_angle)
                input_voxel_points[:,0:3] = input_voxel_points[:,0:3] @ rot_mat_z.T
                
                # translation along xy
                aug_trans = np.random.uniform(-0.1, 0.1, 3)
                aug_trans[2] = 0
                input_voxel_points[:,0:3] = input_voxel_points[:,0:3] + aug_trans

            # color jitter
            image=self.image_transform(image)

            if self.config['feature_match']['dust_backbone']:
                image = self.image_transform_dust(image)

            voxel_id=self.voxel_id
            
        else: # val and test
            C, H, W=image.shape

            voxel_id=self.voxel_id[index]

            # load dense depth map and dense points
            depth_map_path = os.path.join(self.input_path, 'sensors/depth_data', image_id.replace("image", "depth"))
            depth_map = Image.open(depth_map_path) # RGBA
            depth_map = np.array(depth_map)
            
            R = depth_map[:,:,0].astype(np.float32)
            G = depth_map[:,:,1].astype(np.float32)
            B = depth_map[:,:,2].astype(np.float32)
            
            normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1)
            
            depth_map = 1000 * normalized
            
            fov = 90  # horizontal fov
            f = resize_w / (2 * math.tan(fov/2 * math.pi/180))

            intrinsic = np.array([[f,0,resize_w/2],
                                [0,f,resize_h/2],
                                [0,0,1]])
                
            depth_map_image = Image.fromarray(depth_map)
            new_size = (resize_w, resize_h)
            resized_depth_map_nearest = depth_map_image.resize(new_size, Image.NEAREST)
            depth_map = np.array(resized_depth_map_nearest)
            
            image=self.image_transform(image)
            if self.config['feature_match']['dust_backbone']:
                image = self.image_transform_dust(image)

                
            input_voxel_points = []
            self.voxel_scalar = []
            self.voxel_mean = []
            proj_patch_list = []
            proj_patch_idx_list = []
            proj_xy_int_list = []
            proj_xy_list = []
            
            for vox_id in voxel_id:
                
                sample_pcd = self.voxel_id_to_voxel_points[vox_id]
                
                # load point cloud submap
                input_voxel_points_size = self.pcd_size

                # random sampling point clouds
                if len(sample_pcd) > input_voxel_points_size:
                    indices = np.random.choice(len(sample_pcd), size=input_voxel_points_size, replace=False)
                    sample_pcd = sample_pcd[indices]
                elif len(sample_pcd) < input_voxel_points_size:
                    indices = np.random.choice(len(sample_pcd), size=input_voxel_points_size, replace=True)
                    sample_pcd = sample_pcd[indices]
                    
                voxel_mean = self.voxel_id_to_voxel_mean[vox_id]
                voxel_scalar = self.voxel_id_to_voxel_scalar[vox_id]
                voxel_min = self.voxel_id_to_voxel_min[vox_id]
                voxel_max = self.voxel_id_to_voxel_max[vox_id]
                
                input_voxel_points.append(sample_pcd)
                self.voxel_scalar.append(voxel_scalar)
                self.voxel_mean.append(voxel_mean)
                
                # compute points in camera coordinate
                scale_voxel_points = sample_pcd * voxel_scalar + voxel_mean
                cam_coord = np.dot(T_w2c[0:3, 0:3], scale_voxel_points.T) + T_w2c[0:3, 3:]  # transform pc to camera coordinate system
                proj_coord = np.dot(intrinsic, cam_coord) # 3 x num_point
                proj_xy = proj_coord[0:2, :] / proj_coord[2, :]
                proj_xy_int = np.floor(proj_xy).astype(np.int32)
                proj_depth = proj_coord[2,:]
                min_depth = depth_map[np.clip(proj_xy_int[1,:], 0, resize_h-1), np.clip(proj_xy_int[0,:], 0, resize_w-1)]
                depth_thres = 0.5
                
                in_frustum_mask_one = (proj_xy[0, :] >= 0) & (proj_xy[0, :] <= resize_w-1) & \
                    (proj_xy[1, :] >= 0) & (proj_xy[1, :] <= resize_h-1) & \
                    (proj_coord[2, :] > 0) # num_point
                
                
                patch_size = self.config['feature_match']['coarse_patch_size']
                num_patch_row = resize_w / patch_size

                    
                proj_patch = (proj_xy_int // patch_size).T
                proj_patch_idx = proj_patch[:, 1] * num_patch_row + proj_patch[:, 0]
                proj_patch_idx[~in_frustum_mask_one] = -1
                proj_patch_idx_list.append(proj_patch_idx.astype(np.int64))
                proj_patch_list.append(proj_patch)
                proj_xy_int_list.append(proj_xy_int)
                proj_xy_list.append(proj_xy)
                
        if self.mode=='test':
            image_name=image_id           
        else:
            image_name='kapture'

        # for point backbone Point Transformer
        if self.config['feature_match']['point_backbone'] == 'pt':
            num_node = self.config['feature_match']['pt_num_node']
            if self.mode == 'train':
                # # <------ sample the firs-level downsampled points, namely node ------>
                node_np, _ = self.farthest_sampler.sample(input_voxel_points[np.random.choice(input_voxel_points.shape[0], \
                                                            num_node*self.config['feature_match']['pt_rand'], replace=False), :].T,
                                                            k = num_node)

                node_np = node_np.astype(np.float32)
                # <------ construct the node-to-point index ------>
                kdtree = cKDTree(node_np.T)
                _, point2node = kdtree.query(input_voxel_points, k=1)
                kdtree = cKDTree(input_voxel_points)
                _, node2point = kdtree.query(node_np.T, k=1)
                
                # get sparse 
                num_proxy = self.config['feature_match']['pt_num_proxy']
                fps_idx = node2point[:num_proxy] # proxy2point
                
                proxy_in_image_mask = in_image_mask[fps_idx]
                
                coarse_pc_idx = np.where(proxy_in_image_mask==1)[0]
                if len(coarse_pc_idx) == 0:
                    print(f"len coarse_pc_idx is 0")
                    return self.__getitem__(random.randint(0, len(self.idx_list)-1))
                if len(coarse_pc_idx) >= 64:
                    indices=np.random.permutation(len(coarse_pc_idx))[0:64]
                else:
                    fix_idx = np.asarray(range(len(coarse_pc_idx)))
                    while len(coarse_pc_idx) + fix_idx.shape[0] < 64:
                        fix_idx = np.concatenate((fix_idx, np.asarray(range(len(coarse_pc_idx)))), axis=0)
                    random_idx = np.random.choice(len(coarse_pc_idx), 64 - fix_idx.shape[0], replace=False)
                    indices = np.concatenate((fix_idx, random_idx), axis=0)
                coarse_indices=coarse_pc_idx[indices]
                coarse_pc_idx = fps_idx[coarse_indices]
                
                coarse_pc_out_idx = np.where(proxy_in_image_mask==0)[0]
                if len(coarse_pc_out_idx) == 0:
                    print(f"len coarse_pc_out_idx is 0")
                    return self.__getitem__(random.randint(0, len(self.idx_list)-1))
                if len(coarse_pc_out_idx) >= 64:
                    indices=np.random.permutation(len(coarse_pc_out_idx))[0:64]
                else:
                    fix_idx = np.asarray(range(len(coarse_pc_out_idx)))
                    while len(coarse_pc_out_idx) + fix_idx.shape[0] < 64:
                        fix_idx = np.concatenate((fix_idx, np.asarray(range(len(coarse_pc_out_idx)))), axis=0)
                    random_idx = np.random.choice(len(coarse_pc_out_idx), 64 - fix_idx.shape[0], replace=False)
                    indices = np.concatenate((fix_idx, random_idx), axis=0)
                coarse_out_indices=coarse_pc_out_idx[indices]
                coarse_pc_out_idx = fps_idx[coarse_out_indices]
                
            else:
                node_np = []
                point2node = []
                node2point = []
                for one_pcd in input_voxel_points:
                    one_node_np, _ = self.farthest_sampler.sample(one_pcd[np.random.choice(one_pcd.shape[0], \
                                                            num_node*self.config['feature_match']['pt_rand'], replace=False), :].T,
                                                            k = num_node)
                    one_node_np = one_node_np.astype(np.float32)
                    
                    kdtree = cKDTree(one_node_np.T)
                    _, one_point2node = kdtree.query(one_pcd, k=1)
                    kdtree = cKDTree(one_pcd)
                    _, one_node2point = kdtree.query(one_node_np.T, k=1)
                    
                    node_np.append(one_node_np)
                    point2node.append(one_point2node)
                    node2point.append(one_node2point)

        else:
            if self.mode == 'train':
                node_np = torch.tensor(0)
                point2node = torch.tensor(0)
                node2point = torch.tensor(0)
            else:
                node_np = [0] * len(input_voxel_points)
                point2node = [0] * len(input_voxel_points)
                node2point = [0] * len(input_voxel_points)
        
        if self.mode == 'train' and self.config['feature_match']['point_backbone'] == 'pt' and self.config['feature_match']['att_loss']:
            # node point 1280x3 in camera coord
            pc_in_cam = cam_coord
            node_in_cam = pc_in_cam.T[fps_idx]  # node coord in cam coord frame
            x = np.arange(resize_w, step=patch_size) + patch_size//2
            y = np.arange(resize_h, step=patch_size) + patch_size//2
            xv, yv = np.meshgrid(x, y)
            keypoints = np.vstack([xv.ravel(), yv.ravel()]).T
            keypoints = keypoints.astype(np.int16)
            camera_ray = np.concatenate([keypoints, np.ones((keypoints.shape[0], 1))], axis=1)
            camera_ray = (np.linalg.inv(intrinsic) @ camera_ray.T).T  # n_H*n_W camera ray's direction in cam coord frame
            
            node_in_cam_norm = node_in_cam / np.linalg.norm(node_in_cam, axis=1, keepdims=True)
            camera_ray_norm = camera_ray / np.linalg.norm(camera_ray, axis=1, keepdims=True)
            
            patch_to_node_rad = camera_ray_norm @ node_in_cam_norm.T
            patch_to_node_rad = np.clip(patch_to_node_rad, -1.0, 1.0)
            patch_to_node_rad = np.arccos(patch_to_node_rad)
            patch_to_node_rad_mask = patch_to_node_rad < self.config['feature_match']['att_loss_i2p_rad_low'] * (np.pi / 180) # < 10 deg
            patch_to_node_rad_mask = patch_to_node_rad_mask * proxy_in_image_mask[np.newaxis, :]
            patch_to_node_rad_mask_neg = patch_to_node_rad > self.config['feature_match']['att_loss_i2p_rad_up'] * (np.pi / 180) # > 20 deg
            valid_patch_mask = np.sum(patch_to_node_rad_mask, axis=-1) > 10
            patch_to_node_rad_mask_neg = patch_to_node_rad_mask_neg * valid_patch_mask[:, np.newaxis]
            
            points_exp = node_in_cam[:, np.newaxis, :]  # Nx1x3
            rays_exp = camera_ray_norm[np.newaxis, :, :]  # 1xMx3
            cross_product = np.cross(points_exp, rays_exp)  # NxMx3
            node_to_patch_dist = np.linalg.norm(cross_product, axis=2)  # NxM
            node_to_patch_dist_mask = node_to_patch_dist < self.config['feature_match']['att_loss_p2i_dist_low'] # < 3 m
            node_to_patch_dist_mask = node_to_patch_dist_mask * proxy_in_image_mask[:, np.newaxis]
            node_to_patch_dist_mask_neg = node_to_patch_dist > self.config['feature_match']['att_loss_p2i_dist_up'] # > 5 m
            node_to_patch_dist_mask_neg = node_to_patch_dist_mask_neg * proxy_in_image_mask[:, np.newaxis]
            
        
        data = {
            # for att loss
            "patch_to_node_rad_mask":patch_to_node_rad_mask if self.mode == 'train' and self.config['feature_match']['point_backbone'] == 'pt' and self.config['feature_match']['att_loss'] else torch.tensor(0),
            "patch_to_node_rad_mask_neg":patch_to_node_rad_mask_neg if self.mode == 'train' and self.config['feature_match']['point_backbone'] == 'pt' and self.config['feature_match']['att_loss'] else torch.tensor(0),
            "node_to_patch_dist_mask":node_to_patch_dist_mask if self.mode == 'train' and self.config['feature_match']['point_backbone'] == 'pt' and self.config['feature_match']['att_loss'] else torch.tensor(0),
            "node_to_patch_dist_mask_neg":node_to_patch_dist_mask_neg if self.mode == 'train' and self.config['feature_match']['point_backbone'] == 'pt' and self.config['feature_match']['att_loss'] else torch.tensor(0),
            
            "coarse_indices": coarse_indices.astype(np.int64) if self.mode=='train' and self.config['feature_match']['point_backbone'] == 'pt' else torch.tensor(0),
            "coarse_pc_idx": coarse_pc_idx.astype(np.int64) if self.mode=='train' and self.config['feature_match']['point_backbone'] == 'pt' else torch.tensor(0),
            "coarse_out_indices": coarse_out_indices.astype(np.int64) if self.mode=='train' and self.config['feature_match']['point_backbone'] == 'pt' else torch.tensor(0),
            "coarse_pc_out_idx": coarse_pc_out_idx.astype(np.int64) if self.mode=='train' and self.config['feature_match']['point_backbone'] == 'pt' else torch.tensor(0),
            
            "proj_patch_idx": proj_patch_idx.astype(np.int64) if self.mode=='train' else proj_patch_idx_list,
            "proj_patch": proj_patch.astype(np.int64) if self.mode=='train' else proj_patch_list,
            "proj_xy_int": proj_xy_int.astype(np.int32) if self.mode=='train' else proj_xy_int_list,
            "proj_xy": proj_xy if self.mode=='train' else proj_xy_list,
            
            "intrinsic": intrinsic.astype(np.float32), # intrinsic of the actual input image
            
            'image_h':image_h, 
            'image_w':image_w,
            'image': image, 
            'gt_pose' : gt_pose,
            'camera_params': camera_params,  # (3)
            'camera_type':camera_type,
            'dataset_name': 'carla',
            'scale_factor':scale_factor,
            'image_name': image_name,
            'voxel_id':voxel_id,
            'image_path': os.path.join(self.input_path, 'sensors/records_data', image_id),
            
            # input point cloud
            'voxel_points': input_voxel_points.astype(np.float32) if self.mode=='train' else input_voxel_points,  # (N x 3) when train, a list when eval
            'voxel_scalar': new_voxel_scalar if self.mode=='train' else self.voxel_scalar,
            'voxel_mean': new_voxel_mean if self.mode=='train' else self.voxel_mean,
            
            # pt
            'voxel_nodes': node_np,
            'point2node': point2node,
            'node2point': node2point,
            
        }
        return data