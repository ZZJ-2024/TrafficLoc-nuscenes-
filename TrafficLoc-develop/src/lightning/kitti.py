import os
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import math
import open3d as o3d
import cv2
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.sparse import coo_matrix
from scipy.ndimage import binary_dilation
import time
from pathlib import Path
from scipy.spatial import cKDTree
from open3d.ml.torch.layers import KNNSearch

def precompute_point_cloud_stack_mode(points, intensity, normals, lengths, num_stages):
    # assert num_stages == len(neighbor_limits)
    radius_num = 128
    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(np.transpose(points))

    # grid subsampling
    for i in range(num_stages):
        if i > 0:
            # random sample half points of last stage(except stage1)
            pcd = pcd.random_down_sample(0.5)
            # print(pcd.shape)

        points_list.append(torch.Tensor(np.asarray(pcd.points)))
        lengths_list.append(int(lengths))
        lengths = lengths // 2


    # radius search
    for i in range(num_stages):
        nsearch = KNNSearch(return_distances=True)
        # fixed_radius_search = FixedRadiusSearch(return_distances=True)

        cur_points = points_list[i]
        ml_neighbors = nsearch(cur_points, cur_points,radius_num)


        neighbors = ml_neighbors.neighbors_index.reshape(lengths_list[i], radius_num)
        neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            # sub_lengths = lengths_list[i + 1]

            subsampling = nsearch(cur_points, sub_points, radius_num).neighbors_index.reshape(lengths_list[i+1], radius_num)
            subsampling_list.append(subsampling)

            upsampling = nsearch(sub_points, cur_points, radius_num).neighbors_index.reshape(lengths_list[i], radius_num)
            # upsampling = radius_search(sub_pcdtree, lengths_list[i], cur_points, radius=radius * 2, neighbor_limits=100, mode='upsample')
            upsampling_list.append(upsampling)

    return {
        'points': points_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
            }
    

def square_distance(src, tgt, normalize=False):
    '''
    Calculate Euclide distance between every two points
    :param src: source point cloud in shape [B, N, C]
    :param tgt: target point cloud in shape [B, M, C]
    :param normalize: whether to normalize calculated distances
    :return:
    '''

    B, N, _ = src.shape
    _, M, _ = tgt.shape
    dist = -2. * torch.matmul(src, tgt.permute(0, 2, 1).contiguous())
    if normalize:
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
        dist += torch.sum(tgt ** 2, dim=-1).unsqueeze(-2)

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist

def point_to_node(nodes, points):
    '''
    Assign each point to a certain node according to nearest neighbor search
    :param nodes: [M, 3]
    :param points: [N, 3]
    :return: idx [N], indicating the id of node that each point belongs to
    '''
    # M, _ = nodes.size()
    # N, _ = points.size()
    dist = square_distance(points.unsqueeze(0), nodes.unsqueeze(0))[0]

    idx = dist.topk(k=1, dim=-1, largest=False)[1] #[B, N, 1], ignore the smallest element as it's the query itself

    idx = idx.squeeze(-1)
    return idx

class KittiCalibHelper:
    def __init__(self, root_path):
        self.root_path = root_path
        self.calib_matrix_dict = self.read_calib_files()

    def read_calib_files(self):
        seq_folders = [name for name in os.listdir(
            os.path.join(self.root_path, 'calib'))]
        calib_matrix_dict = {}
        for seq in seq_folders:
            calib_file_path = os.path.join(
                self.root_path, 'calib', seq, 'calib.txt')
            with open(calib_file_path, 'r') as f:
                for line in f.readlines():
                    seq_int = int(seq)
                    if calib_matrix_dict.get(seq_int) is None:
                        calib_matrix_dict[seq_int] = {}

                    key = line[0:2]
                    mat = np.fromstring(line[4:], sep=' ').reshape(
                        (3, 4)).astype(np.float32)
                    if 'Tr' == key:
                        P = np.identity(4)
                        P[0:3, :] = mat
                        calib_matrix_dict[seq_int][key] = P
                    else:
                        K = mat[0:3, 0:3]
                        calib_matrix_dict[seq_int][key + '_K'] = K
                        fx = K[0, 0]
                        fy = K[1, 1]
                        cx = K[0, 2]
                        cy = K[1, 2]
                        # mat[0, 3] = fx*tx + cx*tz
                        # mat[1, 3] = fy*ty + cy*tz
                        # mat[2, 3] = tz
                        tz = mat[2, 3]
                        tx = (mat[0, 3] - cx * tz) / fx
                        ty = (mat[1, 3] - cy * tz) / fy
                        P = np.identity(4)
                        P[0:3, 3] = np.asarray([tx, ty, tz])
                        calib_matrix_dict[seq_int][key] = P
        return calib_matrix_dict

    def get_matrix(self, seq: int, matrix_key: str):
        return self.calib_matrix_dict[seq][matrix_key]

class FarthestSampler:
    def __init__(self, dim=3):
        self.dim = dim

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=0)

    def sample(self, pts, k):
        farthest_pts = np.zeros((self.dim, k))
        farthest_pts_idx = np.zeros(k, dtype=int)
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


class kitti_pc_img_dataset(data.Dataset):
    def __init__(self, opt,mode, config):
        super(kitti_pc_img_dataset, self).__init__()
        for k,v in opt.__dict__.items():
            setattr(self,k,v)
        self.mode = mode
        if config['feature_match']['nuscenes']:
            self.dataset = self.make_nuscenes_dataset(self.data_path, mode)
        else:
            self.dataset = self.make_kitti_dataset(self.data_path, mode)
            self.calibhelper = KittiCalibHelper(self.data_path)
        self.farthest_sampler = FarthestSampler(dim=3)
        self.config = config
        print("%s set: %d frames"%(mode,len(self.dataset)))
        print('load %s data complete'%mode)
        
        if self.mode=='train':
            clr_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=[0,0])
        else:
            clr_jitter = transforms.ColorJitter(saturation=[0,0])
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_H),
            clr_jitter,
            transforms.ToTensor()
            ])
        self.image_transform_dust = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def read_velodyne_bin(self, path):

        pc_list = []
        with open(path, 'rb') as f:
            content = f.read()
            pc_iter = struct.iter_unpack('ffff', content)
            for idx, point in enumerate(pc_iter):
                pc_list.append([point[0], point[1], point[2], point[3]])
        return np.asarray(pc_list, dtype=np.float32).T

    def make_nuscenes_dataset(self, root_path, mode):
        if mode == 'train':
            data_dir = os.path.join(root_path,"train")
        elif mode == "val" or mode == "test":
            data_dir = os.path.join(root_path,"test")
        else:
            raise Exception('Invalid mode.')

        dataset = sorted(os.listdir(os.path.join(data_dir , 'PC')))

        return dataset
    
    def make_kitti_dataset(self, root_path, mode):
        dataset = []

        if mode == 'train':
            seq_list = list(range(9))
        elif 'val' == mode:
            seq_list = [9, 10]
        else:
            raise Exception('Invalid mode.')

        skip_start_end = 0
        for seq in seq_list:
            img2_folder = os.path.join(
                root_path, 'sequences', '%02d' % seq, 'img_P2')
            img3_folder = os.path.join(
                root_path, 'sequences', '%02d' % seq, 'img_P3')
            pc_folder = os.path.join(
                root_path, 'sequences', '%02d' % seq, 'pc_npy_with_normal')

            K2_folder = os.path.join(
                root_path, 'sequences', '%02d' % seq, 'K_P2')
            K3_folder = os.path.join(
                root_path, 'sequences', '%02d' % seq, 'K_P3')

            sample_num = round(len(os.listdir(img2_folder)))

            for i in range(skip_start_end, sample_num - skip_start_end):
                dataset.append((img2_folder, pc_folder,
                                K2_folder, seq, i, 'P2', sample_num))
                dataset.append((img3_folder, pc_folder,
                                K3_folder, seq, i, 'P3', sample_num))
        return dataset


    def downsample_with_intensity_sn(self, pointcloud, intensity, sn, voxel_grid_downsample_size):
        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(np.transpose(pointcloud))
        intensity_max=np.max(intensity)

        fake_colors=np.zeros((pointcloud.shape[1],3))
        fake_colors[:,0:1]=np.transpose(intensity)/intensity_max

        pcd.colors=o3d.utility.Vector3dVector(fake_colors)
        pcd.normals=o3d.utility.Vector3dVector(np.transpose(sn))

        down_pcd=pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
        down_pcd_points=np.transpose(np.asarray(down_pcd.points))
        pointcloud=down_pcd_points

        intensity=np.transpose(np.asarray(down_pcd.colors)[:,0:1])*intensity_max
        sn=np.transpose(np.asarray(down_pcd.normals))

        return pointcloud, intensity, sn

    def downsample_np(self, pc_np, intensity_np, sn_np):
        if pc_np.shape[1] >= self.num_pc:
            choice_idx = np.random.choice(pc_np.shape[1], self.num_pc, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < self.num_pc:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], self.num_pc - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        intensity_np = intensity_np[:, choice_idx]
        sn_np=sn_np[:,choice_idx]
        return pc_np, intensity_np, sn_np

    def camera_matrix_cropping(self, K: np.ndarray, dx: float, dy: float):
        K_crop = np.copy(K)
        K_crop[0, 2] -= dx
        K_crop[1, 2] -= dy
        return K_crop

    def camera_matrix_scaling(self, K: np.ndarray, s: float):
        K_scale = s * K
        K_scale[2, 2] = 1
        return K_scale

    def augment_img(self, img_np):
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

        return img_color_aug_np

    def angles2rotation_matrix(self, angles):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R

    def generate_random_transform(self):
        """
        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-self.P_tx_amplitude, self.P_tx_amplitude),
             random.uniform(-self.P_ty_amplitude, self.P_ty_amplitude),
             random.uniform(-self.P_tz_amplitude, self.P_tz_amplitude)]
        angles = [random.uniform(-self.P_Rx_amplitude, self.P_Rx_amplitude),
                  random.uniform(-self.P_Ry_amplitude, self.P_Ry_amplitude),
                  random.uniform(-self.P_Rz_amplitude, self.P_Rz_amplitude)]

        rotation_mat = self.angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        # print('t',t)
        # print('angles',angles)

        return P_random
    
    def search_point_index(self, source_points, target_points):
        '''
        source_points: [M, 3]
        target_points: [N, 3]
        '''
        indices = []
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(source_points)
        source_kdtree = o3d.geometry.KDTreeFlann(pcd)
        for i in range(target_points.shape[0]):
            [_, index, _] = source_kdtree.search_knn_vector_3d(target_points[i], 1)
        # indices = torch.nonzero(torch.isin(source_points, target_points).all(dim=1))[:, 0]
            indices.append(index)
        # print(indices.shape)
        return np.array(indices)

    def __len__(self):
        return len(self.dataset)

    def getitem_nuscenes(self, index):
            
        filename = self.dataset[index]
        if self.mode == 'val' or self.mode == "test":
            folder = 'test'
        elif self.mode == 'train':
            folder = 'train'
        
        img = np.load(os.path.join(self.data_path, folder, 'img', filename)).astype(np.uint8)
        K = np.load(os.path.join(self.data_path, folder,'K', filename)) # intrinsic matrix
        pc_ = np.load(os.path.join(self.data_path, folder,'PC', filename))
        intensity = pc_[3, :].reshape(1, -1)
        pc = pc_[0:3, :]
        
        pc, intensity, _ = self.downsample_np(pc, intensity, np.zeros_like(pc))
        
            
        P = self.generate_random_transform()
        pc = np.dot(P[0:3, 0:3], pc) + P[0:3, 3:]
        
        
        # 2. get multi-level points and neighbor indexes for pyramid feature map
        feats = torch.from_numpy(np.concatenate([intensity, pc], axis=0).T.astype(np.float32))  
        
        if 'train' == self.mode:
            img_crop_dx = random.randint(0, img.shape[1] - self.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.img_H) / 2)
        img = img[img_crop_dy:img_crop_dy + self.img_H,
              img_crop_dx:img_crop_dx + self.img_W, :]
        K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)
        
        
        #get 1/8 scale image for correspondences
        scale_size = 0.125
        K_2 = self.camera_matrix_scaling(K,0.5)
 
        K_4=self.camera_matrix_scaling(K,scale_size)
        
        # Project
        # P_inv为从输入点云到相机坐标系的转换矩阵
        resize_w = self.img_W
        resize_h = self.img_H
        pc = pc.T
        proj_coord = np.dot(K, np.dot(np.linalg.inv(P[0:3, 0:3]), pc.T)-np.dot(np.linalg.inv(P[0:3, 0:3]), P[0:3, 3:])) # 点投影到image上
        proj_xy = proj_coord[0:2, :] / (proj_coord[2, :] + 1e-9)
        proj_xy = np.nan_to_num(proj_xy, nan=-1.0, posinf=-1.0, neginf=-1.0)
        proj_xy_int = np.floor(proj_xy).astype(np.int32)
        proj_depth = proj_coord[2,:]
        
        # point project insiade camera frustum
        in_frustum_mask = (proj_xy[0, :] >= 0) & (proj_xy[0, :] <= resize_w-1) & \
            (proj_xy[1, :] >= 0) & (proj_xy[1, :] <= resize_h-1) & \
            (proj_depth > 0)  # num_point
        
        num_sample_points = self.config['feature_match']['sample_point']
        
        # in_image_pc_idx 在frustum内且无occlusion的3D point idx
        in_image_pc_idx = np.where(in_frustum_mask==1)[0]
        if len(in_image_pc_idx) == 0:
            print(f"len in image pc idx is 0")
            return self.__getitem__(random.randint(0, len(self.dataset)-1))
        if len(in_image_pc_idx) >= num_sample_points:
            indices=np.random.choice(len(in_image_pc_idx), size=num_sample_points, replace=False)
        else:
            fix_idx = np.asarray(range(len(in_image_pc_idx)))
            while len(in_image_pc_idx) + fix_idx.shape[0] < num_sample_points:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(len(in_image_pc_idx)))), axis=0)
            random_idx = np.random.choice(len(in_image_pc_idx), num_sample_points - fix_idx.shape[0], replace=False)
            indices = np.concatenate((fix_idx, random_idx), axis=0)
        in_image_pc_idx = in_image_pc_idx[indices]
        in_image_point = pc[in_image_pc_idx, :]
        in_voxel_xy = proj_xy[:, in_image_pc_idx].T
        
        patch_size = self.config['feature_match']['coarse_patch_size']
        num_patch_row = resize_w / patch_size
                
        proj_patch = (proj_xy_int // patch_size).T
        proj_patch_idx = proj_patch[:, 1] * num_patch_row + proj_patch[:, 0]
        proj_patch_idx[~in_frustum_mask] = -1
        
        if not self.config['feature_match']['no_fusion']:
            # out_image_pc_idx 不在frustum内的3D point index
            out_image_pc_idx = np.where(in_frustum_mask==0)[0]
            if len(out_image_pc_idx) == 0:
                valid_out_image_pc_idx = np.zeros(num_sample_points)
                out_image_pc_idx = np.zeros(num_sample_points)
                # print(f"len out image pc idx is 0")
                # return self.__getitem__(random.randint(0, len(self.idx_list)-1))
            else:
                valid_out_image_pc_idx = np.ones(num_sample_points)
                if len(out_image_pc_idx) >= num_sample_points:
                    indices=np.random.choice(len(out_image_pc_idx), size=num_sample_points, replace=False)
                else:
                    fix_idx = np.asarray(range(len(out_image_pc_idx)))
                    while len(out_image_pc_idx) + fix_idx.shape[0] < num_sample_points:
                        fix_idx = np.concatenate((fix_idx, np.asarray(range(len(out_image_pc_idx)))), axis=0)
                    random_idx = np.random.choice(len(out_image_pc_idx), num_sample_points - fix_idx.shape[0], replace=False)
                    indices = np.concatenate((fix_idx, random_idx), axis=0)
                out_image_pc_idx = out_image_pc_idx[indices]
                out_image_point = pc[out_image_pc_idx, :]
            
            # out_image_xy 不在voxel内的2D keypoint index
            xy2 = proj_xy[:, in_frustum_mask] # 3D Point投影到图片内的pixel坐标
            dense_in_voxel_mask = coo_matrix((np.ones_like(xy2[0, :]), (xy2[1, :], xy2[0, :])), shape=(int(resize_h), int(resize_w))).toarray()
            dense_in_voxel_mask = np.array(dense_in_voxel_mask)
            dense_in_voxel_mask[dense_in_voxel_mask > 0] = 1. # 将被投影到的pixel位置为True
            # dilation
            dense_in_voxel_mask = binary_dilation(dense_in_voxel_mask, structure=np.ones((5, 5))).astype(np.float64)
            
            out_voxel_kp_idx = np.where(dense_in_voxel_mask.squeeze().reshape(-1)==0)[0]
            if len(out_voxel_kp_idx) == 0:
                valid_out_voxel_kp_idx = np.zeros(num_sample_points)
                out_voxel_xy = np.zeros((num_sample_points, 2))
                # print(f"len out voxel kp idx is 0")
                # return self.__getitem__(random.randint(0, len(self.idx_list)-1))
            else:
                valid_out_voxel_kp_idx = np.ones(num_sample_points)
                if len(out_voxel_kp_idx) >= num_sample_points:
                    indices=np.random.choice(len(out_voxel_kp_idx), size=num_sample_points, replace=False)
                else:
                    fix_idx = np.asarray(range(len(out_voxel_kp_idx)))
                    while len(out_voxel_kp_idx) + fix_idx.shape[0] < num_sample_points:
                        fix_idx = np.concatenate((fix_idx, np.asarray(range(len(out_voxel_kp_idx)))), axis=0)
                    random_idx = np.random.choice(len(out_voxel_kp_idx), num_sample_points - fix_idx.shape[0], replace=False)
                    indices = np.concatenate((fix_idx, random_idx), axis=0)
                out_voxel_kp_idx = out_voxel_kp_idx[indices]
                x = np.arange(resize_w)
                y = np.arange(resize_h)
                xv, yv = np.meshgrid(x, y)
                keypoints = np.vstack([xv.ravel(), yv.ravel()]).T
                keypoints = keypoints.astype(np.int16)
                out_voxel_xy = keypoints[out_voxel_kp_idx]
            
            out_image_pc_idx
            out_voxel_xy
        
        input_voxel_points = pc.copy()
        
        if self.config['feature_match']['point_backbone'] == 'kpconv':
            new_voxel_scalar = 1
            input_voxel_points = input_voxel_points / new_voxel_scalar
        else:
            new_voxel_scalar = self.pc_max_range
            input_voxel_points = input_voxel_points / new_voxel_scalar

        # color jitter
        if self.mode == 'train':
            img = self.augment_img(img)
        
        vis_input_image = img.copy()
        
        if self.config['feature_match']['rgb']:
            # rgb
            image = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        else:
            # grey
            image=self.image_transform(img)
            
        if self.config['feature_match']['dust_backbone']:
            image = self.image_transform_dust(image)
            
        voxel_id = "nuscenes"
        camera_params = np.array([K[0,0], K[0,2], K[1,2]])
        '''
        将scale_factor设为1
        则其余所有参数都按照真实input image来计算
        ''' 
        
        if self.config['feature_match']['point_backbone'] == 'kpconv':
            # get multi-level points and neighbor indexes for pyramid feature map
            num_stages = 5
            kpconv_dict = precompute_point_cloud_stack_mode(input_voxel_points.T, intensity, None, lengths=self.num_pc, num_stages=num_stages)
            feats = torch.from_numpy(np.concatenate([intensity, input_voxel_points.T], axis=0).T.astype(np.float32))  

            kpconv_dict['feats'] = feats

            coarse_points = np.array(kpconv_dict['points'][-1], dtype=np.float32).T  # [3, 2560]
            for i in range(num_stages):
                # data_dict['points'][i] = torch.from_numpy(np.asarray(data_dict['points'][i].points, dtype=np.float32))
                kpconv_dict['neighbors'][i] = kpconv_dict['neighbors'][i].long()
                if i < num_stages - 1:
                    kpconv_dict['subsampling'][i] = kpconv_dict['subsampling'][i].long()
                    kpconv_dict['upsampling'][i] = kpconv_dict['upsampling'][i].long()
            
            fps_idx = point_to_node(kpconv_dict['points'][0], kpconv_dict['points'][-1]).numpy() # 最后一层在20480个点中的idx
            
            proxy_in_image_mask = in_frustum_mask[fps_idx]
            coarse_pc_idx = np.where(proxy_in_image_mask==1)[0]
            if len(coarse_pc_idx) == 0:
                print(f"len coarse_pc_idx is 0")
                return self.__getitem__(random.randint(0, len(self.dataset)-1))
            if len(coarse_pc_idx) >= 32:
                indices=np.random.permutation(len(coarse_pc_idx))[0:32]
            else:
                fix_idx = np.asarray(range(len(coarse_pc_idx)))
                while len(coarse_pc_idx) + fix_idx.shape[0] < 32:
                    fix_idx = np.concatenate((fix_idx, np.asarray(range(len(coarse_pc_idx)))), axis=0)
                random_idx = np.random.choice(len(coarse_pc_idx), 32 - fix_idx.shape[0], replace=False)
                indices = np.concatenate((fix_idx, random_idx), axis=0)
            coarse_indices=coarse_pc_idx[indices] # 正样本在1280点中idx
            coarse_pc_idx = fps_idx[coarse_indices] # 正样本在20480点中idx
            # debug_mask = in_frustum_mask[coarse_pc_idx]
            
            coarse_pc_out_idx = np.where(proxy_in_image_mask==0)[0]
            if len(coarse_pc_out_idx) == 0:
                print(f"len coarse_pc_out_idx is 0")
                return self.__getitem__(random.randint(0, len(self.dataset)-1))
            if len(coarse_pc_out_idx) >= 32:
                indices=np.random.permutation(len(coarse_pc_out_idx))[0:32]
            else:
                fix_idx = np.asarray(range(len(coarse_pc_out_idx)))
                while len(coarse_pc_out_idx) + fix_idx.shape[0] < 32:
                    fix_idx = np.concatenate((fix_idx, np.asarray(range(len(coarse_pc_out_idx)))), axis=0)
                random_idx = np.random.choice(len(coarse_pc_out_idx), 32 - fix_idx.shape[0], replace=False)
                indices = np.concatenate((fix_idx, random_idx), axis=0)
            coarse_out_indices=coarse_pc_out_idx[indices] # 负样本在1280点中idx
            coarse_pc_out_idx = fps_idx[coarse_out_indices] # 正样本在20480点中idx
            # debug_mask = in_frustum_mask[coarse_pc_out_idx]
            
            c2f_indices = point_to_node(kpconv_dict['points'][1], kpconv_dict['points'][-1])
   
        elif self.config['feature_match']['point_backbone'] == 'pt':
            # for point backbone Point Transformer
            num_node = self.config['feature_match']['pt_num_node']
            # <------ sample the firs-level downsampled points, namely node ------>
            node_np, _ = self.farthest_sampler.sample(input_voxel_points[np.random.choice(input_voxel_points.shape[0], \
                                                        num_node*self.config['feature_match']['pt_rand'], replace=False), :].T,
                                                        k = num_node)
                                                    #   pc[:, np.random.choice(pc.shape[1],\
                                                    # self.num_node * 8, replace=False)], k=self.num_node)
            node_np = node_np.astype(np.float32)
            # <------ construct the node-to-point index ------>
            kdtree = cKDTree(node_np.T)
            _, point2node = kdtree.query(input_voxel_points, k=1)
            kdtree = cKDTree(input_voxel_points)
            _, node2point = kdtree.query(node_np.T, k=1)
            node_feats = feats[node2point]
            
            # get sparse 
            num_proxy = self.config['feature_match']['pt_num_proxy']
            fps_idx = node2point[:num_proxy] # proxy2point
            
            proxy_in_image_mask = in_frustum_mask[fps_idx]
            
            coarse_pc_idx = np.where(proxy_in_image_mask==1)[0]
            if len(coarse_pc_idx) == 0:
                print(f"len coarse_pc_idx is 0")
                return self.__getitem__(random.randint(0, len(self.dataset)-1))
            if len(coarse_pc_idx) >= 32:
                indices=np.random.permutation(len(coarse_pc_idx))[0:32]
            else:
                fix_idx = np.asarray(range(len(coarse_pc_idx)))
                while len(coarse_pc_idx) + fix_idx.shape[0] < 32:
                    fix_idx = np.concatenate((fix_idx, np.asarray(range(len(coarse_pc_idx)))), axis=0)
                random_idx = np.random.choice(len(coarse_pc_idx), 32 - fix_idx.shape[0], replace=False)
                indices = np.concatenate((fix_idx, random_idx), axis=0)
            coarse_indices=coarse_pc_idx[indices]
            coarse_pc_idx = fps_idx[coarse_indices]
            # debug_mask = in_image_mask[coarse_pc_idx]
            
            coarse_pc_out_idx = np.where(proxy_in_image_mask==0)[0]
            if len(coarse_pc_out_idx) == 0:
                print(f"len coarse_pc_out_idx is 0")
                return self.__getitem__(random.randint(0, len(self.dataset)-1))
            if len(coarse_pc_out_idx) >= 32:
                indices=np.random.permutation(len(coarse_pc_out_idx))[0:32]
            else:
                fix_idx = np.asarray(range(len(coarse_pc_out_idx)))
                while len(coarse_pc_out_idx) + fix_idx.shape[0] < 32:
                    fix_idx = np.concatenate((fix_idx, np.asarray(range(len(coarse_pc_out_idx)))), axis=0)
                random_idx = np.random.choice(len(coarse_pc_out_idx), 32 - fix_idx.shape[0], replace=False)
                indices = np.concatenate((fix_idx, random_idx), axis=0)
            coarse_out_indices=coarse_pc_out_idx[indices]
            coarse_pc_out_idx = fps_idx[coarse_out_indices]
            # debug_mask = in_image_mask[coarse_pc_out_idx]
        else:
            node_np = torch.tensor(0)
            point2node = torch.tensor(0)
            node2point = torch.tensor(0)
            node_feats = torch.tensor(0)
        
        if self.mode == 'train' and self.config['feature_match']['att_loss']:
            # node point 1280x3 in camera coord
            pc_in_cam = np.dot(np.linalg.inv(P[0:3, 0:3]), pc.T)-np.dot(np.linalg.inv(P[0:3, 0:3]), P[0:3, 3:])
            node_in_cam = pc_in_cam.T[fps_idx]
            x = np.arange(resize_w, step=patch_size) + patch_size//2
            y = np.arange(resize_h, step=patch_size) + patch_size//2
            xv, yv = np.meshgrid(x, y)
            keypoints = np.vstack([xv.ravel(), yv.ravel()]).T
            keypoints = keypoints.astype(np.int16)
            camera_ray = np.concatenate([keypoints, np.ones((keypoints.shape[0], 1))], axis=1)
            camera_ray = (np.linalg.inv(K) @ camera_ray.T).T
            
            node_in_cam # 1280个node在相机坐标系下的位置
            camera_ray # n_H*n_W条射线在相机坐标系下的方向 
            node_in_cam_norm = node_in_cam / np.linalg.norm(node_in_cam, axis=1, keepdims=True)
            camera_ray_norm = camera_ray / np.linalg.norm(camera_ray, axis=1, keepdims=True)
            
            patch_to_node_rad = camera_ray_norm @ node_in_cam_norm.T
            patch_to_node_rad = np.clip(patch_to_node_rad, -1.0, 1.0)
            patch_to_node_rad = np.arccos(patch_to_node_rad)
            patch_to_node_rad_mask = patch_to_node_rad < (np.pi / 18) # < 10 deg
            patch_to_node_rad_mask = patch_to_node_rad_mask * proxy_in_image_mask[np.newaxis, :]
            patch_to_node_rad_mask_neg = patch_to_node_rad > (np.pi / 9) # > 20 deg
            # valid_patch_mask = np.sum(patch_to_node_rad_mask, axis=-1) > 10
            # patch_to_node_rad_mask_neg = patch_to_node_rad_mask_neg * valid_patch_mask[:, np.newaxis]
            
            points_exp = node_in_cam[:, np.newaxis, :]  # Nx1x3
            rays_exp = camera_ray_norm[np.newaxis, :, :]  # 1xMx3
            cross_product = np.cross(points_exp, rays_exp)  # NxMx3
            node_to_patch_dist = np.linalg.norm(cross_product, axis=2)  # NxM
            node_to_patch_dist_mask = node_to_patch_dist < 3 # < 3 m
            node_to_patch_dist_mask = node_to_patch_dist_mask * proxy_in_image_mask[:, np.newaxis]
            node_to_patch_dist_mask_neg = node_to_patch_dist > 5 # > 5 m
            node_to_patch_dist_mask_neg = node_to_patch_dist_mask_neg * proxy_in_image_mask[:, np.newaxis]
            
        
        data_dict = {
            "patch_to_node_rad_mask":patch_to_node_rad_mask if self.mode == 'train' and self.config['feature_match']['att_loss'] else torch.tensor(0),
            "patch_to_node_rad_mask_neg":patch_to_node_rad_mask_neg if self.mode == 'train' and self.config['feature_match']['att_loss'] else torch.tensor(0),
            "node_to_patch_dist_mask":node_to_patch_dist_mask if self.mode == 'train' and self.config['feature_match']['att_loss'] else torch.tensor(0),
            "node_to_patch_dist_mask_neg":node_to_patch_dist_mask_neg if self.mode == 'train' and self.config['feature_match']['att_loss'] else torch.tensor(0),
            
            "fps_idx": fps_idx if self.config['feature_match']['point_backbone'] == 'kpconv' else torch.tensor(0),
            "c2f_indices": c2f_indices if self.config['feature_match']['point_backbone'] == 'kpconv' else torch.tensor(0),
            "kpconv_dict": kpconv_dict if self.config['feature_match']['point_backbone'] == 'kpconv' else torch.tensor(0),
            
            "coarse_indices": coarse_indices.astype(np.int64) if self.config['feature_match']['point_backbone'] in ['pt', 'kpconv'] else torch.tensor(0),
            "coarse_pc_idx": coarse_pc_idx.astype(np.int64) if self.config['feature_match']['point_backbone'] in ['pt', 'kpconv'] else torch.tensor(0),
            "coarse_out_indices": coarse_out_indices.astype(np.int64) if self.config['feature_match']['point_backbone'] in ['pt', 'kpconv'] else torch.tensor(0),
            "coarse_pc_out_idx": coarse_pc_out_idx.astype(np.int64) if self.config['feature_match']['point_backbone'] in ['pt', 'kpconv'] else torch.tensor(0),
            
            "in_image_pc_idx": in_image_pc_idx.astype(np.int64),
            "out_image_pc_idx": out_image_pc_idx.astype(np.int64) if not self.config['feature_match']['no_fusion'] else torch.tensor(0),
            "in_image_xy": in_voxel_xy.astype(np.float32),
            "out_image_xy": out_voxel_xy.astype(np.float32) if not self.config['feature_match']['no_fusion'] else torch.tensor(0),
            "proj_patch_idx": proj_patch_idx.astype(np.int64),
            "proj_patch": proj_patch.astype(np.int64),
            "proj_xy_int": proj_xy_int.astype(np.int32),
            "proj_xy": proj_xy,
            
            "valid_out_image_pc_idx" : valid_out_image_pc_idx,
            "valid_out_voxel_kp_idx" : valid_out_voxel_kp_idx,
            
            "in_image_point": in_image_point.astype(np.float32),
            # "aug_rot": aug_rot.astype(np.float32) if self.config['feature_match']['pcd_aug'] and self.mode=='train' else np.eye(3).astype(np.float32), # matrix used for pcd aug
            # "aug_trans": aug_trans.astype(np.float32) if self.config['feature_match']['pcd_aug'] and self.mode=='train' else np.zeros(3).astype(np.float32),
            "intrinsic": K.astype(np.float32), # intrinsic of the actual input image
            
            # "dense_point": dense_point.astype(np.float32), 
            # "norm_dense_point": norm_dense_point.astype(np.float32) if self.mode == 'train' else torch.tensor(0), 
            # "depth_mask": depth_mask, 
            "dense_in_voxel_mask": dense_in_voxel_mask,
            "in_image_mask": in_frustum_mask,
            
            'image_h':resize_h, 
            'image_w':resize_w,
            'image': image, 
            'vis_input_image': vis_input_image,
            'gt_pose' : P, # from camera to world
            'camera_params': camera_params,  # (3)
            'camera_type': "SIMPLE_PINHOLE",
            'dataset_name': 'kitti',
            'scale_factor': 1,
            'image_name': filename,
            'voxel_id':'nuscenes',
            'image_path': os.path.join(self.data_path, folder, 'img', filename),
            'voxel_points': input_voxel_points.astype(np.float32),  # (N x 3)
            'voxel_feats': feats.T,  # (4 x N)
            'voxel_scalar': new_voxel_scalar,
            'voxel_mean': np.zeros(3).astype(np.float32),
            
            # pt
            'voxel_nodes': node_np if self.config['feature_match']['point_backbone'] == 'pt' else torch.tensor(0),
            'point2node': point2node if self.config['feature_match']['point_backbone'] == 'pt' else torch.tensor(0),
            'node2point': node2point if self.config['feature_match']['point_backbone'] == 'pt' else torch.tensor(0),
            'node_feats': node_feats.T if self.config['feature_match']['point_backbone'] == 'pt' else torch.tensor(0),
        }
        
        return data_dict
    
    def __getitem__(self, index):
        if self.config['feature_match']['nuscenes']:
            return self.getitem_nuscenes(index)
        
        # obtain data from disk
        img_folder, pc_folder, K_folder, seq, seq_i, key, _ = self.dataset[index]
        img = np.load(os.path.join(img_folder, '%06d.npy' % seq_i))
        data = np.load(os.path.join(pc_folder, '%06d.npy' % seq_i))
        intensity = data[3:4, :]
        sn = data[4:, :]
        pc = data[0:3, :]
            

        P_Tr = np.dot(self.calibhelper.get_matrix(seq, key),
                      self.calibhelper.get_matrix(seq, 'Tr'))

        pc = np.dot(P_Tr[0:3, 0:3], pc) + P_Tr[0:3, 3:]  # transform pc to camera coordinate system
        sn = np.dot(P_Tr[0:3, 0:3], sn)
        K = np.load(os.path.join(K_folder, '%06d.npy' % seq_i))

            
        # 1. transform pc into 20480 points
        pc, intensity, sn = self.downsample_with_intensity_sn(pc, intensity, sn, voxel_grid_downsample_size=0.1)
        pc, intensity, sn = self.downsample_np(pc, intensity,sn)
        

        P = self.generate_random_transform()
        pc = np.dot(P[0:3, 0:3], pc) + P[0:3, 3:]
        sn = np.dot(P[0:3, 0:3], sn)
        
            
        # 2. get multi-level points and neighbor indexes for pyramid feature map
        feats = torch.from_numpy(np.concatenate([intensity, sn], axis=0).T.astype(np.float32))  
        
        # 3. scale image and camera intrinsic matrix
        img = cv2.resize(img,
                         (int(round(img.shape[1] * 0.5)),
                          int(round((img.shape[0] * 0.5)))),
                         interpolation=cv2.INTER_LINEAR)
        K = self.camera_matrix_scaling(K, 0.5)

        if 'train' == self.mode:
            img_crop_dx = random.randint(0, img.shape[1] - self.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.img_H) / 2)
        img = img[img_crop_dy:img_crop_dy + self.img_H,
              img_crop_dx:img_crop_dx + self.img_W, :]
        K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)


        #get 1/8 scale image for correspondences
        scale_size = 0.125
        K_2 = self.camera_matrix_scaling(K,0.5)
 
        K_4=self.camera_matrix_scaling(K,scale_size)

        # TODO: Project
        # P_inv为从输入点云到相机坐标系的转换矩阵
        resize_w = self.img_W
        resize_h = self.img_H
        pc = pc.T
        proj_coord = np.dot(K, np.dot(np.linalg.inv(P[0:3, 0:3]), pc.T)-np.dot(np.linalg.inv(P[0:3, 0:3]), P[0:3, 3:])) # 点投影到image上
        proj_xy = proj_coord[0:2, :] / (proj_coord[2, :] + 1e-9)
        proj_xy = np.nan_to_num(proj_xy, nan=-1.0, posinf=-1.0, neginf=-1.0)
        proj_xy_int = np.floor(proj_xy).astype(np.int32)
        proj_depth = proj_coord[2,:]
        
        # point project insiade camera frustum
        in_frustum_mask = (proj_xy[0, :] >= 0) & (proj_xy[0, :] <= resize_w-1) & \
            (proj_xy[1, :] >= 0) & (proj_xy[1, :] <= resize_h-1) & \
            (proj_depth > 0)  # num_point
        
        num_sample_points = self.config['feature_match']['sample_point']
        
        # in_image_pc_idx 在frustum内且无occlusion的3D point idx
        in_image_pc_idx = np.where(in_frustum_mask==1)[0]
        if len(in_image_pc_idx) == 0:
            print(f"len in image pc idx is 0")
            return self.__getitem__(random.randint(0, len(self.dataset)-1))
        if len(in_image_pc_idx) >= num_sample_points:
            indices=np.random.choice(len(in_image_pc_idx), size=num_sample_points, replace=False)
        else:
            fix_idx = np.asarray(range(len(in_image_pc_idx)))
            while len(in_image_pc_idx) + fix_idx.shape[0] < num_sample_points:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(len(in_image_pc_idx)))), axis=0)
            random_idx = np.random.choice(len(in_image_pc_idx), num_sample_points - fix_idx.shape[0], replace=False)
            indices = np.concatenate((fix_idx, random_idx), axis=0)
        in_image_pc_idx = in_image_pc_idx[indices]
        in_image_point = pc[in_image_pc_idx, :]
        in_voxel_xy = proj_xy[:, in_image_pc_idx].T
        
        
        patch_size = self.config['feature_match']['coarse_patch_size']
        num_patch_row = resize_w / patch_size
                
        proj_patch = (proj_xy_int // patch_size).T
        proj_patch_idx = proj_patch[:, 1] * num_patch_row + proj_patch[:, 0]
        proj_patch_idx[~in_frustum_mask] = -1
        
        if not self.config['feature_match']['no_fusion']:
            # out_image_pc_idx 不在frustum内的3D point index
            out_image_pc_idx = np.where(in_frustum_mask==0)[0]
            if len(out_image_pc_idx) == 0:
                valid_out_image_pc_idx = np.zeros(num_sample_points)
                out_image_pc_idx = np.zeros(num_sample_points)
                # print(f"len out image pc idx is 0")
                # return self.__getitem__(random.randint(0, len(self.idx_list)-1))
            else:
                valid_out_image_pc_idx = np.ones(num_sample_points)
                if len(out_image_pc_idx) >= num_sample_points:
                    indices=np.random.choice(len(out_image_pc_idx), size=num_sample_points, replace=False)
                else:
                    fix_idx = np.asarray(range(len(out_image_pc_idx)))
                    while len(out_image_pc_idx) + fix_idx.shape[0] < num_sample_points:
                        fix_idx = np.concatenate((fix_idx, np.asarray(range(len(out_image_pc_idx)))), axis=0)
                    random_idx = np.random.choice(len(out_image_pc_idx), num_sample_points - fix_idx.shape[0], replace=False)
                    indices = np.concatenate((fix_idx, random_idx), axis=0)
                out_image_pc_idx = out_image_pc_idx[indices]
                out_image_point = pc[out_image_pc_idx, :]
            
            # out_image_xy 不在voxel内的2D keypoint index
            xy2 = proj_xy[:, in_frustum_mask] # 3D Point投影到图片内的pixel坐标
            dense_in_voxel_mask = coo_matrix((np.ones_like(xy2[0, :]), (xy2[1, :], xy2[0, :])), shape=(int(resize_h), int(resize_w))).toarray()
            dense_in_voxel_mask = np.array(dense_in_voxel_mask)
            dense_in_voxel_mask[dense_in_voxel_mask > 0] = 1. # 将被投影到的pixel位置为True
            # dilation
            dense_in_voxel_mask = binary_dilation(dense_in_voxel_mask, structure=np.ones((5, 5))).astype(np.float64)
            
            out_voxel_kp_idx = np.where(dense_in_voxel_mask.squeeze().reshape(-1)==0)[0]
            if len(out_voxel_kp_idx) == 0:
                valid_out_voxel_kp_idx = np.zeros(num_sample_points)
                out_voxel_xy = np.zeros((num_sample_points, 2))
                # print(f"len out voxel kp idx is 0")
                # return self.__getitem__(random.randint(0, len(self.idx_list)-1))
            else:
                valid_out_voxel_kp_idx = np.ones(num_sample_points)
                if len(out_voxel_kp_idx) >= num_sample_points:
                    indices=np.random.choice(len(out_voxel_kp_idx), size=num_sample_points, replace=False)
                else:
                    fix_idx = np.asarray(range(len(out_voxel_kp_idx)))
                    while len(out_voxel_kp_idx) + fix_idx.shape[0] < num_sample_points:
                        fix_idx = np.concatenate((fix_idx, np.asarray(range(len(out_voxel_kp_idx)))), axis=0)
                    random_idx = np.random.choice(len(out_voxel_kp_idx), num_sample_points - fix_idx.shape[0], replace=False)
                    indices = np.concatenate((fix_idx, random_idx), axis=0)
                out_voxel_kp_idx = out_voxel_kp_idx[indices]
                x = np.arange(resize_w)
                y = np.arange(resize_h)
                xv, yv = np.meshgrid(x, y)
                keypoints = np.vstack([xv.ravel(), yv.ravel()]).T
                keypoints = keypoints.astype(np.int16)
                out_voxel_xy = keypoints[out_voxel_kp_idx]
            
            out_image_pc_idx
            out_voxel_xy
        
        input_voxel_points = pc.copy()
        
        if self.config['feature_match']['point_backbone'] == 'kpconv':
            new_voxel_scalar = 1
            input_voxel_points = input_voxel_points / new_voxel_scalar
        else:
            new_voxel_scalar = self.pc_max_range
            input_voxel_points = input_voxel_points / new_voxel_scalar
        
        # color jitter
        if self.mode == 'train':
            img = self.augment_img(img)
        
        vis_input_image = img.copy()
        
        if self.config['feature_match']['rgb']:
            # rgb
            image = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        else:
            # grey
            image=self.image_transform(img)
            
        if self.config['feature_match']['dust_backbone']:
            image = self.image_transform_dust(image)
            
        voxel_id = "kitti"
        camera_params = np.array([K[0,0], K[0,2], K[1,2]])
        '''
        将scale_factor设为1
        则其余所有参数都按照真实input image来计算
        ''' 
        
        if self.config['feature_match']['point_backbone'] == 'kpconv':
            # get multi-level points and neighbor indexes for pyramid feature map
            num_stages = 5
            kpconv_dict = precompute_point_cloud_stack_mode(input_voxel_points.T, intensity, sn, lengths=self.num_pc, num_stages=num_stages)
            feats = torch.from_numpy(np.concatenate([intensity, sn], axis=0).T.astype(np.float32))  

            kpconv_dict['feats'] = feats

            coarse_points = np.array(kpconv_dict['points'][-1], dtype=np.float32).T  # [3, 2560]
            for i in range(num_stages):
                # data_dict['points'][i] = torch.from_numpy(np.asarray(data_dict['points'][i].points, dtype=np.float32))
                kpconv_dict['neighbors'][i] = kpconv_dict['neighbors'][i].long()
                if i < num_stages - 1:
                    kpconv_dict['subsampling'][i] = kpconv_dict['subsampling'][i].long()
                    kpconv_dict['upsampling'][i] = kpconv_dict['upsampling'][i].long()
            
            fps_idx = point_to_node(kpconv_dict['points'][0], kpconv_dict['points'][-1]).numpy() # 最后一层在20480个点中的idx
            
            proxy_in_image_mask = in_frustum_mask[fps_idx]
            coarse_pc_idx = np.where(proxy_in_image_mask==1)[0]
            if len(coarse_pc_idx) == 0:
                print(f"len coarse_pc_idx is 0")
                return self.__getitem__(random.randint(0, len(self.dataset)-1))
            if len(coarse_pc_idx) >= 64:
                indices=np.random.permutation(len(coarse_pc_idx))[0:64]
            else:
                fix_idx = np.asarray(range(len(coarse_pc_idx)))
                while len(coarse_pc_idx) + fix_idx.shape[0] < 64:
                    fix_idx = np.concatenate((fix_idx, np.asarray(range(len(coarse_pc_idx)))), axis=0)
                random_idx = np.random.choice(len(coarse_pc_idx), 64 - fix_idx.shape[0], replace=False)
                indices = np.concatenate((fix_idx, random_idx), axis=0)
            coarse_indices=coarse_pc_idx[indices] # 正样本在1280点中idx
            coarse_pc_idx = fps_idx[coarse_indices] # 正样本在20480点中idx
            # debug_mask = in_frustum_mask[coarse_pc_idx]
            
            coarse_pc_out_idx = np.where(proxy_in_image_mask==0)[0]
            if len(coarse_pc_out_idx) == 0:
                print(f"len coarse_pc_out_idx is 0")
                return self.__getitem__(random.randint(0, len(self.dataset)-1))
            if len(coarse_pc_out_idx) >= 64:
                indices=np.random.permutation(len(coarse_pc_out_idx))[0:64]
            else:
                fix_idx = np.asarray(range(len(coarse_pc_out_idx)))
                while len(coarse_pc_out_idx) + fix_idx.shape[0] < 64:
                    fix_idx = np.concatenate((fix_idx, np.asarray(range(len(coarse_pc_out_idx)))), axis=0)
                random_idx = np.random.choice(len(coarse_pc_out_idx), 64 - fix_idx.shape[0], replace=False)
                indices = np.concatenate((fix_idx, random_idx), axis=0)
            coarse_out_indices=coarse_pc_out_idx[indices] # 负样本在1280点中idx
            coarse_pc_out_idx = fps_idx[coarse_out_indices] # 正样本在20480点中idx
            # debug_mask = in_frustum_mask[coarse_pc_out_idx]
            
            c2f_indices = point_to_node(kpconv_dict['points'][1], kpconv_dict['points'][-1])
   
        elif self.config['feature_match']['point_backbone'] == 'pt':
            # for point backbone Point Transformer
            num_node = self.config['feature_match']['pt_num_node']
            # <------ sample the firs-level downsampled points, namely node ------>
            node_np, _ = self.farthest_sampler.sample(input_voxel_points[np.random.choice(input_voxel_points.shape[0], \
                                                        num_node*self.config['feature_match']['pt_rand'], replace=False), :].T,
                                                        k = num_node)
                                                    #   pc[:, np.random.choice(pc.shape[1],\
                                                    # self.num_node * 8, replace=False)], k=self.num_node)
            node_np = node_np.astype(np.float32)
            # <------ construct the node-to-point index ------>
            kdtree = cKDTree(node_np.T)
            _, point2node = kdtree.query(input_voxel_points, k=1)
            kdtree = cKDTree(input_voxel_points)
            _, node2point = kdtree.query(node_np.T, k=1)
            node_feats = feats[node2point]
            
            # get sparse 
            num_proxy = self.config['feature_match']['pt_num_proxy']
            fps_idx = node2point[:num_proxy] # proxy2point
            
            proxy_in_image_mask = in_frustum_mask[fps_idx]
            
            coarse_pc_idx = np.where(proxy_in_image_mask==1)[0]
            if len(coarse_pc_idx) == 0:
                print(f"len coarse_pc_idx is 0")
                return self.__getitem__(random.randint(0, len(self.dataset)-1))
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
            # debug_mask = in_image_mask[coarse_pc_idx]
            
            coarse_pc_out_idx = np.where(proxy_in_image_mask==0)[0]
            if len(coarse_pc_out_idx) == 0:
                print(f"len coarse_pc_out_idx is 0")
                return self.__getitem__(random.randint(0, len(self.dataset)-1))
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
            # debug_mask = in_image_mask[coarse_pc_out_idx]
        else:
            node_np = torch.tensor(0)
            point2node = torch.tensor(0)
            node2point = torch.tensor(0)
            node_feats = torch.tensor(0)
        
        if self.mode == 'train' and self.config['feature_match']['att_loss']:
            # node point 1280x3 in camera coord
            pc_in_cam = np.dot(np.linalg.inv(P[0:3, 0:3]), pc.T)-np.dot(np.linalg.inv(P[0:3, 0:3]), P[0:3, 3:])
            node_in_cam = pc_in_cam.T[fps_idx]
            x = np.arange(resize_w, step=patch_size) + patch_size//2
            y = np.arange(resize_h, step=patch_size) + patch_size//2
            xv, yv = np.meshgrid(x, y)
            keypoints = np.vstack([xv.ravel(), yv.ravel()]).T
            keypoints = keypoints.astype(np.int16)
            camera_ray = np.concatenate([keypoints, np.ones((keypoints.shape[0], 1))], axis=1)
            camera_ray = (np.linalg.inv(K) @ camera_ray.T).T
            
            node_in_cam # 1280个node在相机坐标系下的位置
            camera_ray # n_H*n_W条射线在相机坐标系下的方向 
            node_in_cam_norm = node_in_cam / np.linalg.norm(node_in_cam, axis=1, keepdims=True)
            camera_ray_norm = camera_ray / np.linalg.norm(camera_ray, axis=1, keepdims=True)
            
            patch_to_node_rad = camera_ray_norm @ node_in_cam_norm.T
            patch_to_node_rad = np.clip(patch_to_node_rad, -1.0, 1.0)
            patch_to_node_rad = np.arccos(patch_to_node_rad)
            patch_to_node_rad_mask = patch_to_node_rad < (np.pi / 18) # < 10 deg
            patch_to_node_rad_mask = patch_to_node_rad_mask * proxy_in_image_mask[np.newaxis, :]
            patch_to_node_rad_mask_neg = patch_to_node_rad > (np.pi / 9) # > 20 deg
            # valid_patch_mask = np.sum(patch_to_node_rad_mask, axis=-1) > 10
            # patch_to_node_rad_mask_neg = patch_to_node_rad_mask_neg * valid_patch_mask[:, np.newaxis]
            
            points_exp = node_in_cam[:, np.newaxis, :]  # Nx1x3
            rays_exp = camera_ray_norm[np.newaxis, :, :]  # 1xMx3
            cross_product = np.cross(points_exp, rays_exp)  # NxMx3
            node_to_patch_dist = np.linalg.norm(cross_product, axis=2)  # NxM
            node_to_patch_dist_mask = node_to_patch_dist < 3 # < 3 m
            node_to_patch_dist_mask = node_to_patch_dist_mask * proxy_in_image_mask[:, np.newaxis]
            node_to_patch_dist_mask_neg = node_to_patch_dist > 5 # > 5 m
            node_to_patch_dist_mask_neg = node_to_patch_dist_mask_neg * proxy_in_image_mask[:, np.newaxis]
        
        data_dict = {
            "proj_depth": proj_depth, 
            "patch_to_node_rad_mask":patch_to_node_rad_mask if self.mode == 'train' and self.config['feature_match']['att_loss'] else torch.tensor(0),
            "patch_to_node_rad_mask_neg":patch_to_node_rad_mask_neg if self.mode == 'train' and self.config['feature_match']['att_loss'] else torch.tensor(0),
            "node_to_patch_dist_mask":node_to_patch_dist_mask if self.mode == 'train' and self.config['feature_match']['att_loss'] else torch.tensor(0),
            "node_to_patch_dist_mask_neg":node_to_patch_dist_mask_neg if self.mode == 'train' and self.config['feature_match']['att_loss'] else torch.tensor(0),
            
            "fps_idx": fps_idx if self.config['feature_match']['point_backbone'] == 'kpconv' else torch.tensor(0),
            "c2f_indices": c2f_indices if self.config['feature_match']['point_backbone'] == 'kpconv' else torch.tensor(0),
            "kpconv_dict": kpconv_dict if self.config['feature_match']['point_backbone'] == 'kpconv' else torch.tensor(0),
            
            "coarse_indices": coarse_indices.astype(np.int64) if self.config['feature_match']['point_backbone'] in ['pt', 'kpconv'] else torch.tensor(0),
            "coarse_pc_idx": coarse_pc_idx.astype(np.int64) if self.config['feature_match']['point_backbone'] in ['pt', 'kpconv'] else torch.tensor(0),
            "coarse_out_indices": coarse_out_indices.astype(np.int64) if self.config['feature_match']['point_backbone'] in ['pt', 'kpconv'] else torch.tensor(0),
            "coarse_pc_out_idx": coarse_pc_out_idx.astype(np.int64) if self.config['feature_match']['point_backbone'] in ['pt', 'kpconv'] else torch.tensor(0),
            
            "in_image_pc_idx": in_image_pc_idx.astype(np.int64),
            "out_image_pc_idx": out_image_pc_idx.astype(np.int64) if not self.config['feature_match']['no_fusion'] else torch.tensor(0),
            "in_image_xy": in_voxel_xy.astype(np.float32),
            "out_image_xy": out_voxel_xy.astype(np.float32) if not self.config['feature_match']['no_fusion'] else torch.tensor(0),
            "proj_patch_idx": proj_patch_idx.astype(np.int64),
            "proj_patch": proj_patch.astype(np.int64),
            "proj_xy_int": proj_xy_int.astype(np.int32),
            "proj_xy": proj_xy,
            
            "valid_out_image_pc_idx" : valid_out_image_pc_idx,
            "valid_out_voxel_kp_idx" : valid_out_voxel_kp_idx,
            
            "in_image_point": in_image_point.astype(np.float32),
            # "aug_rot": aug_rot.astype(np.float32) if self.config['feature_match']['pcd_aug'] and self.mode=='train' else np.eye(3).astype(np.float32), # matrix used for pcd aug
            # "aug_trans": aug_trans.astype(np.float32) if self.config['feature_match']['pcd_aug'] and self.mode=='train' else np.zeros(3).astype(np.float32),
            "intrinsic": K.astype(np.float32), # intrinsic of the actual input image
            
            # "dense_point": dense_point.astype(np.float32), 
            # "norm_dense_point": norm_dense_point.astype(np.float32) if self.mode == 'train' else torch.tensor(0), 
            # "depth_mask": depth_mask, 
            "dense_in_voxel_mask": dense_in_voxel_mask,
            "in_image_mask": in_frustum_mask,
            
            'image_h':resize_h, 
            'image_w':resize_w,
            'image': image, 
            'vis_input_image': vis_input_image,
            'gt_pose' : P, # from camera to world
            # 'gt_pose_world_to_cam_q':gt_pose_world_to_cam_q.astype(np.float32),  # quaternion of gt pose world to camera (tx, ty, tz, qw, qx, qy, qz)
            'camera_params': camera_params,  # (3)
            'camera_type': "SIMPLE_PINHOLE",
            'dataset_name': 'kitti',
            'scale_factor': 1,
            'image_name': '%06d.npy' % seq_i,
            'voxel_id':key,
            'image_path': os.path.join(img_folder, '%06d.npy' % seq_i),
            'voxel_points': input_voxel_points.astype(np.float32),  # (N x 3)
            'voxel_feats': feats.T,  # (4 x N)
            'voxel_scalar': new_voxel_scalar,
            'voxel_mean': np.zeros(3).astype(np.float32),
            
            # pt
            'voxel_nodes': node_np if self.config['feature_match']['point_backbone'] == 'pt' else torch.tensor(0),
            'point2node': point2node if self.config['feature_match']['point_backbone'] == 'pt' else torch.tensor(0),
            'node2point': node2point if self.config['feature_match']['point_backbone'] == 'pt' else torch.tensor(0),
            'node_feats': node_feats.T if self.config['feature_match']['point_backbone'] == 'pt' else torch.tensor(0),
        }
        
        return data_dict
               

