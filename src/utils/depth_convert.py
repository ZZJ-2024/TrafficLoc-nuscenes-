import numpy as np

class dpt_3d_convert():
    def __init__(self):
        pass

    def to_harmonic(self, input):
        M = input.shape[0]
        input = np.concatenate([input, np.ones([M,1])],axis=1)
        return input

    def proj_2to3(self, uv, depth, intrinsic, extrinsic, depth_unit = 1000):
        # input:
        # uv            M*2     the image coordinates of predicted pairs on sample image
        # depth         M       the depth of the matched voxels of sample image
        # intrinsic     3*3     the intrinsic matrix
        # extrinsic     4*4     the extrinsic matrix the the sample/depth image
        # output:
        # the corresponding depth of the matched pixels on the sample image
        # formula       xyz = extrinsic@(inv(intrinsic)@uvd)
        uv_harmonic = self.to_harmonic(uv)
        uv_harmonic = uv_harmonic * depth[:,None]/depth_unit
        camera_coor = (np.linalg.inv(intrinsic) @ uv_harmonic.T).T
        camera_coor = self.to_harmonic(camera_coor)
        world_coor  = (extrinsic @ camera_coor.T).T
        return world_coor[:,0:3]

    def proj_3to2(self, xyz, intrinsic, extrinsic):
        # input:
        # xyz           M*3     the xyz points
        # depth         M       the depth of the matched voxels of sample image
        # intrinsic     3*3     the intrinsic matrix
        # extrinsic     4*4     the extrinsic matrix the the sample/depth image
        # output:
        # the corresponding depth of the matched pixels on the sample image
        # formula       uvd=intrinsic(inv(extrinsic)@xyz)
        xyz = self.to_harmonic(xyz)
        xyz = np.linalg.inv(extrinsic) @ xyz.T
        uvd = intrinsic @ xyz[0:3]
        uvd = uvd.T
        uv, d = uvd[:,0:2]/(uvd[:,-1:]+1e-5), uvd[:,-1]
        return uv, d
    
    def proj_depth(self, depth, intrinsic, extrinsic = np.eye(4), depth_unit = 1000,
                   filter_edge = False, window_s = 3, max_range = 0.2, 
                   return_uv = False,
                   filter_far = False, far_thres = 80,
                   filter_near = False, near_thres = 0.01):
        if depth.ndim>2:
            depth = depth[:,:,0]
        h, w = depth.shape[0:2]
        u = np.arange(w)[None,:,None].repeat(h,axis=0)
        v = np.arange(h)[:,None,None].repeat(w,axis=1)
        uvd = np.concatenate([u, v, depth[:,:,None]],axis=-1)
        # condeuct mask
        if filter_edge:
            mask = np.zeros_like(depth)
            for i in range(window_s, h):
                for j in range(window_s, w):
                    check = depth[(i-window_s):(i+window_s), (j-window_s):(j+window_s)] / depth_unit
                    check = np.max(check) - np.min(check)
                    if check < max_range:
                        mask[i,j] = 1
            uvd = uvd[mask>0.5]
        uvd = uvd.reshape(-1,3)
        if filter_far:
            uvd = uvd[uvd[:,-1]<far_thres*depth_unit]
        if filter_near:
            uvd = uvd[uvd[:,-1]>near_thres*depth_unit]
        pc = self.proj_2to3(uvd[:,0:2], uvd[:,-1], intrinsic, extrinsic, depth_unit)
        if return_uv:
            return uvd[:,0:2], uvd[:,-1], pc
        else:
            return pc
    
    def proj_pc2dpt(self, ply, extrinsic, intrinsic, h, w):
        if type(ply) is not np.ndarray:
            ply = np.array(ply.points)
        uv, dpt = self.proj_3to2(ply, intrinsic, extrinsic)
        mask_w = (uv[:,0]<w) & (uv[:,0]>=0)
        mask_h = (uv[:,1]<h) & (uv[:,1]>=0)
        # mask mask off the back-project points
        mask_d = dpt>0.05
        mask = mask_h & mask_w & mask_d
        uv = uv[mask].astype(np.int32)
        dpt = dpt[mask]
        result = np.ones([h,w])*10000
        for i in range(uv.shape[0]):
            u,v = uv[i]
            d = dpt[i]
            result[v,u] = min(result[v,u],d)
        result[result>9999] = 0.0
        return result