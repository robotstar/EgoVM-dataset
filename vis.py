import os
import sys
import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R
from pypcd import pypcd

class CreateBEVSegGT(object):
    """ CreateBEVSegGT """
    def __init__(self, class_names, class_types, map_resolution, map_range, color_bevmap):
        """ CreateBEVSegGT init """
        super(CreateBEVSegGT, self).__init__()
        self.class_names = class_names
        self.class_types = class_types
        self.class_num = len(self.class_names)
        self.map_resolution = map_resolution
        self.map_range = map_range
        self.color_bevmap = color_bevmap
        self.cell_size = ((int)((map_range[2] - map_range[0]) / map_resolution), \
                          (int)((map_range[3] - map_range[1]) / map_resolution))

    
    # norm_bev.dot([x,y,z] - position_bev) = 0, and calculate z
    def _getz_on_bevplane_world(self, x, y, norm_bev, position_bev):
        t1 = norm_bev[0][0] * position_bev[0][0] + norm_bev[1][0] * position_bev[1][0] + \
             norm_bev[2][0] * position_bev[2][0]
        t2 = norm_bev[0][0] * x + norm_bev[1][0] * y
        return (t1 - t2) / norm_bev[2][0]
    
    def _get_coord_onbevplane(self, x_world, y_world, norm_bev, position_bev, rot_w2l, t_w2l):
        # Get z coordinate of (x_world, y_world) on the bev plane
        z_world = self._getz_on_bevplane_world(x_world, y_world, norm_bev, position_bev)
        # Transform the point from world to lidar
        p_world = np.array([x_world, y_world, z_world])
        p_world = np.expand_dims(p_world, -1)
        p_lidar = np.matmul(rot_w2l, p_world) + t_w2l
        # get bev coordinate
        x_bev = int((p_lidar[0][0] - self.map_range[0]) / self.map_resolution)
        y_bev = int((p_lidar[1][0] - self.map_range[1]) / self.map_resolution)
        return x_bev, y_bev

    # For worldpoint 'p' on the bev plane, z_vec_world.dot(p - position_world) = 0
    def _get_bevplane_inworld(self, rot_l2w, t_l2w):
        z_vec = np.array([0.0, 0.0, 1.0])
        z_vec = np.expand_dims(z_vec, -1)
        z_vec_world = np.matmul(rot_l2w, z_vec)
        position_world = t_l2w
        return z_vec_world, position_world

    # Whether (x, y) in the range of bev space or not
    def _in_range(self, p):
        return p[0] >= 0 and p[0] < self.cell_size[0] and p[1] >= 0 and p[1] < self.cell_size[1]

    def _show_layout(self, scores, score_threshold=0.5):
        """
        Show all classification in the same image.
        Args:
            scores: [num_classes, H_bev, W_bev]
        Returns:
            bev_map_all: [H_bev, W_bev, 3]
        """
        C, H, W = scores.shape
        preds = torch.tensor(scores) > score_threshold
        bev_map_all = torch.zeros(H, W, 3)

        for c in range(C):
            bev_map = torch.tensor(self.color_bevmap[c]).reshape(1, 1, 3).repeat(H, W, 1)
            class_mask = preds[c].reshape(H, W, 1).repeat(1, 1, 3)
            bev_map_all = bev_map * class_mask + bev_map_all * (~class_mask)

        # bev_map_all /= 255

        return bev_map_all

    def __call__(self, results):
        """ CreateBEVSegGT call """
        
        # convert pose to [R, t]
        pose_trans = results['pose'][0:3]
        pose_quat = results['pose'][3:7]
        pose_rot = R.from_quat(pose_quat).as_matrix()
        pose_trans = np.expand_dims(pose_trans, -1)

        results['pose'] = np.concatenate([pose_rot, pose_trans], -1)

        # Get the pose of lidar_to_world(l2w) and world_to_lidar(w2l)
        t_l2w = results['pose'][:,3:4]
        rot_l2w = results['pose'][:,0:3]
        rot_w2l = np.linalg.inv(rot_l2w) # rot_l2w.T
        t_w2l = -np.matmul(rot_w2l, t_l2w)

        # Get the BEV plane equation in the world frame
        norm_bev, position_bev = self._get_bevplane_inworld(rot_l2w, t_l2w)

        # Convert the HDMap from world frame to BEV plane of LiDAR frame
        class_labels = np.zeros((self.class_num, *self.cell_size), dtype = np.uint8)
        for key in results:
            class_id = -1
            for index, name in enumerate(self.class_names):
                if name in key:
                    class_id = index
                    break
            if class_id == -1:
                continue
            if self.class_names[class_id] == 'crosswalk': 
                coords = []
                for obj in results[key]:
                    x_start, y_start = obj[0], obj[1]
                    bev_start = self._get_coord_onbevplane(x_start, y_start, norm_bev, position_bev, rot_w2l, t_w2l)
                    coords.append(bev_start)
                coords_int32 = np.array(coords).astype(np.int32).reshape(1,-1,2)
                class_labels[class_id,:,:] = cv2.fillPoly(class_labels[class_id,:,:], coords_int32, 1)
            elif self.class_types[class_id] == 'horizon':
                for obj in results[key]:
                    x_start, y_start = obj[0], obj[1]
                    bev_start = self._get_coord_onbevplane(x_start, y_start, norm_bev, position_bev, rot_w2l, t_w2l)
                    x_end, y_end = obj[2], obj[3]
                    bev_end = self._get_coord_onbevplane(x_end, y_end, norm_bev, position_bev, rot_w2l, t_w2l)
                    class_labels[class_id, :, :] = cv2.line(class_labels[class_id, :, :], bev_start, bev_end, 1, 2)
            else:
                for obj in results[key]:
                    x_start, y_start = obj[0], obj[1]
                    bev_start = self._get_coord_onbevplane(x_start, y_start, norm_bev, position_bev, rot_w2l, t_w2l)
                    class_labels[class_id, :, :] = cv2.circle(class_labels[class_id, :, :], bev_start, 1, 1, -1)

        seg_tensor = self._show_layout(class_labels.astype(np.float32))
        seg_arr = seg_tensor.numpy().astype(np.uint8)

        # cv2.imshow("seg_gt", seg_arr)
        # cv2.waitKey()

        os.system('mkdir -p vis/seg_gt')
        cv2.imwrite(os.path.join('vis/seg_gt', frame_id + '.jpg'), seg_arr)

        return 


class ProjectPointCloud2Image(object):
    """ ProjectPointCloud2Image """
    def __init__(self, camera_names):
        self.camera_names = camera_names
        self.camera_num = len(camera_names)

    def interpolate(self, val, y0, x0, y1, x1):
        return (val - x0) * (y1 - y0) / (x1 - x0) + y0

    def base(self, val):
        if val <= 0.125:
            return 0.0
        elif val <= 0.375:
            return self.interpolate(2.0 * val - 1.0, 0.0, -0.75, 1.0, -0.25)
        elif val <= 0.625:
            return 1.0
        elif val <= 0.87:
            return self.interpolate(2.0 * val - 1.0, 1.0, 0.25, 0.0, 0.75)
        else:
            return 0.0

    def red(self, gray):
        return self.base(gray - 0.25)

    def green(self, gray):
        return self.base(gray)

    def blue(self, gray):
        return self.base(gray + 0.25)

    def __call__(self, results):

        img_list = []
        for index in range(self.camera_num):
            img_i = cv2.imread(os.path.join(data_path, self.camera_names[index], frame_id + '.jpg'))
            img_list.append(img_i)
        results['img'] = np.stack(img_list, 0)

        undistort_imgs = []
        intrinsics = []
        for index, image in enumerate(results['img']):
            camera_intrinsic = results['camera_intrinsic'][index]
            K = [[camera_intrinsic[0], 0, camera_intrinsic[2]], [0, camera_intrinsic[1], camera_intrinsic[3]], [0, 0, 1]]
            D = [camera_intrinsic[4], camera_intrinsic[5], camera_intrinsic[6], camera_intrinsic[7], camera_intrinsic[8]]
            img_distort = cv2.undistort(image, np.array(K), np.array(D))
            undistort_imgs.append(img_distort)
            intrinsics.append(np.array(K))
        results['intrinsics'] = intrinsics

        images = np.stack(results['img'], 0)
        intrinsics = np.stack(results['intrinsics'], 0)
        camera_extrinsics = np.stack(results['camera_extrinsic'], 0)

        results['images'] = np.transpose(images, [0, 3, 1, 2])
        results['intrinsics'] = intrinsics.astype(np.float32)
        results['camera_extrinsics'] = camera_extrinsics
        ex_trans = camera_extrinsics[:, 0:3]
        ex_quat = camera_extrinsics[:, 3:7]
        ex_rots = R.from_quat(ex_quat).as_matrix()
        ex_trans = np.expand_dims(ex_trans, -1)

        extrinsics = np.concatenate([ex_rots, ex_trans], -1)
        results['extrinsics'] = extrinsics.astype(np.float32)

        results['img_shape'] = [img.shape for img in results['img']]
        results['img_shape'] = results['img_shape'][0]

        results['extrinsics'][:, 0:3, 0:3] = results['extrinsics'][:, 0:3, 0:3].transpose(0, 2, 1)
        results['extrinsics'][:, 0:3, 3:4] = np.matmul(-results['extrinsics'][:, 0:3, 0:3],
                                                       results['extrinsics'][:, 0:3, 3:4])

        point_cloud = results['point_cloud']
        point_cloud[:, 3] = point_cloud[:, 3] / 255.0

        xyz = point_cloud[:, 0:3]
        i = [[i] for i in point_cloud[:,3]]

        new_cloud = pypcd.make_xyz_label_point_cloud(point_cloud.astype(np.float32))
        os.system('mkdir -p vis/cloud/')
        new_cloud.save(os.path.join('vis/cloud', frame_id + '.pcd'))

        for cam_id in range(self.camera_num):
            for x in point_cloud:
                p = np.array([[x[0].item()], [x[1].item()], [x[2].item()], [1.0]])
                p_cam = np.matmul(results['extrinsics'][cam_id], p)
                p_cam = p_cam[0:3, :]
                depth = p_cam[2]
                if depth < 0.1:
                    continue
                depth = min(1.0, depth / 50.0)
                r, g, b = 255.0 * self.red(depth), 255.0 * self.green(depth), 255.0 * self.blue(depth)
                p_uvz = np.matmul(results['intrinsics'][cam_id], p_cam)
                u, v = int(p_uvz[0] / p_uvz[2]), int(p_uvz[1] / p_uvz[2])
                for rx in range(-1, 2):
                    for ry in range(-1, 2):
                        if v+ry >= 0 and v+ry < results['img_shape'][0] and u+rx >= 0 and u+rx < results['img_shape'][1]:
                            images[cam_id][v+ry][u+rx][0] = b
                            images[cam_id][v+ry][u+rx][1] = g
                            images[cam_id][v+ry][u+rx][2] = r

            images[cam_id] = images[cam_id].astype(np.uint8)

            img_path = os.path.join('vis/cloud2image', self.camera_names[cam_id])
            os.system('mkdir -p ' + img_path)
            cv2.imwrite(os.path.join(img_path, frame_id + '.jpg'), images[cam_id])

            # cv2.imshow('image_point', images[cam_id])
            # cv2.waitKey()
        
if __name__ == '__main__':
    frame_id = sys.argv[1]

    # load npz
    data_path = 'data'
    npz_file = os.path.join(data_path, 'npz', frame_id + '.npz')
    data = np.load(npz_file)

    # generate segmentation ground truth
    class_names =['lane', 'curb', 'crosswalk', 'stop_line', 'pole' , 'surfel']
    class_types = ['horizon', 'horizon', 'horizon', 'horizon', 'vertical', 'vertical']
    map_resolution = 0.25
    map_range = [-40, -40, 40, 40]
    color_bevmap = [[255, 255, 0],   # yellow
                    [0, 191, 255],   # sky blue
                    [65, 105, 225],  # light blue
                    [255, 0, 0],     # red
                    [0, 255, 0],     # green
                    [250, 0, 250],   # purple
                   ]

    # create semantic segmentation GT
    create_gt = CreateBEVSegGT(class_names, class_types, map_resolution, map_range, color_bevmap)
    create_gt(dict(data))

    # project point cloud to image
    camera_names = ['backward', 'left_backward', 'left_forward', 'right_backward', 'right_forward', 'short']
    cloud2image = ProjectPointCloud2Image(camera_names)
    cloud2image(dict(data))

