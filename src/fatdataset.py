from utils import *
from torch.utils.data import Dataset
from dataloader_pointcloud_helper import get_graphs_for_matching
import os
from tqdm import tqdm
import random
import torch
from maths import merge_two_transformations
import open3d as o3d
from utils import init_camera
from icp import get_icp

class FATDataset(Dataset):

    def __init__(self, root):

        self.object_ids = [0, 1]#, 1, 2, 3, 4, 5, 6] #[1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        self.instances_per_object = 3
        self.pairs_for_each_instance = 2
        self.root = root
        self.processed_paths = self.processed_file_names()

        self.raws = {}
        for k in self.object_ids:
            self.raws[k] = {}

        self.process()

        self.data = []

        for path in self.processed_paths:
            
            res = torch.load(path)

            for k in res:

                self.raws[k['meta']['id']][k['meta']['f1']] = o3d.io.read_point_cloud(self.raw_files_name(str(k['meta']['id'])) + str(k['meta']['f1']) + '.pcd')
                self.raws[k['meta']['id']][k['meta']['f2']] = o3d.io.read_point_cloud(self.raw_files_name(str(k['meta']['id'])) + str(k['meta']['f2']) + '.pcd')

            self.data.extend(torch.load(path))


    def __getitem__(self, idx):

        res = self.data[idx]

        a_raw = self.raws[res['meta']['id']][res['meta']['f1']]
        b_raw = self.raws[res['meta']['id']][res['meta']['f2']]

        res['a_raw'] = a_raw
        res['b_raw'] = b_raw

        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def processed_file_names(self):
        return [ self.root + 'processed/' + 'FAT_id-{}.pt'.format(i) for i in (self.object_ids) ]

    def raw_files_name(self, object_id):
        return self.root + 'processed/raws/{}/'.format(object_id)

    def download(self):
        pass

    def process(self):

        for i in range(len(self.object_ids)):
            
            data = []
            object_id = self.object_ids[i]
            path = self.processed_paths[i]
            
            if(os.path.exists(path)):
                print("Object {} already exists. Skipping.".format(str(object_id)))
                continue
            
            print("Processing Object {}".format(str(object_id)))
                    
            object_name, object_pres, val_poses = load_object(object_id)
            
            pivots = np.random.choice(range(len(object_pres)), self.instances_per_object)

            for jj in tqdm(range(len(pivots))):

                j = pivots[jj]
                
                d1 = get_fat_obj_for_id_and_frame(j, object_id, object_name, object_pres, val_poses)
                
                if(d1 is None):
                    continue

                d1_positions = torch.cat([g.positions for g in d1['graphs']], dim=0)

                pairs = np.random.choice(range(max(0, j - 7), min(len(object_pres), j + 7)), self.pairs_for_each_instance)
                
                for k in pairs:
                    
                    d2 = get_fat_obj_for_id_and_frame(k, object_id, object_name, object_pres, val_poses)

                    if(d2 is None):
                        continue

                    d2_positions = torch.cat([g.positions for g in d2['graphs']], dim=0) 

                    gt_r, gt_t = merge_two_transformations(d1['transformation']['translation'], d1['transformation']['rotation'], d2['transformation']['translation_inverse'], d2['transformation']['rotation_inverse'])

                    fat_object =  {
                        'a': d1['graphs'],
                        'b': d2['graphs'],
                        'r': gt_r,
                        't': gt_t,

                        'meta':{'id': object_id, 'f1': j, 'f2': k}
                    }

                    os.makedirs(os.path.dirname(self.raw_files_name(str(object_id))), exist_ok=True)
                    o3d.io.write_point_cloud(self.raw_files_name(str(object_id)) + str(j) + '.pcd', d1['raw_points'])
                    o3d.io.write_point_cloud(self.raw_files_name(str(object_id)) + str(k) + '.pcd', d2['raw_points'])

                    data.append(fat_object)


            torch.save(data, path)

def get_fat_obj_for_id_and_frame(frame, object_id, object_name, object_pres, val_poses):

    cam_int = init_camera('fat')

    color_image, _, depth_segmented, _, _ = read_related_images_of_specific_class_and_index(object_name, object_pres, frame)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_segmented, depth_scale=1, depth_trunc=66000, convert_rgb_to_intensity=False)
    points = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(cam_int))

    g = get_graphs_for_matching(color_image, depth_segmented, 'fat')

    if(len(g) == 0):
        return None

    t, t_inv, r, r_inv = get_transformation_matrices(val_poses[frame, :].squeeze())
    
    return  {
        'graphs': g,
        'transformation':{'translation': t, 'rotation': r, 'translation_inverse': t_inv, 'rotation_inverse': r_inv},
        'raw_points': points
    }