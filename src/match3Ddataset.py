from utils import *
from torch.utils.data import Dataset
import torch
import os.path as osp
import os
from tqdm import tqdm
import random
from params import *
from bunny import create_random_patches
from graph import *

# import multiprocessing
# multiprocessing.set_start_method("spawn", True)
VOXEL_SIZE = 0.04

# Fixed RNG so the random keypoint set is reproducible across runs.
KEYPOINT_RNG = np.random.RandomState(42)
# Number of random keypoints to sample per fragment when no precomputed file
# is available.  3DMatch's reference benchmark uses 5000 random keypoints.
DEFAULT_NUM_KEYPOINTS = int(os.environ.get('GRAPHITE_MATCH3D_NUM_KEYPOINTS', '5000'))


def read_keypoint_list(file_path):
    list_k = np.loadtxt(file_path, dtype=int)
    return list_k


def _generate_random_keypoints(num_points, num_keypoints):
    """Sample ``num_keypoints`` indices uniformly from ``num_points``.

    This mirrors how the original 3DMatch geometric registration benchmark
    operates when no precomputed keypoint file is available - it simply picks
    random points from the fragment.
    """
    n = max(1, min(num_keypoints, num_points))
    return KEYPOINT_RNG.choice(num_points, size=n, replace=False).astype(int)


class Match3DDataset(Dataset):
    """Dataset wrapping the 3DMatch Geometric Registration Benchmark fragments.

    Expected directory layout (under ``root``):

        fragments/<scene>/cloud_bin_X.ply               (required)
        fragments/<scene>/01_Keypoints/cloud_bin_XKeypoints.txt   (optional)
        eval_files/<scene>/gt.log                       (required for testing)

    If the ``01_Keypoints`` directory is missing for a scene, random
    keypoints are sampled instead (configurable via the
    ``GRAPHITE_MATCH3D_NUM_KEYPOINTS`` env var, default 5000).

    ``object_ids`` defaults to all eight scenes from the official 3DMatch
    benchmark.  Override via the ``GRAPHITE_MATCH3D_SCENES`` env var (comma
    separated list of scene names).
    """

    def __init__(self, root, transform=None, pre_transform=None,
                 object_ids=None, scene_len=None):

        all_object_ids = ['home1', 'home2', 'hotel1', 'hotel2',
                          'hotel3', 'kitchen', 'mit', 'study']
        all_scene_len = [60, 60, 57, 55, 37, 60, 38, 66]

        env_scenes = os.environ.get('GRAPHITE_MATCH3D_SCENES')
        if object_ids is None and env_scenes:
            object_ids = [s.strip() for s in env_scenes.split(',') if s.strip()]

        if object_ids is None:
            self.object_ids = list(all_object_ids)
            self.scene_len = list(all_scene_len)
        else:
            self.object_ids = list(object_ids)
            if scene_len is None:
                lookup = dict(zip(all_object_ids, all_scene_len))
                self.scene_len = [lookup.get(name, None) for name in self.object_ids]
            else:
                self.scene_len = list(scene_len)

        
        self.root = root

        self.processed_paths = self.processed_file_names()
        self.raw_paths = self.processed_file_raw_folders()

        self.process()

        self.data = []
        self.raw = {}

        for i in range(len(self.processed_paths)):
            path = self.processed_paths[i]
            temp_data = torch.load(path)
            for j in range(len(temp_data)):
                temp_data[j]['raw'] = o3d.io.read_point_cloud(self.raw_paths[i] + temp_data[j]['name'],format='auto')
            self.data.extend(temp_data)
            # if(self.train == True):
            #     pair_data = []
            #     for j in range(len(temp_data)):
            #         for k in range(len(temp_data[j]['a'])):
            #             pair_data.append(temp_data[j]['a'][k])
            #             pair_data.append(temp_data[j]['b'][k])

            #     self.data.extend(pair_data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def processed_file_names(self):
        return [ self.root + 'processed/' + 'Match3d_n{}-{}_v{}.pt'.format(str(NUMBER_OF_NODES_IN_GRAPH),i,VOXEL_SIZE*100) for i in (self.object_ids) ]

    def processed_file_raw_folders(self):
        return [ self.root + 'processed/raw/' + 'n{}-{}_v{}/'.format(str(NUMBER_OF_NODES_IN_GRAPH),i,VOXEL_SIZE*100) for i in (self.object_ids) ]

    def download(self):
        pass

    def process(self):
        
        for i in range(len(self.object_ids)):
            
            if(osp.exists(self.processed_paths[i])):
                # print("Scene {} already exists. Skipping.".format(str(self.object_ids[i])))
                continue
            
            print("Processing Scene {}".format(self.object_ids[i]))

            data = []
            sub_dir_path = self.root + 'fragments/{}'.format(self.object_ids[i])

            bad = 0


            path = self.processed_paths[i]
            raw_path = self.raw_paths[i]
            object_id = self.object_ids[i]

            if(os.path.exists(path)):
                # print("Object {} already exists. Skipping.".format(str(object_id)))
                continue
            

            data = []
            raw = {}

            # Auto-detect scene length when not supplied (matches the number of
            # ``cloud_bin_X.ply`` files in the fragments folder).
            scene_len_i = self.scene_len[i]
            if scene_len_i is None:
                ply_files = [f for f in os.listdir(sub_dir_path)
                             if f.startswith('cloud_bin_') and f.endswith('.ply')]
                scene_len_i = len(ply_files)
                self.scene_len[i] = scene_len_i

            for j in tqdm(range(0, scene_len_i, 1)):

                ply_name = '/cloud_bin_%d.ply' % (j)
                keypoint_file_name = 'cloud_bin_%dKeypoints.txt' % (j)
                keypoint_path = sub_dir_path + '/01_Keypoints/' + keypoint_file_name
                ply_path = sub_dir_path + ply_name

                ply = o3d.io.read_point_cloud(ply_path)

                if os.path.exists(keypoint_path):
                    index_keys = read_keypoint_list(keypoint_path)
                else:
                    index_keys = _generate_random_keypoints(
                        len(ply.points), DEFAULT_NUM_KEYPOINTS)
    

                # max_bound=np.median(ply.get_max_bound()-ply.get_min_bound())
                # vox_size=0.01
                # ply=ply.voxel_down_sample(vox_size)
                ply.estimate_normals()
                ply.orient_normals_to_align_with_direction()

                
                
                g_fearture_index = 0

                # if augment:
                #     g_points=random_perturb(g_points,coef=0.1)
                #     g_points=random_transform(g_points,rotate=False)
                #     g_points=color_jitter(g_points,0.2)
                #     g_points=color_noise(g_points,0.05)

                # patches1,c1 = create_dense_patches(ply,scale=8,augment=False,uniform=False)
                # patches1_2,c1 = create_dense_patches(ply,scale=4,augment=False,uniform=False)
                # patches1_3,c1 = create_dense_patches(ply,scale=2,augment=False,uniform=False)
                patches,c=create_indexed_patches(ply,list(index_keys),voxel_size=VOXEL_SIZE)
                # patches1_3,c1 = create_dense_patches(ply,scale=10,augment=False,uniform=True)
                patches1=patches

                graphed_patches1 = [create_graph(patches[x], -1, True,False,center=c[x,:]) for x in range(len(patches1))]
                # for i in range(len(graphed_patches1)):
                #     visualize_torch_graphs(graphed_patches1[i],graphed_patches2[i])
                for i_hash in range(len(graphed_patches1)):
                    hash_code = random.getrandbits(16)
                    graphed_patches1[i_hash]['hash'] = hash_code

                result = {'a': graphed_patches1,
                            'scene': self.object_ids[i],
                            'frame': j,
                            'name': ply_name,
                            'center': ply.get_center()
                }


                raw[ply_name ] = ply
                data.append(result)


            os.makedirs(os.path.dirname(raw_path), exist_ok=True)
            for k in raw.keys():
                o3d.io.write_point_cloud(raw_path + k , raw[k])

            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(data, path)

            torch.save(data, self.processed_paths[i])