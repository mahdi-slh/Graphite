from utils import *
from dataloader_pointcloud_helper import get_graph_pairs
from torch.utils.data import Dataset
import open3d as o3d
import torch
import os
from tqdm import tqdm
import random
from params import *
import gc
from bunny import create_random_patches
import numpy as np
from scipy.spatial.transform import Rotation
from maths import *
from graph import *
from pointcloud_utils import find_closest_matching_patches
from augmentations import copy_pc
import multiprocessing
from augmentations import random_transform,random_perturb
import random
import contextlib
import sys, os

from graph import visualize_torch_graphs

multiprocessing.set_start_method("spawn", True)

def load_point_cloud_from_off(filename):

    try:
        mesh = o3d.io.read_triangle_mesh(filename)
        pc = mesh.sample_points_uniformly(16000)
    except:
        print(filename)
        lines = open(filename).readlines()
        badline_corrected = lines[0].split('OFF')
        goodlines = lines[1:]
        goodlines.insert(0, badline_corrected[1])
        goodlines.insert(0, 'OFF\n')
        with open('temp.off', 'w') as f:
            for l in goodlines:
                f.write(l)

        mesh = o3d.io.read_triangle_mesh('temp.off')
        pc = mesh.sample_points_uniformly(16000)
        

    pc.estimate_normals()
    pc.normalize_normals()
    # pc.colors = o3d.utility.Vector3dVector(np.zeros([len(pc.points), 3]))

    return pc

def get_random_transformation():

    # from dcp

    anglex = np.random.uniform() * np.pi / 4
    angley = np.random.uniform() * np.pi / 4
    anglez = np.random.uniform() * np.pi / 4
    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                    [0, cosx, -sinx],
                    [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                    [0, 1, 0],
                    [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                    [sinz, cosz, 0],
                    [0, 0, 1]])
    R_ab = Rx.dot(Ry).dot(Rz)

    euler_ab = np.asarray([anglez, angley, anglex])

    translation_ab = np.array([np.random.uniform(-0.5, .5), np.random.uniform(-0.5, .5),
                                np.random.uniform(-0.5, 0.5)])

    return translation_ab, R_ab, euler_ab

class ModelNetDataset(Dataset):

    def __init__(self, root, train,mode):

        self.test_train = 'train/' if train else 'test/'
        if not (mode == 'patch' or  mode == 'noise' or mode== 'patchnoise'):
            print('wrong mode')
        self.mode=mode+'/'
        self.train = train

        if self.mode=='patch/':
            self.object_ids = [0, 1, 2,3,4,5,6,7,8,9,10] if self.train else [20,21,22,23,24,25,26,27,28,29,30]
        elif self.mode=='noise/':
            self.object_ids = [0, 1, 2,3,4,5,6,7,8,9,10,11,12]#,13,14,15,16,17,18,19,20]# if self.train else  [5,6,7,8,9,10]
            # self.object_ids =[12] #[4,6]
        else:
            self.object_ids = [0, 1, 2,3,4,5,6,7,8,9,10,11,12]
        # self.object_ids = [0, 1, 2] if self.train else [  20]
        
        with open(root + 'object_names.txt') as f:
            self.object_names = f.readlines()
            self.object_names = [x.replace('\n', '') for x in self.object_names if len(x) > 3]

        self.root = root
        self.processed_paths = self.processed_file_names()
        self.raw_paths = self.processed_file_raw_folders()

        self.process()

        self.data = []
        self.raw = {}

        for i in range(len(self.processed_paths)):
            path = self.processed_paths[i]
            temp_data = torch.load(path)
            if(self.train == False):
                for j in range(len(temp_data)):
                    temp_data[j]['raw'] = o3d.io.read_point_cloud(self.raw_paths[i] + temp_data[j]['name'],format='auto')

                self.data.extend(temp_data)
            if(self.train == True):
                pair_data = []
                for j in range(len(temp_data)):
                    for k in range(len(temp_data[j]['a'])):
                        pair_data.append(temp_data[j]['a'][k])
                        pair_data.append(temp_data[j]['b'][k])

                self.data.extend(pair_data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def processed_file_names(self):
        return [ self.root + 'processed/'+ self.mode + self.test_train + 'ModelNet_-{}-{}.pt'.format(str(NUMBER_OF_NODES_IN_GRAPH),i) for i in (self.object_ids) ]

    def processed_file_raw_folders(self):
        return [ self.root + 'processed/'+ self.mode + 'raw/'+ self.test_train + '{}-{}'.format(str(NUMBER_OF_NODES_IN_GRAPH),i) for i in (self.object_ids) ]

    def download(self):
        pass

    def process(self):

        for i in range(len(self.object_ids)):
            
            path = self.processed_paths[i]
            raw_path = self.raw_paths[i]
            object_id = self.object_ids[i]
            object_name = self.object_names[i]

            if(os.path.exists(path)):
                # print("Object {} already exists. Skipping.".format(str(object_id)))
                continue
            
            print("Processing Object {}".format(str(object_id)))

            base_path = self.root + object_name + '/' + self.test_train
            
            data = []
            raw = {}
            
            count = 0

            for off in tqdm(os.listdir(base_path)):
                
                count += 1
                if(not off.endswith('off')):
                    continue

                # if(self.train and count > 200):
                #     break

                # if(not self.train and count > 10):
                #     break

                point_cloud1 = load_point_cloud_from_off(base_path + off)
                if(point_cloud1 is None):
                    continue

                point_cloud1_np=np.asarray(point_cloud1.points)
                t_off= np.mean(point_cloud1_np,0)
                point_cloud1_np=point_cloud1_np-t_off
                scale=np.max(np.abs(point_cloud1_np))

                if scale==0 :
                    continue
                point_cloud1_np=point_cloud1_np/scale
                point_cloud1.points=o3d.utility.Vector3dVector(point_cloud1_np)

                # max_bound=np.median(point_cloud1.get_max_bound()-point_cloud1.get_min_bound())
                # vox_size=max_bound/60
                # point_cloud1_sub=point_cloud1.voxel_down_sample(vox_size)

                randperm=np.random.permutation(point_cloud1_np.shape[0])
                point_cloud1_sub=point_cloud1.select_down_sample(randperm[0:1024])

                point_cloud2_sub = copy.deepcopy(point_cloud1_sub)

                random_t, random_r, random_euler = get_random_transformation()

                point_cloud2_sub.rotate(random_r,False)
                point_cloud2_sub.translate(random_t, True)


                
                if(self.train):
                    point_cloud2_sub=random_perturb(point_cloud2_sub,False,sigma=0.015,clip=0.05)
                if self.mode=='noise/':
                    
                    if not self.train:
                        point_cloud2_sub=random_perturb(point_cloud2_sub,False,sigma=0.01,clip=0.05)
                    # point_cloud2_sub.paint_uniform_color([1,0,0])
                    # point_cloud1_sub.paint_uniform_color([0,1,0])
                    augment_train=False
                    augment_test=False
                    uniform_train=True
                    uniform_test=False
                elif self.mode=='patch/':
                    augment_train=True
                    augment_test=False
                    uniform_train=True
                    uniform_test=False
                elif self.mode=='patchnoise/':
                    if not self.train:
                        point_cloud2_sub=random_perturb(point_cloud2_sub,False,sigma=0.01,clip=0.05)
                    # point_cloud2_sub.paint_uniform_color([1,0,0])
                    # point_cloud1_sub.paint_uniform_color([0,1,0])
                    augment_train=True
                    augment_test=False
                    uniform_train=True
                    uniform_test=False

                patches1 = []
                patches2 = []
                
                if(self.train):
                # if(True):

                    patches1,c1 = create_random_patches(point_cloud1_sub, scale=40,count=30,augment=augment_train,uniform=uniform_train)
                    patches2,c2 = create_random_patches(point_cloud2_sub, scale=40,count=30,augment=augment_train,uniform=uniform_train)
                    # patches1_2,c1 = create_random_patches(point_cloud1_sub, scale=20,count=15,augment=augment_train,uniform=uniform_train)
                    # patches2_2,c2 = create_random_patches(point_cloud2_sub, scale=20,count=15,augment=augment_train,uniform=uniform_train)
                    # patches1_3,c1 = create_random_patches(point_cloud1_sub, scale=15,count=7,augment=augment_train,uniform=uniform_train)
                    # patches2_3,c2 = create_random_patches(point_cloud2_sub, scale=15,count=7,augment=augment_train,uniform=uniform_train)
                    patches1=patches1
                    patches2=patches2
                    # pairs = find_closest_matching_patches(patches_1, patches_2, t=random_t, r=random_r)

                    # for j in range(0, len(pairs)):
                    #     patches1.append(pairs[j][0])
                    #     patches2.append(pairs[j][1])
                    
                else:

                    patches1,c1 = create_random_patches(point_cloud1_sub,count=70,augment=augment_test,uniform=uniform_test)
                    patches2,c2 = create_random_patches(point_cloud2_sub,count=70,augment=augment_test,uniform=uniform_test)
                    # patches1_2,c1 = create_random_patches(point_cloud1_sub, scale=20,count=15,augment=augment_test,uniform=uniform_test)
                    # patches2_2,c2 = create_random_patches(point_cloud2_sub, scale=20,count=15,augment=augment_test,uniform=uniform_test)
                    # patches1_3,c1 = create_random_patches(point_cloud1_sub, scale=15,count=7,augment=augment_test,uniform=uniform_test)
                    # patches2_3,c2 = create_random_patches(point_cloud2_sub, scale=15,count=7,augment=augment_test,uniform=uniform_test)

                    
                
                graphed_patches1 = [create_graph(x, -1, True, self.train) for x in patches1 if len(x.points) == NUMBER_OF_NODES_IN_GRAPH]
                graphed_patches2 = [create_graph(x, -1, True, self.train) for x in patches2 if len(x.points) == NUMBER_OF_NODES_IN_GRAPH]
                # for i in range(len(graphed_patches1)):
                #     visualize_torch_graphs(graphed_patches1[i],graphed_patches2[i])
                for i_hash in range(len(graphed_patches1)):
                    hash_code = random.getrandbits(16)
                    graphed_patches1[i_hash]['hash'] = hash_code
                    graphed_patches2[i_hash]['hash'] = hash_code

                result = {'a': graphed_patches1,
                          'b': graphed_patches2,
                          't': random_t,
                          'r': random_r,
                          'e': random_euler,
                          'name': off + '.pcd'
                }




                # pc1=[result['a'][i].positions for i in range(len(result['a']))] 
                # pc1=torch.stack(pc1).view([-1,3]).cpu().detach().numpy()
                # pc1_o3d=o3d.geometry.PointCloud()
                # pc1_o3d.points=o3d.utility.Vector3dVector(pc1)
                # pc1_o3d.paint_uniform_color([1,0,0])

                # pc2=[result['b'][i].positions for i in range(len(result['b']))] 
                # pc2=torch.stack(pc2).view([-1,3]).cpu().detach().numpy()
                # pc2_o3d=o3d.geometry.PointCloud()
                # pc2_o3d.points=o3d.utility.Vector3dVector(pc2)
                # pc2_o3d.paint_uniform_color([0,1,0])





                raw[off + '.pcd'] = point_cloud1
                data.append(result)

            print(count)

            os.makedirs(os.path.dirname(raw_path), exist_ok=True)
            for k in raw.keys():
                o3d.io.write_point_cloud(raw_path + k , raw[k])

            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(data, path)