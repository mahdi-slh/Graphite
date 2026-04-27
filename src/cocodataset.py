from utils import *
from dataloader_pointcloud_helper import get_graph_for_coco
from torch_geometric.data import Dataset
import torch
import os.path as osp
import os
from tqdm import tqdm
import random
from params import *

from graph import visualize_torch_graphs

# import multiprocessing
# multiprocessing.set_start_method("spawn", True)

class CocoDataset(Dataset):
    """torch_geometric.data.Dataset to create pointcloud dataset we need."""

    def __init__(self, root,dirs, transform=None, pre_transform=None,augment=True):
        """the get_item will return [g1, g2, g3] which g1 and g2 are based on a same feature point and g3 is based on a random point."""
        
        self.root_path = root
        self.sub_dirs = dirs

        super(CocoDataset, self).__init__(root, transform, pre_transform,augment)

        self.size = 0
        self.data = []

        for path in self.processed_paths:
            self.data.extend(torch.load(path))
        
        self.size = len(self.data)
        self.augment=augment

    def num_edge_features():
        """XYZ distance of edge nodes."""
        return 3

    def num_node_features():
        """RGBXYZnorm of node."""
        return 6

    def get(self, idx):
        """Return [g1, g2, g3]."""
        return self.data[idx]

    def __len__(self):
        return len(self.data) - 1

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [ 'coco_dataset_object-{}_id-{}.pt'.format(str(NUMBER_OF_NODES_IN_GRAPH), i) for i in (self.sub_dirs) ]

    def download(self):
        pass

    def process(self):
        
        for i in range(len(self.sub_dirs)):
            
            if(osp.exists(self.processed_paths[i])):
                print("Object {} already exists. Skipping.".format(str(self.sub_dirs[i])))
                continue
            
            print("Processing Object {}".format(str(self.sub_dirs[i])))

            data = []
            sub_dir_path = self.root_path + 'renders_s{}/'.format(self.sub_dirs[i])

            bad = 0
            for j in tqdm(range(1, 10000, 2)):

                g1 = get_graph_for_coco(sub_dir_path + '%06d.jpg'%(j), sub_dir_path+'%06d_d.png'%(j))
                g2 = get_graph_for_coco(sub_dir_path + '%06d.jpg'%(j+1), sub_dir_path+'%06d_d.png'%(j+1))
                # visualize_torch_graphs(g1,g2)

                if(g1 is None or g2 is None):
                    bad += 2
                    continue
                
                hash_code = random.getrandbits(16)
                g1.hash = hash_code
                g2.hash = hash_code
                
                data.extend([g1, g2])

            torch.save(data, self.processed_paths[i])