from utils import *
from dataloader_pointcloud_helper import get_graph_pairs
from torch_geometric.data import Dataset
import torch
import os.path as osp
from tqdm import tqdm
import random
from params import *
import gc

import multiprocessing
multiprocessing.set_start_method("spawn", True)

class GraphDataset(Dataset):
    """torch_geometric.data.Dataset to create pointcloud dataset we need."""

    def __init__(self, root, transform=None, pre_transform=None):
        """the get_item will return [g1, g2, g3] which g1 and g2 are based on a same feature point and g3 is based on a random point."""
        self.object_ids = [0, 1, 2] #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.porccesed_file_paths = [ 'dataset_object_{}-{}_id-{}.pt'.format(str(WINDOW_SIZE), str(WINDOW_SIZE), i) for i in (self.object_ids) ]
        
        super(GraphDataset, self).__init__(root, transform, pre_transform)

        self.data = []
        for path in self.processed_paths:
            self.data.extend(torch.load(path))
        

    def num_edge_features():
        """Graycolor distance of edge nodes."""
        return 1

    def num_node_features():
        """RGB color of node."""
        return 3

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
        return self.porccesed_file_paths

    def download(self):
        pass

    def process(self):
        
        for i in range(len(self.object_ids)):
            
            data = []
            object_id = self.object_ids[i]
            path = self.processed_paths[i]
            
            if(osp.exists(path)):
                print("Object {} already exists. Skipping.".format(str(object_id)))
                continue
            
            print("Processing Object {}".format(str(object_id)))

            object_name, object_pres, val_poses = load_object(object_id)

            for frame_number in tqdm(range(min(1000, len(object_pres)))):
                try:
                    graphs = get_graph_pairs(object_id, object_name, object_pres, val_poses, frame_number, 'fat')
                    data.extend(graphs)
                except:
                    print("sth went wrong.")
                    continue

            torch.save(data, path)
            data = []
            gc.collect()