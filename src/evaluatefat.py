from graphdataset import *
from cocodataset import *
from fatdataset import *
from model import Net
from utils import *
import torch
import numpy as np
from tqdm import tqdm
import random
import cv2
from scipy.spatial import distance
from os import path
from maths import *
from params import *

from torch_geometric.data import DataLoader, Batch
from torch.utils.data import DataLoader as UtilsDataLoader
from torch.utils.data import Sampler, BatchSampler
from torch import nn

from dataloader_pointcloud_helper import _get_points_from_rgb_and_depth_image
from modelnetdataset import ModelNetDataset
from augmentations import copy_pc
from pointcloud_utils import create_random_patches

import visdom

model = None

def visualize_modelnet(pc1, pc2, line_set, er, et):

    pc3 = o3d.geometry.PointCloud()
    copy_pc(pc3, pc1)

    pc1.paint_uniform_color([1, 0.706, 0])
    pc2.paint_uniform_color([0, 1, 0])
    pc3.paint_uniform_color([0, 0, 1])

    pc3.transform(er)
    pc3.transform(et)

    patches1 = create_random_patches(pc1)
    patches2 = create_random_patches(pc2)
    
    o3d.visualization.draw_geometries(patches1 + patches2 + [line_set])
    o3d.visualization.draw_geometries([pc1, pc2, pc3])

def evaluate_model(model_input, settings, fat_dataset):
    
    global model

    model = model_input
    model.eval()

    descriptorMatcher = DescriptorMatcher()
    
    sample_data_idx = random.sample(range(len(fat_dataset)), 5)

    result = []
    any_good = False

    for idx in sample_data_idx:

        data = fat_dataset[idx]

        if(len(data['a']) > 120 or len(data['b']) > 120):
            continue
        
        with torch.no_grad():
            data_s_1 = [model(Batch.from_data_list(data['a'][i : i + settings.mb_size]).to(torch.cuda.current_device())) for i in range(0, len(data['a']), settings.mb_size)]
            data_s_2 = [model(Batch.from_data_list(data['b'][i : i + settings.mb_size]).to(torch.cuda.current_device())) for i in range(0, len(data['b']), settings.mb_size)]

        g1_result = merge_dict(data_s_1)
        g2_result = merge_dict(data_s_2)

        g1_positions = [data['a'][i].positions[np.argmax(g1_result['probabilities'][i].cpu().detach().numpy())] for i in range(len(g1_result['probabilities']))]
        g2_positions = [data['b'][i].positions[np.argmax(g2_result['probabilities'][i].cpu().detach().numpy())] for i in range(len(g2_result['probabilities']))]

        final1  ={'positions': g1_positions, 'descriptors': g1_result['descriptors']}
        final2 = {'positions': g2_positions, 'descriptors': g2_result['descriptors']}

        setA, setB = descriptorMatcher.match(final1, final2)
        all_points = np.concatenate([setA, setB])
        lines = [[i, i + len(setA)] for i in range(len(setA))]
        line_set = o3d.geometry.LineSet(
            points = o3d.utility.Vector3dVector(all_points),
            lines = o3d.utility.Vector2iVector(lines),
        )
        colors = [[random.random(), random.random(), random.random()] for i in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        estimatedRotaion, estimatedTranslation, diff = get_pose_from_two_matching_sets(setA, setB)
        groundTruthRotaion, groundTruthTranslation = data['r'], data['t']

        if(estimatedRotaion is None):
            continue
        
        visualize_modelnet(data['a_raw'], data['b_raw'], line_set, estimatedRotaion, estimatedTranslation)


    return 0, 0