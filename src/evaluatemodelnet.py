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
from maths import get_accuracy_between_transformations

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

def visualize_modelnet(pc1, line_set, er, et, gr, gt):

    pc3 = o3d.geometry.PointCloud()
    copy_pc(pc3, pc1)

    pc2 = o3d.geometry.PointCloud()
    copy_pc(pc2, pc1)

    pc1.paint_uniform_color([1, 0.706, 0])
    pc2.paint_uniform_color([0, 1, 0])
    pc3.paint_uniform_color([0, 0, 1])

    pc2.rotate(gr, False)
    pc2.translate(gt, True)

    pc3.rotate(er, False)
    pc3.translate(et, True)

    patches1 = create_random_patches(pc1)
    patches2 = create_random_patches(pc2)

    o3d.visualization.draw_geometries([pc1, pc2])
    o3d.visualization.draw_geometries(patches1 + patches2)

    o3d.visualization.draw_geometries(patches1 + patches2 + [line_set])
    o3d.visualization.draw_geometries([pc1, pc2, pc3])

def evaluate_model(model_input, settings, dataset, visualize=True):
    
    global model
    model = model_input
    model.eval()

    descriptorMatcher = DescriptorMatcher()
    
    sample_data_idx = random.sample(range(len(dataset)), min(len(dataset), 1000))

    result = []

    for idx in sample_data_idx:

        data = dataset[idx]

        if(len(data['a']) > 150 or len(data['b']) > 150):
            print(len(data['a']), len(data['b']))
            continue
        
        with torch.no_grad():
            data_s_1 = [model(Batch.from_data_list(data['a'][i : i + settings.mb_size]).to(torch.cuda.current_device())) for i in range(0, len(data['a']), settings.mb_size)]
            data_s_2 = [model(Batch.from_data_list(data['b'][i : i + settings.mb_size]).to(torch.cuda.current_device())) for i in range(0, len(data['b']), settings.mb_size)]

        g1_result = merge_dict(data_s_1)
        g2_result = merge_dict(data_s_2)

        # torch.backends.cudnn.deterministic = False
        # with torch.no_grad():
        #     g1_result = model(Batch.from_data_list(data['a']).to(torch.cuda.current_device()))
        #     g2_result = model(Batch.from_data_list(data['b']).to(torch.cuda.current_device()))
            # else:
            #     g1_result = model(Batch.from_data_list(data['a']))
            #     g2_result = model(Batch.from_data_list(data['b']))

        g1_positions = [np.mean(data['a'][i].positions, axis=0) for i in range(len(g1_result['probabilities']))]
        g2_positions = [np.mean(data['b'][i].positions, axis=0) for i in range(len(g2_result['probabilities']))]

        # g1_positions = [torch.mean(data['a'][i].positions.cpu().detach().numpy(), axis=0) for i in range(len(g1_result['probabilities']))]
        # g2_positions = [torch.mean(data['b'][i].positions.cpu().detach().numpy(), axis=0) for i in range(len(g2_result['probabilities']))]

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
        
        if(visualize):
            visualize_modelnet(data['raw'], line_set, estimatedRotaion, estimatedTranslation, groundTruthRotaion, groundTruthTranslation)
        
        result.append(get_accuracy_between_transformations(groundTruthRotaion, groundTruthTranslation, estimatedRotaion, estimatedTranslation, data['raw']))

    result = np.array(result)
    result = np.mean(result)
    return result