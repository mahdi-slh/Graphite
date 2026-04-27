from graphdataset import *
from cocodataset import *
# from fatdataset import *
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
from match3Ddataset import Match3DDataset

import visdom

model = None
matchdataset = Match3DDataset(MATCH3D_DIR,['val'])

def evaluate(model_input, settings):
    
    global model
    model = model_input
    model.eval()

    descriptorMatcher = DescriptorMatcher()
    
    result = []
    any_good = False

    for idx in range(0,len(matchdataset),2):
        print(idx)
        data_1 = matchdataset[idx].to(torch.cuda.current_device())
        data_2 = matchdataset[idx+1].to(torch.cuda.current_device())

        input_data_1 = Batch.from_data_list([data_1])
        input_data_2 = Batch.from_data_list([data_2])

        input_data_1.to(torch.cuda.current_device())
        input_data_2.to(torch.cuda.current_device())

        try:
            g1_result = model(input_data_1, pretrain=False)
            g2_result = model(input_data_2, pretrain=False)
        except:
            result.append(0)
            continue

        # input_data_s_1 = [Batch.from_data_list(data['d1']['graphs'][i : i+settings.mb_size]).to(torch.cuda.current_device()) for i in range(0, len(data['d1']['graphs']), settings.mb_size)]
        # input_data_s_2 = [Batch.from_data_list(data['d2']['graphs'][i : i+settings.mb_size]).to(torch.cuda.current_device()) for i in range(0, len(data['d2']['graphs']), settings.mb_size)]
        
        # outputs_1 = [model(input_data_1, pretrain=False) for input_data_1 in input_data_s_1]
        # outputs_2 = [model(input_data_2, pretrain=False) for input_data_2 in input_data_s_2]

        # g1_result = merge_dict(outputs_1)
        # g2_result = merge_dict(outputs_2)
        g1res=g1_result['descriptors'].detach().cpu().numpy()
        g2res=g2_result['descriptors'].detach().cpu().numpy()
        dist=np.linalg.norm(g1res-g2res)
        result.append(dist)

    result = np.array(result)
    np.savetxt('res.txt',np.asarray(result),fmt='%f')
    model.train()

    if(any_good):
        icp_res = np.mean(result[:, 0]) * 100
        gift_res = np.mean(result[:, 1]) * 100
    else:
        icp_res = 0
        gift_res = 0

    return icp_res, gift_res