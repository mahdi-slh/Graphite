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
from torch_scatter import scatter_mean
from dataloader_pointcloud_helper import _get_points_from_rgb_and_depth_image
from sklearn.neighbors import RadiusNeighborsTransformer, NearestNeighbors
import visdom
from torch_geometric.data import DataLoader, Batch
from torch.utils.data import Sampler, BatchSampler
from torch.utils.data.sampler import SubsetRandomSampler

report_k = 0

# graph_dataset = GraphDataset(root=GRAPH_DATASET)

model = None
optimizer = None

criterion2 = torch.nn.MSELoss()

def train(pretrain_model, settings, dataset):

    global model
    global report_k

    model = pretrain_model

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    report_k = 0

    tag = str(settings.file_tag)
    plotter = VisdomLinePlotter(env_name='TRAIN '+tag)

    train_mask, val_mask = create_test_validation_masks_for_fat(dataset, VALIDATION_SPLIT)
    # train_sampler = SubsetRandomSampler(train_mask)
    # val_sampler = SubsetRandomSampler(val_mask)
    # train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler)
    # val_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler)

    train_dataset = [dataset[x] for x in train_mask]
    val_dataset = [dataset[x] for x in val_mask]

    for epoch in range(settings.epoch_size):

        try:
            l_train_loss, d_train_loss = run_epoch(train_dataset, epoch + 1, model,plotter, train=True,settings=settings,optimizer=optimizer)
        except KeyboardInterrupt:
            torch.save(model.state_dict(), TRAINED_MODEL_PATH+'{}.model'.format(settings.file_tag))
        
        with torch.no_grad():
            l_valid_loss, d_valid_loss  = run_epoch(val_dataset, epoch + 1, model, plotter, train=False,settings=settings,optimizer=optimizer)
        
        plotter.plot('label loss ', 'train', 'label loss  file ' + tag, epoch, l_train_loss)
        plotter.plot('descriptor loss ', 'train', 'descriptor loss  file '+tag, epoch, d_train_loss)

        plotter.plot('label loss ', 'validation', 'label loss  file '+tag, epoch, l_valid_loss)
        plotter.plot('descriptor loss ', 'validation', 'descriptor loss  file '+tag, epoch, d_valid_loss)

    print("training done.")

    return model

def run_epoch(dataset, epoch_number, model, plotter, train, settings,optimizer):

    global report_k

    samples_length = len(dataset)

    l_total_loss = 0
    d_total_loss = 0
    window_loss = 0

    for i in tqdm(range(samples_length)):

        data = dataset[i]
        data_q = dataset[random.randint(0, samples_length - 1)]
        
        optimizer.zero_grad()

        input_data = []

        l_loss = 0
        d_loss = 0

        input_data_1 = Batch.from_data_list(data['a']).to(torch.cuda.current_device())
        input_data_2 = Batch.from_data_list(data['b']).to(torch.cuda.current_device())
        input_data_3 = Batch.from_data_list(data_q['a']).to(torch.cuda.current_device())

        if(len(input_data_1.x) != NUMBER_OF_NODES_IN_GRAPH * len(data['a']) or len(input_data_2.x) != NUMBER_OF_NODES_IN_GRAPH * len(data['b'])):
            print("bad data.")
            continue

        try:
            output_1 = model(input_data_1)
            output_2 = model(input_data_2)
            output_3 = model(input_data_3)

        except Exception as e:
            print(e)
            continue

        for j in range(0, len(data['a'])):

            m=settings.m
            l_loss += criterion2(output['probabilities'][j], output['probabilities'][j+1])
            qa = output_1['descriptors'][j]
            qp = output_2['descriptors'][j]
            qn = output_3['descriptors'][j]

            l_loss += criterion2(output_1['probabilities'][j], output_2['probabilities'][j])
            d_loss += criterion2(qa, qp) / (criterion2(qa, qn) + criterion2(qa, qp))

        l_total_loss += l_loss.item()
        d_total_loss += d_loss.item()

        loss = settings.alpha * l_loss + d_loss
        window_loss += loss

        if(train):
            loss.backward()
            optimizer.step()

        if((i % settings.report_size == 0 and train and i > 0)):
            
            dev_num=str(torch.cuda.current_device())
            tag=str(settings.file_tag)
            plotter.plot('iteration loss', 'train', 'total loss  file '+tag, report_k, window_loss.item()/(settings.mb_size*settings.report_size))
            report_k += 1
            torch.save(model.state_dict(), TRAINED_MODEL_PATH+'{}.model'.format(settings.file_tag))
            window_loss = 0

    return l_total_loss / (settings.mb_size*samples_length), d_total_loss / (settings.mb_size*samples_length)