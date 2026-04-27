from graphdataset import *
from cocodataset import *
# from fatdataset import *
from model import Net, init_model
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
import visdom
from log import log


import test_rt,test_match
from modelnetdataset import *

report_k = 1

criterion_cos = torch.nn.CosineSimilarity()
criterion1 = torch.nn.L1Loss(reduction='sum')
criterion2 = torch.nn.MSELoss(reduction='sum')
triplet_loss = None

def pretrain(settings, dataset_train,dataset_test,dataset_test_match=None, pretrain=True, model_path=None):

    global triplet_loss
    global report_k

    model = init_model(model_path, settings)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    triplet_loss = nn.TripletMarginLoss(margin=settings.m, p=2, reduction='sum')

    tag=str(settings.to_string())
    plotter = VisdomLinePlotter(env_name='PRETRAIN ' + tag + '-' + ('1' if pretrain else '2'))

    train_mask, val_mask = create_test_validation_masks(dataset_train, VALIDATION_SPLIT)
    val_loader = DataLoader(dataset_train, batch_sampler=BatchSampler(TripletSampler(len(dataset_train) - 1, val_mask), batch_size=settings.mb_size, drop_last=True))
    train_loader = DataLoader(dataset_train, batch_sampler=BatchSampler(TripletSampler(len(dataset_train) - 1, train_mask), batch_size=settings.mb_size, drop_last=True))

    for epoch in range(settings.epoch_size):
        log('-------Starting epoch %d--------'%(epoch+1),str(settings.to_string()))
        train_loader_iter = iter(train_loader)
        l_train_loss, d_train_loss, acc_train, avg_pos_t, avg_neg_t,c_train_loss  = run_epoch(train_loader_iter, epoch + 1, model, plotter, train=True, settings=settings, optimizer=optimizer, pretrain=pretrain)

        with torch.no_grad():
            validation_loader_iter = iter(val_loader)
            l_valid_loss, d_valid_loss, acc_loss, avg_pos_v, avg_neg_v,c_valid_loss  = run_epoch(validation_loader_iter, epoch + 1, model, plotter, train=False, settings=settings, optimizer=optimizer, pretrain=pretrain)
        
        plotter.plot('label loss ', 'train', 'label loss  file '+tag, epoch, l_train_loss)
        plotter.plot('descriptor loss ', 'train', 'descriptor loss  file '+tag, epoch, d_train_loss)
        plotter.plot('accuracy epoch ', 'train', 'accuracy epoch  file '+tag, epoch, acc_train)
        plotter.plot('avg similarity positive ', 'train', 'avg similarity positive  file '+tag, epoch, avg_pos_t)
        plotter.plot('avg similarity negative ', 'train', 'avg similarity negative file '+tag, epoch, avg_neg_t)
        plotter.plot('confidence loss ', 'train', 'confidence loss file '+tag, epoch, c_train_loss)

        plotter.plot('label loss ', 'validation', 'label loss  file '+tag, epoch, l_valid_loss)
        plotter.plot('descriptor loss ', 'validation', 'descriptor loss  file '+tag, epoch, d_valid_loss)
        plotter.plot('accuracy epoch ', 'validation', 'accuracy epoch  file '+tag, epoch, acc_loss)
        plotter.plot('avg similarity positive ', 'validation', 'avg similarity positive  file '+tag, epoch, avg_pos_v)
        plotter.plot('avg similarity negative ', 'validation', 'avg similarity negative file '+tag, epoch, avg_neg_v)
        plotter.plot('confidence loss ', 'validation', 'confidence loss file '+tag, epoch, c_valid_loss)
        
        if epoch%5==4 or pretrain is not True:
            test_rt.test(settings, model, dataset_test)
            test_match.test(settings,model,dataset_test_match)
            torch.save(model.state_dict(), get_model_name(pretrain, settings))
            torch.save(model.state_dict(), get_model_name(pretrain, settings,epoch=str(epoch)))

    log("pretraining done.",str(settings.to_string()))

    return model

def run_epoch(loader_iter, epoch_number, model, plotter, train, settings, optimizer, pretrain):

    global triplet_loss
    global report_k

    samples_length = (int)(len(loader_iter))

    l_total_loss = 0
    d_total_loss = 0
    c_total_loss = 0
    total_corrects = 0
    corrects = 0
    window_loss = 0
    window_desc = 0
    window_label = 0
    window_conf=0
    sum_pos = 0
    sum_neg = 0

    sum_pos_window = 0
    sum_neg_window = 0

    for i in tqdm(range(samples_length)):

        data = next(loader_iter).to_data_list()
        optimizer.zero_grad()

        l_loss_pre = 0
        l_loss_met = 0
        d_loss = 0
        c_loss = 0

        input_data = Batch.from_data_list(data).to(torch.cuda.current_device()) if settings.cuda else Batch.from_data_list(data)
        

        gt_labels = input_data.y.view([len(input_data.y)//NUMBER_OF_NODES_IN_GRAPH, NUMBER_OF_NODES_IN_GRAPH])
        gt_conf = input_data.confidence.view([len(input_data.y)//NUMBER_OF_NODES_IN_GRAPH, 1])
        
        output = model(input_data)
        # if pretrain:
        #     visualize_graph(input_data)


        
        for j in range(0, settings.mb_size, 3):

            qa = output['descriptors'][j]
            qp = output['descriptors'][j+1]
            qn = output['descriptors'][j+2]
            
            sum_neg += np.asscalar(torch.dist(qa, qn).cpu().detach().numpy())
            sum_neg_window += np.asscalar(torch.dist(qa, qn).cpu().detach().numpy())

            sum_pos += np.asscalar(torch.dist(qa, qp).cpu().detach().numpy())
            sum_pos_window += np.asscalar(torch.dist(qa, qp).cpu().detach().numpy())
        
        if pretrain:
            c = (torch.argmax(gt_labels, dim=1) == torch.argmax(output['probabilities'], dim=1)).sum().item()
        else:
            c = (torch.argmax(output['probabilities'][range(1, settings.mb_size, 3)], dim=1) == torch.argmax(output['probabilities'][range(0, settings.mb_size, 3)], dim=1)).sum().item()


        corrects += c
        total_corrects += c

                    
        a1 = triplet_loss(output['descriptors'][range(0, settings.mb_size, 3)], output['descriptors'][range(1, settings.mb_size, 3)], output['descriptors'][range(2, settings.mb_size, 3)])
        # pos=criterion2(output['descriptors'][range(0, settings.mb_size, 3)],output['descriptors'][range(1, settings.mb_size, 3)])
        # neg=criterion2(output['descriptors'][range(0, settings.mb_size, 3)],output['descriptors'][range(2, settings.mb_size, 3)])
        # a1=pos / ( neg + pos* settings.m )
        d_loss += a1
        
        a5=criterion2(gt_conf,output['confidence'])
        c_loss+=a5
        a2 = criterion1(gt_labels, output['probabilities'])
        a4 = criterion1(output['probabilities'][range(0, settings.mb_size, 3)], output['probabilities'][range(1, settings.mb_size, 3)])
        l_loss_pre += a2
        l_loss_met += a4

        a3 = (l_loss_pre if pretrain else l_loss_met) + settings.alpha * d_loss + c_loss
        loss = a3
       
        window_loss += a3
        window_conf += a5
        window_label += a2 if pretrain else a4
        window_desc += a1

        l_total_loss += a2.item() if pretrain else a4.item()
        d_total_loss += a1.item()
        c_total_loss += a5.item()

        if(train):
            loss.backward()
            optimizer.step()

        if((i % settings.report_size == 0 and train and i > 0)):
            
            tag = str(settings.file_tag)

            plotter.plot('iteration descriptor loss', 'train', 'descriptor loss'+tag, report_k, window_desc.item()/(settings.mb_size*settings.report_size//3))
            plotter.plot('iteration confidence', 'train', 'confidence'+tag, report_k, window_conf.item()/(settings.mb_size*settings.report_size))
            # plotter.plot('iteration positive term', 'train', 'positive term'+tag, report_k, sum_pos_window/(settings.mb_size*settings.report_size//3))
            # plotter.plot('iteration negative term', 'train', 'negative term'+tag, report_k, sum_neg_window/(settings.mb_size*settings.report_size//3))
            plotter.plot('iteration label loss', 'train', 'label loss'+tag, report_k, window_label.item()/(settings.mb_size*settings.report_size))
            plotter.plot('iteration accuracy', 'train', 'accuracy  file '+tag, report_k, (corrects*100)/(settings.mb_size*settings.report_size))
            report_k += 1

            torch.save(model.state_dict(), get_model_name(pretrain, settings))

            corrects = 0
            window_loss = 0
            window_conf = 0
            window_desc = 0
            window_label = 0
            sum_neg_window = 0
            sum_pos_window = 0


    length_loss = settings.mb_size*samples_length
    length_desc = settings.mb_size*samples_length//3
    return l_total_loss / length_loss, d_total_loss / length_desc, total_corrects*100/length_loss, sum_pos/length_desc, sum_neg/length_desc, c_total_loss/length_loss