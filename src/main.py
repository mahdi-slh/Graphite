from model import Net
from utils import *
import torch
from os import path
from params import *
from cocodataset import CocoDataset
from fatdataset import FATDataset
from torch import nn
from modelnetdataset import *
from match3Ddataset import *
import evaluatefat
import evaluatemodelnet
from model import init_model

from train import train
from pretrain import pretrain

import test_rt
import test_match


if __name__ == "__main__":

    settings = training_configs()
    settings.input_params()

    pretrained_path = get_model_name(True, settings)
    trained_path = get_model_name(False, settings)

    # match3d_dataset_test = Match3DDataset(MATCH3D_DIR)
    # test_match.test(settings, model, match3d_dataset_test)
   
    # evaluate
    if path.exists(trained_path):
        
        print("Evaluation.")
        
        match3d_dataset_test = Match3DDataset(MATCH3D_DIR)
        # modelnet_dataset_test = ModelNetDataset(MODELNET_DIR, train=False,mode=settings.data_mode)
        for e in range(18,19):
            # model = init_model(get_model_name(False, settings,epoch=str(e)), settings)
            model = init_model(get_model_name(False, settings), settings)
            
            # test_rt.test(settings, model, modelnet_dataset_test)

        
            test_match.test(settings, model, match3d_dataset_test)
            # test_match.test_visualize_desc(settings, model, match3d_dataset_test)


        # evaluatemodelnet.evaluate_model(model, settings, modelnet_dataset_test, True)
        # fat_dataset_test = FATDataset (FAT_DIR)
        # evaluatefat.evaluate_model(model, settings, modelnet_dataset_test)
    
    # train
    elif path.exists(pretrained_path):
        
        print("Training 2.")

        match3d_dataset_test =Match3DDataset(MATCH3D_DIR)
        modelnet_dataset_train = ModelNetDataset(MODELNET_DIR, train=True,mode=settings.data_mode)
        modelnet_dataset_test = ModelNetDataset(MODELNET_DIR, train=False,mode=settings.data_mode)
        model = pretrain(settings, modelnet_dataset_train,modelnet_dataset_test,dataset_test_match=match3d_dataset_test, pretrain=False, model_path=pretrained_path)
    
    # pretrain
    else:

        print("Training 1.")
        
        coco_sub_dirs = [3]
       
        coco_dataset = CocoDataset(root=COCO_DIR, dirs=coco_sub_dirs) 
        modelnet_dataset_test = ModelNetDataset(MODELNET_DIR, train=False,mode=settings.data_mode)
        match3d_dataset_test =Match3DDataset(MATCH3D_DIR)
        model = pretrain(settings, coco_dataset,modelnet_dataset_test,dataset_test_match=match3d_dataset_test, pretrain=True)

        print("Training 2.")
        
        modelnet_dataset_train = ModelNetDataset(MODELNET_DIR, train=True,mode=settings.data_mode)
        
        model = pretrain(settings, modelnet_dataset_train,modelnet_dataset_test,dataset_test_match=match3d_dataset_test, pretrain=False, model_path=pretrained_path)