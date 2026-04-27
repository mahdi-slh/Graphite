import csv
import numpy as np
import open3d as o3d
import cv2
from pyquaternion import Quaternion
import random
from torch import argmax
import torch
from torch.utils.data import Sampler
from params import *

def load_object(id):

    with open(OBJECT_CLASS_LABELS, 'r') as csv_file:
        spam_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        object_names = np.asarray(list(spam_reader))
        object_name = object_names[id][0]

    filename = OBJECT_PRESENT_FRAMES_PATH.format(object_name)
    with open(filename, 'r') as csv_file:
        spam_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        object_pres = np.asarray(list(spam_reader))

    filename = OBJECT_GROUND_TRUTH_POSE_PATH.format(object_name)
    with open(filename, 'r') as csv_file:
        spam_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        poses = np.asarray(list(spam_reader)).astype(float)

    return object_name, object_pres, poses

def init_camera(dataset):

    if(dataset == 'coco'):
        camera_intrinsics = COCO_CAMERA_INTRINSICS

    if(dataset == 'coco2'):
        camera_intrinsics = COCO_CAMERA_INTRINSICS_2
    
    if(dataset == 'fat'):
        camera_intrinsics = FAT_CAMERA_INTRINSICS

    cam_int = o3d.camera.PinholeCameraIntrinsic()
    cam_int.set_intrinsics(camera_intrinsics['width'], camera_intrinsics['height'],
                           camera_intrinsics['fx'], camera_intrinsics['fy'],
                           camera_intrinsics['cx'], camera_intrinsics['cy'])
    return cam_int

def get_transformation_matrices(val_poses):

        Tref = val_poses[4:7] * 100
        tra_t = np.identity(4)
        tra_t[0:3, 3] = - Tref
        tra_t_inv = np.identity(4)
        tra_t_inv[0:3, 3] = Tref

        Qref = np.zeros((1, 4))
        Qref[0, 1:4] = val_poses[0:3]
        Qref[0, 0] = val_poses[3]
        Q = Quaternion(Qref[0, :]).inverse
        tra_r = Q.transformation_matrix
        tra_r_inv = Q.inverse.transformation_matrix

        return tra_t, tra_t_inv, tra_r, tra_r_inv

def read_related_images_of_specific_class_and_index(object_name, object_pres, idx):

    color_image = o3d.io.read_image(COLOR_SRC.format(object_name, object_pres[idx][0]))
    depth_image = o3d.io.read_image(DEPTH_SRC.format(object_name, object_pres[idx][0]))
    seg_image = o3d.io.read_image(SEG_SRC.format(object_name, object_pres[idx][0]))
    image = cv2.imread(IMG_SRC.format(object_name, object_pres[idx][0]))
    gray_image = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    seg_array = np.asarray(seg_image)
    depth_array = np.asarray(depth_image)
    depth_segmented = depth_array * (seg_array // 255)
    depth_segmented = o3d.geometry.Image(depth_segmented.astype(np.uint16))

    return color_image, depth_array, depth_segmented, seg_array, gray_image

def get_correct_indices_from_probabilty(probabilities):

    probabilities = probabilities.clone().detach()
    indices = []
    for i in range(probabilities.shape[0]):
        indices.append(np.argmax(probabilities[i, :]))

    return np.array(indices)


def create_test_validation_masks(dataset, split_ratio):
    """Split Dataset to train/val sets in random way with split_ratio."""
    even_index = [2 * x for x in list(range((len(dataset) - 2)//2))]
    train_mask = random.sample(even_index, k=(int)(len(even_index) * split_ratio))
    val_mask = [i for i in even_index if i not in train_mask]

    return train_mask, val_mask

def create_test_validation_masks_for_fat(dataset, split_ratio):
    """Split Dataset to train/val sets in random way with split_ratio."""
    index = list(range((len(dataset))))
    train_mask = random.sample(index, k=(int)(len(index) * split_ratio))
    val_mask = [i for i in index if i not in train_mask]

    return train_mask, val_mask

class DescriptorMatcher():

    def __init__(self):

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def match(self, res1, res2,return_id=False):


        d1 = res1['descriptors'].detach().cpu().numpy()
        d2 = res2['descriptors'].detach().cpu().numpy()
        p1 =  torch.stack(res1['positions']).detach().cpu().numpy()
        p2 = torch.stack(res2['positions']).detach().cpu().numpy()

        # matches = self.flann.knnMatch(d1, d2,1)
        matches = self.bf.match(d1, d2)
        # matches = sorted(matches, key=lambda x: x.distance)

        number_of_matches =len(matches)
        ind = np.zeros([number_of_matches, 2],dtype=np.int)
        for k in range(0, number_of_matches):

            i = matches[k].queryIdx
            j = matches[k].trainIdx

            ind[k, 0] = i
            ind[k, 1] = j
        
        key1_matched=p1[ind[:,0]]
        key2_matched=p2[ind[:,1]]

        desc1_matched=d1[ind[:,0]]
        desc2_matched=d2[ind[:,1]]
        if return_id:

            return key1_matched, key2_matched,desc1_matched,desc2_matched,ind
        else:
            return key1_matched, key2_matched,desc1_matched,desc2_matched

    def match_old(self, res1, res2,return_id=False):


        d1 = res1['descriptors'].detach().cpu().numpy()
        d2 = res2['descriptors'].detach().cpu().numpy()
        p1 = res1['positions']
        p2 = res2['positions']

        matches = self.flann.knnMatch(d1, d2,1)
        # matches = self.bf.Match(d1, d2,1)
        matches = sorted(matches, key=lambda x: x[0].distance)

        number_of_matches =np.minimum(40000,len(matches))
        setA = np.zeros([number_of_matches, 3])
        setB = np.zeros([number_of_matches, 3])
        ind = np.zeros([number_of_matches, 2],dtype=np.int)
        for k in range(0, number_of_matches):

            i = matches[k][0].queryIdx
            j = matches[k][0].trainIdx

            setA[k, :] = p1[i]
            setB[k, :] = p2[j]

            ind[k, 0] = i
            ind[k, 1] = j
        
        if return_id:

            return setA, setB,ind
        else:
            return setA, setB

    
    def match_id(self, res1, res2):

        d1 = res1['descriptors'].detach().cpu().numpy()
        d2 = res2['descriptors'].detach().cpu().numpy()
        p1 = res1['positions']
        p2 = res2['positions']

        matches = self.bf.match(d1, d2)
        matches = sorted(matches, key=lambda x: x.distance)

        number_of_matches = len(matches)
        ind = np.zeros([number_of_matches, 2],dtype=np.int)


        for k in range(0, number_of_matches):

            i = matches[k].queryIdx
            j = matches[k].trainIdx

            ind[k, 0] = i
            ind[k, 1] = j
        
        return ind



class TripletSampler(Sampler):

    def __init__(self, nb_samples, even_indexes):

        self.num_samples = nb_samples
        self.even_indexes = even_indexes

        self.indexes = self._get_indexes()

    def _get_indexes(self):

        evens = self.even_indexes
        random.shuffle(evens)
        indexes = []

        for even in evens:
            first = even
            second = first + 1
            third = random.randint(0, self.num_samples - 1)
            indexes.extend([first, second, third])

        return indexes

    def __iter__(self):
        self.indexes = self._get_indexes()
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)

def merge_dict(dicts):
   ''' Merge dictionaries and keep values of common keys in list'''
   dict_res = {}
   
   for key in dicts[0].keys():

       items = [d[key] for d in dicts]
       dict_res[key] = torch.cat(items, dim=0)
 
   return dict_res

def get_model_name(pretrain, settings,epoch=''): return (PRETRAINED_MODEL_PATH if pretrain else TRAINED_MODEL_PATH) + str(settings.to_string())+epoch + '.model'

from visdom import Visdom

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs | Iters/report',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')