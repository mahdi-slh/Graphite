import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors

from torch_geometric.data import DataLoader, Batch
from torch.utils.data import DataLoader as UtilsDataLoader
from torch.utils.data import Sampler, BatchSampler
from torch import nn
from visualize_graph import show_from_center
import copy

# from typing_extensions import final

from dataloader_pointcloud_helper import _get_points_from_rgb_and_depth_image
from modelnetdataset import ModelNetDataset
from augmentations import copy_pc
from pointcloud_utils import create_random_patches
from graph import visualize_torch_graphs

from maths import *
from utils import *
from log import log
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def test_visualize_desc(settings, net, test_data):

    net.eval()


    descriptors={x:[] for x in test_data.object_ids}
    for data in tqdm(test_data):

        
        scene = data['scene']

        pc_raw=o3d.io.read_point_cloud(test_data.root+'fragments/'+scene+data['name'])
        pc_raw_np=np.asarray(pc_raw.points)

        desc = get_desc_from_net(settings, net, data)
        descriptors[scene].append(desc)

        all_descriptors = desc['descriptors']
        all_keys =np.asarray( [k.detach().numpy() for k in desc['positions']])
        keypoints_all=o3d.geometry.PointCloud()
        keypoints_all.points=o3d.utility.Vector3dVector(all_keys)
        keypoints_all.paint_uniform_color([0,0.5,0])
        high_keys =np.asarray( [k.detach().numpy() for k in desc['positions_high']])
        high_keys = desc['positions_high']
        keypoints_high=o3d.geometry.PointCloud()
        keypoints_high.points=o3d.utility.Vector3dVector(high_keys)
        keypoints_high.paint_uniform_color([0,0.5,0])
        all_positions = [x.center.detach().numpy() for x in data['a']] 
        conf = [x.confidence.detach().numpy() for x in data['a']] 

        

        colors = all_descriptors.cpu().detach().numpy()
        colors_pca = PCA(n_components=3).fit_transform(colors)
        colors_pca = colors_pca - np.min(colors_pca, axis=0)
        colors_pca = colors_pca / np.max(colors_pca, axis=0)

    
        final_positions = np.array(all_positions)
        final_conf = np.squeeze(np.array(conf),1)
        # final_positions = np.array(final_positions)
        final_colors = np.array(colors_pca)


        nbrs_knn= NearestNeighbors(n_neighbors=4, algorithm='ball_tree')
        nbrs_knn.fit(final_positions)
        dist, labels = nbrs_knn.kneighbors(pc_raw_np, return_distance=True)
        dist=np.max(dist)/(dist+np.max(dist))

        weighted_color= final_colors[labels,:]*np.expand_dims(dist,2)
        weighted_color=np.sum(weighted_color,1)/np.sum(dist,1,keepdims=True)
        pc_raw.colors = o3d.utility.Vector3dVector(weighted_color)
        
        show_from_center([pc_raw])

        weighted_conf= final_conf[labels,:]*np.expand_dims(dist,2)
        weighted_conf=np.sum(weighted_conf,1)/np.sum(dist,1,keepdims=True)
        color_conf=np.squeeze(np.repeat(np.expand_dims(weighted_conf,1),3,1))
        color_conf[:,2]=0
        color_conf[:,0]=0
        pc_raw.colors = o3d.utility.Vector3dVector(color_conf)
        show_from_center([pc_raw])

        pc_raw.paint_uniform_color([0.7,0.7,.7])
        show_from_center([pc_raw])
        show_from_center([keypoints_all])
        show_from_center([keypoints_high])

        # o3d.visualization.draw_geometries([pc_raw,keypoints_all])
        # o3d.visualization.draw_geometries([keypoints_all,pc_raw])
        # o3d.visualization.draw_geometries([pc_raw,keypoints_high])
 

def test(settings, net, test_data):

    net.eval()


    descriptors={x:[] for x in test_data.object_ids}
    for data in tqdm(test_data):


        scene = data['scene']
        desc = get_desc_from_net(settings, net, data)
        descriptors[scene].append(desc)

    for scene_num in range( len(test_data.object_ids)):

        scene_length=test_data.scene_len[scene_num]
        scene_name=test_data.object_ids[scene_num]

        list_src_trg=[]
        filepath = MATCH3D_DIR+'/eval_files/'+scene_name+"/gt.log"
        with open(filepath) as fp:
            line = fp.readline()
            cnt = 1
            while line:
                if cnt%5==1:
                    list_src_trg.append((int(line.strip().split('\t')[0]),int(line.strip().split('\t')[1])))
                line = fp.readline()
                cnt += 1
        
        # f = open(MATCH3D_DIR+'/eval_files/'+scene_name+"/graphite.log", "w")

        pos=0
        sum_matches=0
        for pair in tqdm(list_src_trg):
            frame1=pair[0]
            frame2=pair[1]
            success,number_match=register_scenes(descriptors[scene_name][frame1], descriptors[scene_name][frame2])
            sum_matches+=number_match
            if success:
                pos+=1
        log("%s : recall %f average number of keypoint matches %d"%(scene_name,100*pos/len(list_src_trg),sum_matches//len(list_src_trg)),str(settings.to_string()))
        #     f.write('{}\t{}\t{}\n'.format(frame1,frame2,scene_length))
        #     f.write('{}\t{}\t{}\t{}\n'.format(tra[0][0],tra[0][1],tra[0][2],tra[0][3]))
        #     f.write('{}\t{}\t{}\t{}\n'.format(tra[1][0],tra[1][1],tra[1][2],tra[1][3]))
        #     f.write('{}\t{}\t{}\t{}\n'.format(tra[2][0],tra[2][1],tra[2][2],tra[2][3]))
        #     f.write('{}\t{}\t{}\t{}\n'.format(tra[3][0],tra[3][1],tra[3][2],tra[3][3]))

        # f.close()

def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def quat2mat(quat):

    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def get_desc_from_net(settings, model, data):

    
    with torch.no_grad():
        data_s_1 = [model(Batch.from_data_list(data['a'][i : i + settings.mb_size]).to(torch.cuda.current_device())) for i in range(0, len(data['a']), settings.mb_size)]


    g_result = merge_dict(data_s_1)

    confidence_thresh=.2
    g_centers_high= [data['a'][i].center for i in range(len(g_result['probabilities'])) if data['a'][i].confidence>confidence_thresh]
    # g_centers= [data['a'][i].center for i in range(len(g_result['probabilities'])) ]
    g_keypoints_high = [data['a'][i].positions[np.argmax(g_result['probabilities'][i].cpu().detach().numpy())] for i in range(len(g_result['probabilities'])) if data['a'][i].confidence>confidence_thresh]
    g_keypoints_all = [data['a'][i].positions[np.argmax(g_result['probabilities'][i].cpu().detach().numpy())] for i in range(len(g_result['probabilities'])) if data['a'][i].confidence>0]
    
    g_desc_high=torch.stack([g_result['descriptors'][i] for i in range(len(g_result['probabilities'])) if data['a'][i].confidence>confidence_thresh])
    g_desc=g_result['descriptors']
    
    final  ={'positions': g_keypoints_high,'positions_high': g_keypoints_high, 'descriptors': g_desc_high,'raw': data['raw']}

    # poses=[data['a'][i].positions for i in range(len(data['a']))]
    # pc1=torch.stack(poses).view([-1,3]).cpu().detach().numpy()
    # pc1_o3d=o3d.geometry.PointCloud()
    # pc1_o3d.points=o3d.utility.Vector3dVector(pc1)
    # pc1_o3d.paint_uniform_color([1,0,0])


    return final

def register_scenes(final1, final2):
    visualize=False
    descriptorMatcher = DescriptorMatcher()
    keysA, keysB,descA,descB, ind = descriptorMatcher.match(final1, final2,return_id=True)
    # setA, setB = descriptorMatcher.match_old(final1, final2)
    pc1_o3d=final1['raw']
    # center_ply1=pc1_o3d.get_center()
    pc2_o3d=final2['raw']
    # center_ply2=pc2_o3d.get_center()
    # setA=setA-center_ply1
    # setB=setB-center_ply1
    # pc1_o3d.translate(-center_ply1)
    # pc2_o3d.translate(-center_ply1)

    keypoints_o3d_a=o3d.geometry.PointCloud()
    keypoints_o3d_a.points=o3d.utility.Vector3dVector(keysA)
    keypoints_o3d_a.paint_uniform_color([1,1,0])
    keypoints_o3d_b=o3d.geometry.PointCloud()
    keypoints_o3d_b.points=o3d.utility.Vector3dVector(keysB)
    keypoints_o3d_b.paint_uniform_color([0,1,1])
    list_matches=[[i,i] for i in range(keysB.shape[0])]
    tuples=[tuple(l) for l in list_matches]
    lines=o3d.geometry.LineSet.create_from_point_cloud_correspondences(keypoints_o3d_a,keypoints_o3d_b,tuples)
    
    # if visualize:
    #     pc1=torch.stack(pc1).view([-1,3]).cpu().detach().numpy()
    #     pc1_o3d=o3d.geometry.PointCloud()
    #     pc1_o3d.points=o3d.utility.Vector3dVector(pc1)
    #     pc1_o3d.paint_uniform_color([1,0,0])


        
        


    #     pc2=[data['b'][i].positions for i in range(len(data['b']))] 
    #     pc2=torch.stack(pc2).view([-1,3]).cpu().detach().numpy()
    #     pc2_o3d=o3d.geometry.PointCloud()
    #     pc2_o3d.points=o3d.utility.Vector3dVector(pc2)
    #     pc2_o3d.paint_uniform_color([0,1,0])

    # pc1_o3d.estimate_normals()
    # pc2_o3d.estimate_normals()

    

    
    
    corres=o3d.utility.Vector2iVector(tuples)
    inliers=4#int(len(tuples)*0.1)
    #  results=o3d.registration.registration_ransac_based_on_feature_matching(cloud_ref,cloud_tar,feat_ref,feat_tar,ransac_n=4 ,estimation_method=o3d.registration.TransformationEstimationPointToPoint(),max_correspondence_distance =10,criteria=o3d.registration.RANSACConvergenceCriteria(max_validation=1000))
    
    # d1 = final1['descriptors'].detach().cpu().numpy()
    # d2 = final2['descriptors'].detach().cpu().numpy()
    # p1 = final1['positions']
    # p2 = final2['positions']

    f1_feat = o3d.registration.Feature()
    f1_feat.data = descA.T

    f2_feat = o3d.registration.Feature()
    f2_feat.data = descB.T

    distance_threshold=0.05
    inliers=4#int(len(tuples)*0.1)
    res=o3d.registration.registration_ransac_based_on_feature_matching(keypoints_o3d_a, keypoints_o3d_b, f1_feat,f2_feat,
     max_correspondence_distance=distance_threshold,estimation_method=o3d.registration.TransformationEstimationPointToPoint(),
     ransac_n=inliers,checkers =[o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            criteria=o3d.registration.RANSACConvergenceCriteria(max_validation=500,max_iteration=4000000))#
    

    # #old style
    # estimatedRotation, estimatedTranslation, diff = get_pose_from_two_matching_sets(setA, setB)
    # trans_pr=np.eye(4)
    # if estimatedTranslation is not None :
    #     trans_pr[0:3,0:3]=estimatedRotation
    #     trans_pr[0:3,3]=estimatedTranslation.squeeze()
    # res.transformation=trans_pr

    # keysA_hom=np.ones((keysA.shape[0],4))
    # keysA_hom[:,:3]=keysA
    # keysB_hom=np.ones((keysB.shape[0],4))
    # keysB_hom[:,:3]=keysB
    # res=o3d.registration.registration_ransac_based_on_correspondence(pc1_o3d, pc2_o3d, corres, max_correspondence_distance=0.1,estimation_method=o3d.registration.TransformationEstimationPointToPoint(),ransac_n=inliers,criteria=o3d.registration.RANSACConvergenceCriteria(max_validation=500,max_iteration=4000000))
    reg_copy=copy.deepcopy(pc1_o3d)
    reg_copy2=copy.deepcopy(pc1_o3d)

    keypoints_o3d_a_trans=copy.deepcopy(keypoints_o3d_a)
    keypoints_o3d_a_trans=keypoints_o3d_a_trans.transform(res.transformation)
    keysA_trans=np.asarray(keypoints_o3d_a_trans.points)
    c1=0.1
    percent=[np.linalg.norm(keysA_trans[i,:]-keysB[i,:])<0.1 for i in range(keysA.shape[0])]
    

    # print('%04f of total %d keypoints'%(100*sum(percent)/len(percent),len(percent)))
    
    # trans_gt=np.eye(4)
    # trans_gt[0:3,0:3]=data['r']
    # trans_gt[0:3,3]=data['t']

    # trans_pr=np.eye(4)
    # if estimatedTranslation is not None :
    #     trans_pr[0:3,0:3]=estimatedRotation
    #     trans_pr[0:3,3]=estimatedTranslation.squeeze()

    # res=o3d.registration.registration_icp(pc1_o3d, pc2_o3d,  max_correspondence_distance=9.9, init=res.transformation,estimation_method=o3d.registration.TransformationEstimationPointToPlane())
    

    reg_copy2.transform(res.transformation)
    # reg_copy.transform(trans_pr)
    if visualize:
        
        pc2_o3d.paint_uniform_color([0,0,1])
        reg_copy2.paint_uniform_color([1,0,0])
        pc1_o3d.paint_uniform_color([1,0,0])
        # o3d.visualization.draw_geometries([pc1_o3d,pc2_o3d])
        show_from_center([pc2_o3d,reg_copy2])
        pc1_o3d.paint_uniform_color([0.99,0.7,0.7])
        pc2_o3d.paint_uniform_color([0.7,0.7,0.99])
        keypoints_o3d_a.paint_uniform_color([0.0,0.5,0.0])
        keypoints_o3d_b.paint_uniform_color([0.0,0.5,0.0])
        # o3d.visualization.draw_geometries([pc1_o3d,pc2_o3d,keypoints_o3d_a,keypoints_o3d_b,lines])
        show_from_center([pc1_o3d])
        show_from_center([keypoints_o3d_a])
        show_from_center([pc2_o3d])
        show_from_center([keypoints_o3d_b])
    
    # estimatedRotationICP=res.tra[0:3,0:3]
    # estimatedTranslationICP=np.expand_dims(res.tra[0:3,3],1)
    return sum(percent)/len(percent)>0.05,len(percent)


def transform_point_cloud(point_cloud, rotation, translation):

    r = rotation.squeeze(0)
    t = translation.squeeze(0)

    return torch.matmul(point_cloud, r) + t