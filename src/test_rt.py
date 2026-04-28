import os
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from torch_geometric.data import Batch
try:
    from torch_geometric.loader import DataLoader  # PyG >= 2.0
except ImportError:  # pragma: no cover
    from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as UtilsDataLoader
from torch.utils.data import Sampler, BatchSampler
from torch import nn
import copy

from dataloader_pointcloud_helper import _get_points_from_rgb_and_depth_image
from modelnetdataset import ModelNetDataset
from augmentations import copy_pc
from pointcloud_utils import create_random_patches

from maths import *
from utils import *
from log import log
from visualize_graph import *


# Translation sanity gate (in normalised units; ground-truth translations
# are sampled from [-0.5, 0.5]).  Predictions outside this range are
# considered hallucinations and dropped from the metrics.
_TRANS_SANITY_MAX = float(os.environ.get('GRAPHITE_TRANS_SANITY_MAX', '1.0'))

# Optional minimum confidence for keypoints used in matching/pose estimation.
# 0.0 keeps every keypoint (paper-equivalent default).  Set e.g. 0.2 to use
# only "salient" keypoints (Sec. S3 of the paper).
_CONFIDENCE_THRESH = float(os.environ.get('GRAPHITE_CONFIDENCE_THRESH', '0.0'))

# ICP estimation method.  Defaults to point-to-plane (the paper's choice).
# Set ``GRAPHITE_ICP_METHOD=point`` to use point-to-point ICP, which is
# more stable when the RANSAC init is poor.
_ICP_METHOD = os.environ.get('GRAPHITE_ICP_METHOD', 'plane').lower()

# Eval RNG seed; set to a non-empty value for deterministic numbers.  When
# unset, eval keeps RANSAC's randomness (matches earlier behaviour).
_EVAL_SEED = os.environ.get('GRAPHITE_EVAL_SEED', '0')


def _seed_eval():
    if _EVAL_SEED == '':
        return
    seed = int(_EVAL_SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test(settings, net, test_loader):

    test_mse_ab, test_mae_ab, test_rotations_ab, test_translations_ab, \
    test_rotations_grf_pred, test_translations_grf_pred, \
    test_rotations_icp_pred, test_translations_icp_pred, test_eulers_ab = test_one_epoch(settings, net, test_loader)

    test_rmse_ab = np.sqrt(test_mse_ab)

    test_rotations_grf_pred_euler = npmat2euler(test_rotations_grf_pred)
    test_rotations_ab_euler = npmat2euler(test_rotations_ab)
    test_r_mse_ab = np.mean((test_rotations_grf_pred_euler - test_rotations_ab_euler) ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_rotations_grf_pred_euler - test_rotations_ab_euler))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_grf_pred) ** 2)
    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_grf_pred))

    metrics = {
        'graphite/rot_MSE': float(test_r_mse_ab),
        'graphite/rot_RMSE': float(test_r_rmse_ab),
        'graphite/rot_MAE': float(test_r_mae_ab),
        'graphite/trans_MSE': float(test_t_mse_ab),
        'graphite/trans_RMSE': float(test_t_rmse_ab),
        'graphite/trans_MAE': float(test_t_mae_ab),
    }

    log('Graphite rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (test_r_mse_ab, test_r_rmse_ab,
                     test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab),str(settings.to_string()))


    test_rotations_icp_pred_euler = npmat2euler(test_rotations_icp_pred)
    test_rotations_ab_euler = npmat2euler(test_rotations_ab)
    test_r_mse_ab = np.mean((test_rotations_icp_pred_euler - test_rotations_ab_euler) ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_rotations_icp_pred_euler - test_rotations_ab_euler))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_icp_pred) ** 2)
    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_icp_pred))

    metrics.update({
        'icp/rot_MSE': float(test_r_mse_ab),
        'icp/rot_RMSE': float(test_r_rmse_ab),
        'icp/rot_MAE': float(test_r_mae_ab),
        'icp/trans_MSE': float(test_t_mse_ab),
        'icp/trans_RMSE': float(test_t_rmse_ab),
        'icp/trans_MAE': float(test_t_mae_ab),
    })

    log('Grap+ICP rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (test_r_mse_ab, test_r_rmse_ab,
                     test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab),str(settings.to_string()))

    return metrics

def test_one_epoch(settings, net, test_data):

    net.eval()
    _seed_eval()

    mse_icp = 0
    mae_icp = 0
    mse_grf = 0
    mae_grf = 0

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0

    rotations_ab = []
    translations_ab = []
    rotations_grf_pred = []
    translations_grf_pred = []
    rotations_icp_pred = []
    translations_icp_pred = []

    eulers_ab = []
    failing_cases=0
    for data in tqdm(test_data):

        rotation_ab = torch.tensor(data['r']).unsqueeze(0)
        translation_ab = torch.tensor(data['t']).unsqueeze(0)
        euler_ab = torch.tensor(data['e']).unsqueeze(0)

        rotation_icp_pred, translation_icp_pred, rotation_grf_pred, translation_grf_pred = get_r_t_from_net(settings, net, data)
        if(translation_grf_pred is None):
            failing_cases += 1
            continue

        rotation_grf_pred = torch.tensor(rotation_grf_pred).unsqueeze(0)
        translation_grf_pred = torch.tensor(translation_grf_pred).squeeze(1).unsqueeze(0)
        rotation_icp_pred = torch.tensor(rotation_icp_pred).unsqueeze(0)
        translation_icp_pred = torch.tensor(translation_icp_pred).squeeze(1).unsqueeze(0)

        if torch.max(torch.abs(translation_grf_pred)) > _TRANS_SANITY_MAX:
            failing_cases += 1
            continue

        num_examples += 1

        rotations_ab.append(rotation_ab)
        translations_ab.append(translation_ab)
        rotations_grf_pred.append(rotation_grf_pred)
        translations_grf_pred.append(translation_grf_pred)
        rotations_icp_pred.append(rotation_icp_pred)
        translations_icp_pred.append(translation_icp_pred)
        eulers_ab.append(euler_ab)

        transformed_grf = transform_point_cloud(torch.tensor(data['raw'].points), rotation_grf_pred, translation_grf_pred)
        transformed_icp = transform_point_cloud(torch.tensor(data['raw'].points), rotation_icp_pred, translation_icp_pred)
        target = transform_point_cloud(torch.tensor(data['raw'].points), rotation_ab, translation_ab)

        # identity = torch.eye(3).unsqueeze(0).repeat(1, 1, 1)
        # loss = F.mse_loss(torch.matmul(rotation_grf_pred, rotation_ab), identity) \
        #        + F.mse_loss(translation_grf_pred, translation_ab)

        # total_loss += loss.item()

        mse_grf += torch.mean((transformed_grf - target) ** 2).item()
        mae_grf += torch.mean(torch.abs(transformed_grf - target)).item()
        mse_icp += torch.mean((transformed_icp - target) ** 2).item()
        mae_icp += torch.mean(torch.abs(transformed_icp - target)).item()

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_grf_pred = np.concatenate(rotations_grf_pred, axis=0)
    translations_grf_pred = np.concatenate(translations_grf_pred, axis=0)
    rotations_icp_pred = np.concatenate(rotations_icp_pred, axis=0)
    translations_icp_pred = np.concatenate(translations_icp_pred, axis=0)
    eulers_ab = np.concatenate(eulers_ab, axis=0)

    print("{} cases failed".format(failing_cases))
    return mse_grf * 1.0 / num_examples, mae_grf * 1.0 / num_examples, \
        rotations_ab, translations_ab, rotations_grf_pred, translations_grf_pred, \
        rotations_icp_pred, translations_icp_pred, eulers_ab

def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        from_dcm = getattr(Rotation, 'from_dcm', Rotation.from_matrix)
        r = from_dcm(mats[i])
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

def get_r_t_from_net(settings, model, data):


    visualize=False
    if(len(data['a']) > 150 or len(data['b']) > 150):
        print(len(data['a']), len(data['b']))
        return None, None, None, None
    
    with torch.no_grad():
        
        data_s_1 = [model(Batch.from_data_list(data['a'][i : i + settings.mb_size]).to(torch.cuda.current_device())) for i in range(0, len(data['a']), settings.mb_size)]
        data_s_2 = [model(Batch.from_data_list(data['b'][i : i + settings.mb_size]).to(torch.cuda.current_device())) for i in range(0, len(data['b']), settings.mb_size)]

    point_cloud1_np=np.asarray(data['raw'].points)
    cent1=np.mean(point_cloud1_np,0)

    g1_result = merge_dict(data_s_1)
    g2_result = merge_dict(data_s_2)


    g1_positions = [data['a'][i].positions[np.argmax(g1_result['probabilities'][i].cpu().detach().numpy())] for i in range(len(g1_result['probabilities']))]
    g2_positions = [data['b'][i].positions[np.argmax(g2_result['probabilities'][i].cpu().detach().numpy())] for i in range(len(g2_result['probabilities']))]

    g1_descs = g1_result['descriptors']
    g2_descs = g2_result['descriptors']

    if _CONFIDENCE_THRESH > 0.0 and 'confidence' in g1_result and 'confidence' in g2_result:
        c1 = g1_result['confidence'].squeeze(-1).cpu().detach().numpy()
        c2 = g2_result['confidence'].squeeze(-1).cpu().detach().numpy()
        keep1 = np.where(c1 > _CONFIDENCE_THRESH)[0]
        keep2 = np.where(c2 > _CONFIDENCE_THRESH)[0]
        # Refuse to over-prune: need a healthy minimum of keypoints for RANSAC.
        if len(keep1) >= 8 and len(keep2) >= 8:
            g1_positions = [g1_positions[i] for i in keep1]
            g2_positions = [g2_positions[i] for i in keep2]
            g1_descs = g1_descs[keep1]
            g2_descs = g2_descs[keep2]

    final1 = {'positions': g1_positions, 'descriptors': g1_descs}
    final2 = {'positions': g2_positions, 'descriptors': g2_descs}

    descriptorMatcher = DescriptorMatcher()
    setA, setB,ind = descriptorMatcher.match_old(final1, final2,return_id=True)
    keysA, keysB,descA,descB, ind = descriptorMatcher.match(final1, final2,return_id=True)
    # ind = descriptorMatcher.match_id(final1, final2)


    # pc1=torch.stack(final1['positions']).cpu().detach().numpy()
    # pc1_o3d=o3d.geometry.PointCloud()
    # pc1_o3d.points=o3d.utility.Vector3dVector(pc1)
    # pc1_o3d.paint_uniform_color([1,0,0])
    # pc2=torch.stack(final2['positions']).cpu().detach().numpy()
    # pc2_o3d=o3d.geometry.PointCloud()
    # pc2_o3d.points=o3d.utility.Vector3dVector(pc2)
    # pc2_o3d.paint_uniform_color([0,1,0])



    pc1=[data['a'][i].positions for i in range(len(data['a']))] 
    pc1=torch.stack(pc1).view([-1,3]).cpu().detach().numpy()
    pc1_o3d=o3d.geometry.PointCloud()
    pc1_o3d.points=o3d.utility.Vector3dVector(pc1)
    pc1_o3d.paint_uniform_color([1,0,0])

    pc2=[data['b'][i].positions for i in range(len(data['b']))] 
    pc2=torch.stack(pc2).view([-1,3]).cpu().detach().numpy()
    pc2_o3d=o3d.geometry.PointCloud()
    pc2_o3d.points=o3d.utility.Vector3dVector(pc2)
    pc2_o3d.paint_uniform_color([0,0,1])

    pc1_o3d.estimate_normals()
    pc2_o3d.estimate_normals()




    

    # f1_feat = o3d.registration.Feature()
    # f1_feat.data = descA.T

    # f2_feat = o3d.registration.Feature()
    # f2_feat.data = descB.T

    # distance_threshold=0.02
    # inliers=4#int(len(tuples)*0.1)
    # res=o3d.registration.registration_ransac_based_on_feature_matching(keypoints_o3d_a, keypoints_o3d_b, f1_feat,f2_feat,
    #  max_correspondence_distance=distance_threshold,estimation_method=o3d.registration.TransformationEstimationPointToPoint(),
    #  ransac_n=inliers,
    #         criteria=o3d.registration.RANSACConvergenceCriteria(max_validation=500,max_iteration=100000))


    estimatedRotation, estimatedTranslation, diff = get_pose_from_two_matching_sets(setA, setB)
    if estimatedRotation is None or estimatedTranslation is None:
        return None, None, None, None

    if visualize:
        keypoints_o3d_a=o3d.geometry.PointCloud()
        keypoints_o3d_a.points=o3d.utility.Vector3dVector(keysA)
        keypoints_o3d_a.paint_uniform_color([0,0,0])
        keypoints_o3d_b=o3d.geometry.PointCloud()
        keypoints_o3d_b.points=o3d.utility.Vector3dVector(keysB)
        keypoints_o3d_b.paint_uniform_color([0,0,0])
        list_matches=[[i,i] for i in range(keysA.shape[0])]
        tuples=[tuple(l) for l in list_matches]
        lines=o3d.geometry.LineSet.create_from_point_cloud_correspondences(keypoints_o3d_a,keypoints_o3d_b,tuples)
    
        pc1_copy=copy.deepcopy(pc1_o3d)

    # res=o3d.registration.registration_ransac_based_on_correspondence(pc1_o3d, pc2_o3d, corres, max_correspondence_distance=0.9,estimation_method=o3d.registration.TransformationEstimationPointToPlane(),ransac_n=20)
    # trans_gt=np.eye(4)
    # trans_gt[0:3,0:3]=data['r']
    # trans_gt[0:3,3]=data['t']

    trans_pr=np.eye(4)
    if estimatedTranslation is not None :
        trans_pr[0:3,0:3]=estimatedRotation
        trans_pr[0:3,3]=estimatedTranslation.squeeze()
    # trans_pr=res.transformation
    
 
    if _ICP_METHOD == 'plane':
        icp_estimator = o3d.registration.TransformationEstimationPointToPlane()
    else:
        icp_estimator = o3d.registration.TransformationEstimationPointToPoint()
    res = o3d.registration.registration_icp(
        pc1_o3d, pc2_o3d,
        max_correspondence_distance=0.1,
        init=trans_pr,
        estimation_method=icp_estimator)
    
    if visualize:
        show_with_no_light([pc1_o3d,pc2_o3d])
        pc1_copy=copy.deepcopy(pc1_o3d)
        pc1_copy.transform(trans_pr)
        pc1_copy.paint_uniform_color([1,0,0])
        show_with_no_light([pc2_o3d,pc1_copy])
        # 
        pc1_copy=copy.deepcopy(pc1_o3d)
        pc1_copy.transform(res.transformation)
        pc1_copy.paint_uniform_color([1,0,0])
        show_with_no_light([pc2_o3d,pc1_copy])

        pc1_o3d.paint_uniform_color([0.99,0.7,0.7])
        pc2_o3d.paint_uniform_color([0.7,0.7,0.99])
        keypoints_o3d_a.paint_uniform_color([0.0,0.5,0.0])
        keypoints_o3d_b.paint_uniform_color([0.0,0.5,0.0])
        show_with_no_light([lines,pc1_o3d,pc2_o3d,keypoints_o3d_a,keypoints_o3d_b])
        # o3d.visualization.draw_geometries([lines,pc1_o3d,pc2_o3d,keypoints_o3d_a,keypoints_o3d_b])
        
        
        
    
    estimatedRotationICP=res.transformation[0:3,0:3]
    estimatedTranslationICP=np.expand_dims(res.transformation[0:3,3],1)
    return estimatedRotationICP, estimatedTranslationICP,estimatedRotation, estimatedTranslation 


def transform_point_cloud(point_cloud, rotation, translation):

    r = rotation.squeeze(0)
    t = translation.squeeze(0)

    return torch.matmul(point_cloud, r) + t