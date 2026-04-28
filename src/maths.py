import os
import random
import numpy as np
import torch
import open3d as o3d
import math
from pyquaternion import Quaternion
from scipy.spatial import distance


# Inlier threshold (in normalized object units) used by the SVD-RANSAC
# rigid-pose solver below.  Point clouds are fit into a unit sphere before
# random transform sampling, with translations in [-0.5, 0.5], so 0.05 is
# the right order of magnitude.  Override via env var if needed.
RANSAC_INLIER_THRESH = float(os.environ.get('GRAPHITE_RANSAC_INLIER_THRESH', '0.05'))
RANSAC_NUM_ITER = int(os.environ.get('GRAPHITE_RANSAC_NUM_ITER', '1000'))
RANSAC_MIN_SAMPLE = int(os.environ.get('GRAPHITE_RANSAC_MIN_SAMPLE', '4'))
RANSAC_ROT_GATE = float(os.environ.get('GRAPHITE_RANSAC_ROT_GATE_RAD', '1.4'))

def seperate_rotation_and_translation(tr):

    t_h = np.identity(4)
    r_h = np.identity(4)

    t_h[0:3, 3] = tr[0:3, 3]
    r_h[0:3, 0:3] = tr[0:3, 0:3]

    return r_h, t_h

def merge_two_transformations(t1, r1, t2, r2):
    
    tr_final = t2.dot(r2).dot(r1).dot(t1)
    r_final, t_final = seperate_rotation_and_translation(tr_final)

    return r_final, t_final

def create_homogenous_matrix(t, r):
    
    tr = np.identity(4)
    tr[0:3, 3] = t.squeeze()
    tr[0:3, 0:3] = r

    return tr

def get_pose_from_two_matching_sets(a, b,
                                    inlier_thresh=None,
                                    num_iter=None,
                                    min_sample=None,
                                    rot_gate=None):
    """Robust SVD-based rigid-pose estimation between two ordered keypoint
    sets ``a`` and ``b`` (i.e. ``a[i]`` is matched to ``b[i]``).

    Implements proper minimal-sample RANSAC:

    1. Repeat ``num_iter`` times: pick ``min_sample`` random correspondences,
       solve SVD on that subset, count inliers within ``inlier_thresh`` after
       transforming ``a`` by the candidate pose.
    2. Keep the hypothesis with the largest inlier count (ties broken by
       lowest mean inlier residual), discarding hypotheses whose rotation
       angle exceeds ``rot_gate`` (radians).
    3. Re-fit SVD on all inliers of the best hypothesis to tighten the pose.

    The previous implementation sampled 30% of correspondences (too many for
    real RANSAC) and used a 30-unit inlier threshold (meaningless on data
    normalised to ``[-1, 1]``), so the routine effectively returned the SVD
    of a random subset.  This version follows the textbook recipe.
    """

    n = a.shape[0]
    if a.shape != b.shape:
        print("not a pairing set.")
        return None, None, None

    if n < 6:
        print("not enough points.")
        return None, None, None

    inlier_thresh = RANSAC_INLIER_THRESH if inlier_thresh is None else inlier_thresh
    num_iter = RANSAC_NUM_ITER if num_iter is None else num_iter
    min_sample = max(3, RANSAC_MIN_SAMPLE if min_sample is None else min_sample)
    rot_gate = RANSAC_ROT_GATE if rot_gate is None else rot_gate

    best_inliers = None
    best_score = (-1, np.inf)
    best_t, best_r = None, None

    for _ in range(num_iter):
        indexes = random.sample(range(n), min_sample)
        a_s = a[indexes, :]
        b_s = b[indexes, :]

        t, r = solve_svd(a_s, b_s)
        if t is None or r is None:
            continue
        try:
            angle = abs(Quaternion(matrix=r).angle)
        except Exception:
            continue
        if angle > rot_gate:
            continue

        a_to_b = rotate_np_points_with_t_r(a, t, r)
        residuals = np.linalg.norm(a_to_b - b, axis=1)
        inliers = np.where(residuals < inlier_thresh)[0]
        if len(inliers) < min_sample:
            continue
        score = (len(inliers), float(residuals[inliers].mean()))

        if (score[0] > best_score[0]) or \
           (score[0] == best_score[0] and score[1] < best_score[1]):
            best_score = score
            best_inliers = inliers
            best_t, best_r = t, r

    if best_inliers is None or len(best_inliers) < min_sample:
        return None, None, None

    refit_t, refit_r = solve_svd(a[best_inliers], b[best_inliers])
    if refit_t is not None and refit_r is not None:
        try:
            if abs(Quaternion(matrix=refit_r).angle) <= rot_gate:
                a_to_b = rotate_np_points_with_t_r(a, refit_t, refit_r)
                residuals = np.linalg.norm(a_to_b - b, axis=1)
                refit_inliers = np.where(residuals < inlier_thresh)[0]
                if len(refit_inliers) >= len(best_inliers):
                    best_t, best_r = refit_t, refit_r
                    best_score = (len(refit_inliers),
                                  float(residuals[refit_inliers].mean()))
        except Exception:
            pass

    return best_r, best_t, best_score[1]

def solve_svd(a, b,c_a=None,c_b=None):

    t, r = rigid_transform_3D(a, b,c_a,c_b)

    # tr = create_homogenous_matrix(t, r)
    # r, t = seperate_rotation_and_translation(tr)

    return t, r
    # http://nghiaho.com/?page_id=671

    if(a.shape[0] < 6 or b.shape[0] < 6):
        print("not enough points.")
        return

    centroid_a = np.mean(a, axis=0)
    centroid_b = np.mean(b, axis=0)

    h = ((a - centroid_a).transpose().dot(b - centroid_b)).transpose()
    u, s, v = np.linalg.svd(h, full_matrices=False)
    r = v.transpose().dot(u.transpose())
    t = centroid_b - r.dot(centroid_a)
    # t = centroid_b - centroid_a

    tr_t, tr_r = np.identity(4), np.identity(4)
    tr_t[0:3, 3] = t
    tr_r[0:3, 0:3] = r  

    return tr_t, tr_r

def get_accuracy_between_transformations(r1, t1, r2, t2, points, th=100):

    points1 = rotate_np_points_with_t_r(np.asarray(points.points), t1, r1)
    points2 = rotate_np_points_with_t_r(np.asarray(points.points), t2, r2)

    goods = 0
    for i in range(len(points1)):
        if np.linalg.norm(points1[i] - points2[i]) < th:
            goods += 1
    
    accuracy = goods / len(points1)
    
    return accuracy


def rotate_np_points_with_t_r(points, t, r):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.rotate(r,False)
    pcd.translate(t, True)

    points_final = np.asarray(pcd.points)

    return points_final

from math import sqrt

def rigid_transform_3D(A, B,C_A,C_B):

    A = np.mat(A.T)
    B = np.mat(B.T)

    assert len(A) == len(B)

    num_rows, num_cols = A.shape

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    if C_A is not None:
        centroid_A=np.matrix(C_A).T
        centroid_B=np.matrix(C_B).T

    # subtract mean
    Am = A - np.tile(centroid_A, (1, num_cols))
    Bm = B - np.tile(centroid_B, (1, num_cols))

    # dot is matrix multiplication for array
    H = Am * np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...\n")
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A + centroid_B
    # t = -centroid_A + centroid_B

    return t, R