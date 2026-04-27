import random
import numpy as np
import torch
import open3d as o3d
import math
from pyquaternion import Quaternion
from scipy.spatial import distance

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

def get_pose_from_two_matching_sets(a, b):
    
    n = a.shape[0]
    if(a.shape != b.shape):
        print("not a pairing set.")
        return

    number_of_points = n
    # print(n)
    number_of_samples =int(number_of_points*0.3)


    if(n < 6):
        print("not enough points.")
        return None, None, None

    
    number_of_iterations = 1000

    min_dif = 1000
    best_tr_t = None
    best_tr_r = None

    for i in range(number_of_iterations):

        indexes = random.sample(range(0, number_of_points), number_of_samples)

        a_s = a[indexes, :]
        b_s = b[indexes, :]

        cent_a=np.mean(a[range(0,n), :],0)
        cent_b=np.mean(b[range(0,n), :],0)
        t, r = solve_svd(a_s, b_s,cent_a,cent_b)

        a_to_b = rotate_np_points_with_t_r(a, t, r)
        dist_temp = np.linalg.norm(a_to_b - b, axis=1)
        ind = np.where(dist_temp < 30)[0]
        if(len(ind) < 6) or abs(Quaternion(matrix=r).angle)>1.7:
            continue
        dif = np.mean(np.linalg.norm(a_to_b - b, axis=1)[ind])
        
        if(dif < min_dif):
            best_tr_t = t
            best_tr_r = r
            # print(Quaternion(matrix=r).angle)
            min_dif = dif

    # print(min_dif)

    return best_tr_r, best_tr_t, min_dif

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