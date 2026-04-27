import open3d as o3d
import numpy as np

import random
from pyquaternion import Quaternion
from sklearn.neighbors import NearestNeighbors


def find_optimum_distance_threshold(positions,k=2,operation=np.mean):
    nbrs_knn= NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nbrs_knn.fit(positions)
    dist, _ = nbrs_knn.kneighbors(positions, return_distance=True)
    dist_final = operation(dist[-1:,:])

    return dist_final

def copy_pc(output, point_cloud):
    output.colors=point_cloud.colors
    output.normals=point_cloud.normals
    output.points=point_cloud.points
    return True

def random_perturb(point_cloud, adaptive=True, coef=0.1,sigma=0.01,clip=0.05):
    #moves points randomly in 3D
    point_cloud_np=np.asarray(point_cloud.points)
    if adaptive:

        size_points=np.asarray(point_cloud.points).shape[0]
        #points_subsample=point_cloud.uniform_down_sample(size_points//100)
        #points_subsample_np=np.asarray(points_subsample.points)
        dist=find_optimum_distance_threshold(np.asarray(point_cloud.points))
        perturb=(np.random.rand(point_cloud_np.shape[0],3)-0.5)*dist*coef
    else:
        perturb=np.clip(sigma * np.random.randn(point_cloud_np.shape[0],3), -1 * clip, clip)
        
    
    #point_cloud.points=o3d.utility.Vector3dVector(point_cloud_np+perturb)
    output=o3d.geometry.PointCloud()
    copy_pc(output,point_cloud)
    output.points=o3d.utility.Vector3dVector(point_cloud_np+perturb)
    
    return output

def color_jitter(point_cloud,  amplitude=0.3):
    #adds a global color (brightness,hue) change to all points
    colors=np.asarray(point_cloud.colors)
    
    jitter=np.repeat((np.random.rand(1,3)-amplitude)*amplitude*2,colors.shape[0],0)
    colors_new=np.clip(colors+jitter,0,1)
    output=o3d.geometry.PointCloud()
    copy_pc(output,point_cloud)
    output.colors=o3d.utility.Vector3dVector(colors_new)
    return output

def color_noise(point_cloud,  amplitude=0.1):
    #add a noise to colors
    colors=np.asarray(point_cloud.colors)
    
    noise=(np.random.rand(colors.shape[0],3)-amplitude)*amplitude*2
    colors_new=np.clip(colors+noise,0,1)
    output=o3d.geometry.PointCloud()
    copy_pc(output,point_cloud)
    output.colors=o3d.utility.Vector3dVector(colors_new)
    return output

def random_transform(point_cloud,pose=None,translate=False,rotate=False,scale=False,translate_amp=1.0,scale_amp=0.2):
    #performs a random transformation to Point cloud and it's pose (not tested)
    point_cloud_trans=o3d.geometry.PointCloud(point_cloud)
    # if not (translate and rotate and scale):
    #     print("warning: no transform mode selected")
    values=[]
    if rotate:
        quat=Quaternion(np.random.rand(4,1)*2-1).normalised
        point_cloud_trans.rotate(np.asarray(quat.rotation_matrix),False)
        if pose is not None:
            Quaternion(pose.rotation).rotate(quat)
        values.append(quat.rotation_matrix)
    if translate:
        trans=np.random.rand(3,1)*translate_amp*2-translate_amp
        point_cloud_trans.translate(trans)
        if pose is not None:
            pose.translation=pose.translation+trans
        values.append(trans)
    if scale:
        sca=np.random.rand()*scale_amp*2-scale_amp+1
        point_cloud_trans.scale(sca)
    return point_cloud_trans,values


def random_subsample(point_cloud,max_remove,fixed=True):
    if isinstance(point_cloud,list):
        return [random_subsample(x,max_remove,fixed) for x in point_cloud]
    #randomly drops a number of points, fixed parameter defines if the the max_remove is a range or fixed
    point_cloud_size=np.asarray(point_cloud.points).shape[0]
    if fixed:
        sub_size=point_cloud_size-max_remove
    else:
        sub_size=np.random.randint(point_cloud_size-max_remove,point_cloud_size)

    permute=np.random.permutation(point_cloud_size)
    point_cloud_sub=point_cloud.select_down_sample(list(permute[:sub_size]))
    return point_cloud_sub

def random_upsample(point_cloud,max_add,fixed=True):
    #not yet implemented
    if isinstance(point_cloud,list):
        return [random_upsample(x,max_add,fixed) for x in point_cloud]
    mesh=o3d.geometry.TriangleMesh()
    radius=find_optimum_distance_threshold(point_cloud,5)
    radii = [radius, radius*2]
    # mesh.create_from_point_cloud_ball_pivoting(point_cloud, o3d.utility.DoubleVector(radii))

    # o3d.visualization.draw_geometries([mesh,point_cloud])
    
    # print('bill')

    return point_cloud