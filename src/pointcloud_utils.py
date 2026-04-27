import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors,RadiusNeighborsClassifier
from params import NUMBER_OF_NODES_IN_GRAPH
from augmentations import copy_pc
from scipy.spatial import distance_matrix
import random
from sklearn import decomposition

def create_random_patches(points, cluster_size=NUMBER_OF_NODES_IN_GRAPH, uniform=False,augment=False,scale=40,count=60):

    points_np=np.asarray(points.points)
    number_points=points_np.shape[0]

    # cols=np.asarray(points.colors)
    # r=np.expand_dims(np.arange(2000),1)/2000
    # # r=np.repeat(r,3,1)
    # r=np.tile(r,(number_points//2000-1,3))
    # cols[0:r.shape[0],:]=r
    # points.colors=o3d.utility.Vector3dVector(cols)

    # number_patches=number_points//(cluster_size*3)
    # subsample_rate=np.maximum((number_points)//(NUMBER_OF_NODES_IN_GRAPH*1.3*count),1)
    subsample_rate_cen=(number_points)//count

    if uniform:
        centroids_np=np.asarray(points.uniform_down_sample(int(subsample_rate_cen)).points)
    else:
        # points_np_sorted=points_np[np.lexsort(points_np.T),:]
        # centroids_np=points_np_sorted[range(0,number_points,subsample_rate_cen),:]#
        randperm=np.random.permutation(number_points)
        centroids_np=np.asarray(points.select_down_sample(randperm[0:count]).points)
        
    
    # max_bound=np.median(points.get_max_bound()-points.get_min_bound())
    # vox_size=max_bound/scale
    # points_sub=points.voxel_down_sample(vox_size)
    # points_sub=points.uniform_down_sample(int(subsample_rate))
    # points_sub_np=np.asarray(points_sub.points)


    nbrs_knn= NearestNeighbors(n_neighbors=NUMBER_OF_NODES_IN_GRAPH, algorithm='ball_tree')
    nbrs_knn.fit(points_np)
    labels = nbrs_knn.kneighbors(centroids_np, return_distance=False)

    if augment:
        new_ids=labels[np.arange(centroids_np.shape[0]),np.random.randint(0,10,centroids_np.shape[0])]
        new_centroids=points_np[new_ids,:]
        labels = nbrs_knn.kneighbors(new_centroids, return_distance=False)


    patches=[]
    for patch_ctr in range(labels.shape[0]):
        patch=points.select_down_sample(labels[patch_ctr,:])
        # patch.colors=o3d.utility.Vector3dVector(np.repeat(np.random.rand(1,3),cluster_size,0))
        patches.append(patch)
    # nbrs_knn= NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    # nbrs_knn.fit(centroids_np)
    # labels = nbrs_knn.kneighbors(points_np, return_distance=False)
    # labels=labels[:,0]
    # patches=[]
    # for patch_ctr in range(number_patches):
    #     selection=list(np.where(labels==patch_ctr)[0])
    #     patch=points.select_down_sample(selection)
    #     if random_sample:
    #         sub_sample=np.random.permutation(len(selection))
    #         patch=patch.select_down_sample(list(sub_sample[:cluster_size]))
    #     else:
    #         if len(selection)<cluster_size:
    #             continue
    #         rate=len(selection)//cluster_size
    #         sub_sample=range(0,cluster_size*rate,rate)
    #         patch=patch.select_down_sample(list(sub_sample))
    #     patch.colors=o3d.utility.Vector3dVector(np.repeat(np.random.rand(1,3),cluster_size,0))
    #     if np.asarray(patch.points).shape[0] is cluster_size:
    #         patches.append(patch)
    centroids_o3d=o3d.geometry.PointCloud()
    centroids_o3d.points=o3d.utility.Vector3dVector(centroids_np)
    centroids_o3d.paint_uniform_color([1,0,0])
    return patches,centroids_o3d

def create_indexed_patches(points,indexes, cluster_size=NUMBER_OF_NODES_IN_GRAPH, uniform=True,radius=5,voxel_size=0.04):

    points_np=np.asarray(points.points)

    # centroids_o3d=points.select_down_sample(indexes)
    centroids_np=points_np[indexes,:]
    points=points.voxel_down_sample(voxel_size)

    points_np=np.asarray(points.points)
    # nbrs_radius= NearestNeighbors(radius=0.18, algorithm='ball_tree')
    # nbrs_radius.fit(points_np)
    # labels = nbrs_radius.radius_neighbors(centroids_np, return_distance=False)
    nbrs_radius= NearestNeighbors(n_neighbors=NUMBER_OF_NODES_IN_GRAPH, algorithm='ball_tree')
    nbrs_radius.fit(points_np)
    labels = nbrs_radius.kneighbors(centroids_np, return_distance=False)
    


    # subsample_rate=np.maximum((number_points)//(NUMBER_OF_NODES_IN_GRAPH*1.0*count),1)
    # subsample_rate_cen=(number_points/10)//count
    


    # if uniform:
    #     centroids_np=np.asarray(points.uniform_down_sample(int(subsample_rate_cen)).points)
    #     points_sub=points.uniform_down_sample(int(subsample_rate))
    #     points_sub_np=np.asarray(points_sub.points)
    # else:
    #     # randperm=np.random.permutation(number_points)
    #     # centroids_np=np.asarray(points.select_down_sample(randperm[0:count]).points)
    #     max_bound=np.median(points.get_max_bound()-points.get_min_bound())
    #     vox_size=max_bound/scale
    #     centroids_np=np.asarray(points.voxel_down_sample(vox_size).points)
    #     points_sub=points.voxel_down_sample(vox_size/7)
    #     points_sub_np=np.asarray(points_sub.points)
    



    # nbrs_knn= NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    # nbrs_knn.fit(centroids_np)
    # labels = nbrs_knn.kneighbors(points_sub_np, return_distance=False)
    # labels=labels[:,0]
    # patches=[]
    # for patch_ctr in range(count):
    #     selection=list(np.where(labels==patch_ctr)[0])
    #     patch=points_sub.select_down_sample(selection)
    #     if not uniform:
    #         sub_sample=np.random.permutation(len(selection))
    #         patch=patch.select_down_sample(list(sub_sample[:cluster_size]))
    #     else:
    #         if len(selection)<cluster_size:
    #             continue
    #         rate=len(selection)//cluster_size
    #         sub_sample=range(0,cluster_size*rate,rate)
    #         patch=patch.select_down_sample(list(sub_sample))
    #     patch.colors=o3d.utility.Vector3dVector(np.repeat(np.random.rand(1,3),cluster_size,0))
    #     if np.asarray(patch.points).shape[0] is cluster_size:
    #         patches.append(patch)

    # nbrs_knn= NearestNeighbors(n_neighbors=NUMBER_OF_NODES_IN_GRAPH, algorithm='ball_tree')
    # nbrs_knn.fit(points_sub_np)
    # labels = nbrs_knn.kneighbors(centroids_np, return_distance=False)

    patches=[]
    for patch_ctr in range(labels.shape[0]):
        patch_np=points_np[labels[patch_ctr],:]
        patch=o3d.geometry.PointCloud()
        patch.points=o3d.utility.Vector3dVector(patch_np)


    # patches=[]
    # for patch_ctr in range(labels.shape[0]):
    #     patch=labels[patch_ctr]
    #     if patch.shape[0]<NUMBER_OF_NODES_IN_GRAPH:
    #         continue
    #     random_indexes=random.sample(range(0, patch.shape[0]), NUMBER_OF_NODES_IN_GRAPH)
    #     patch_sub_np=points_np[patch[random_indexes],:]
    #     patch=o3d.geometry.PointCloud()
    #     patch.points=o3d.utility.Vector3dVector(patch_sub_np)
       
        # positions=np.asarray(patch.points)
        # center=np.mean(positions,0)
        # positions=positions-center
        # pca=decomposition.PCA(3)
        # pca.fit(positions)
        # positions = pca.transform(positions)
        # if np.max(pca.explained_variance_ratio_)/np.min(pca.explained_variance_ratio_)<10:
        #     patch.paint_uniform_color(np.random.rand(1,3)[0])
            

            
        # else:
        #     patch.paint_uniform_color([0,0,0])

        patch.paint_uniform_color(np.random.rand(1,3)[0])
        patches.append(patch)
            
        

    # centroids_o3d=o3d.geometry.PointCloud()
    # centroids_o3d.points=o3d.utility.Vector3dVector(centroids_np)
    # centroids_o3d.paint_uniform_color([1,0,0])
    return patches,centroids_np


def create_dense_patches(points, cluster_size=NUMBER_OF_NODES_IN_GRAPH, uniform=True,augment=False,scale=5,count=60):

    points_np=np.asarray(points.points)
    number_points=points_np.shape[0]


    subsample_rate=np.maximum((number_points)//(NUMBER_OF_NODES_IN_GRAPH*1.0*count),1)
    subsample_rate_cen=(number_points/10)//count
    


    if uniform:
        centroids_np=np.asarray(points.uniform_down_sample(int(subsample_rate_cen)).points)
        points_sub=points.uniform_down_sample(int(subsample_rate))
        points_sub_np=np.asarray(points_sub.points)
    else:
        # randperm=np.random.permutation(number_points)
        # centroids_np=np.asarray(points.select_down_sample(randperm[0:count]).points)
        max_bound=np.median(points.get_max_bound()-points.get_min_bound())
        vox_size=max_bound/scale
        centroids_np=np.asarray(points.voxel_down_sample(vox_size).points)
        points_sub=points.voxel_down_sample(vox_size/7)
        points_sub_np=np.asarray(points_sub.points)
    



    # nbrs_knn= NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    # nbrs_knn.fit(centroids_np)
    # labels = nbrs_knn.kneighbors(points_sub_np, return_distance=False)
    # labels=labels[:,0]
    # patches=[]
    # for patch_ctr in range(count):
    #     selection=list(np.where(labels==patch_ctr)[0])
    #     patch=points_sub.select_down_sample(selection)
    #     if not uniform:
    #         sub_sample=np.random.permutation(len(selection))
    #         patch=patch.select_down_sample(list(sub_sample[:cluster_size]))
    #     else:
    #         if len(selection)<cluster_size:
    #             continue
    #         rate=len(selection)//cluster_size
    #         sub_sample=range(0,cluster_size*rate,rate)
    #         patch=patch.select_down_sample(list(sub_sample))
    #     patch.colors=o3d.utility.Vector3dVector(np.repeat(np.random.rand(1,3),cluster_size,0))
    #     if np.asarray(patch.points).shape[0] is cluster_size:
    #         patches.append(patch)

    nbrs_knn= NearestNeighbors(n_neighbors=NUMBER_OF_NODES_IN_GRAPH, algorithm='ball_tree')
    nbrs_knn.fit(points_sub_np)
    labels = nbrs_knn.kneighbors(centroids_np, return_distance=False)



    patches=[]
    for patch_ctr in range(labels.shape[0]):
        patch=points_sub.select_down_sample(labels[patch_ctr,:])

        positions=np.asarray(patch.points)
        center=np.mean(positions,0)
        positions=positions-center
        pca=decomposition.PCA(3)
        pca.fit(positions)
        positions = pca.transform(positions)
        if np.max(pca.explained_variance_ratio_)/np.min(pca.explained_variance_ratio_)<10:
            patch.paint_uniform_color(np.random.rand(1,3)[0])
            patches.append(patch)

            
        else:
            patch.paint_uniform_color([0,0,0])
            
        

    centroids_o3d=o3d.geometry.PointCloud()
    centroids_o3d.points=o3d.utility.Vector3dVector(centroids_np)
    centroids_o3d.paint_uniform_color([1,0,0])
    return patches,centroids_o3d


def find_closest_matching_patches(patches1, patches2, t, r, number=10):
    
    cf=o3d.geometry.TriangleMesh.create_coordinate_frame(0.01)
    
    centroids1=[np.mean(np.asarray(p.points),0) for p in patches1]
    centroids1_trans=[np.matmul(r,c)+t.transpose() for c in centroids1]
    centroids2=[np.mean(np.asarray(p.points),0) for p in patches2]
    dist_mat=distance_matrix(centroids1_trans, centroids2)
    indexes =[]
    matches =[]
    output =[]
    matching_centroids =[]
    for i in range(number):
        ind = np.unravel_index(np.argmin(dist_mat, axis=None), dist_mat.shape)
        indexes.append(ind)
        dist_mat[ind[0],:]=1000
        dist_mat[:,ind[1]]=1000
        matches.append(patches1[ind[0]])
        matches.append(patches2[ind[1]])
        pointpairs,weights=find_knn_interpolation(patches1[ind[0]],patches2[ind[1]],t,r)
        output.append([patches1[ind[0]],patches2[ind[1]],pointpairs,weights])
        matching_centroids.append((ind[0],ind[1]))

        
    centroids1_o3d = o3d.geometry.PointCloud()
    centroids2_o3d = o3d.geometry.PointCloud()
    centroids1_o3d.points =o3d.utility.Vector3dVector(np.stack( centroids1, axis=0 ))
    centroids2_o3d.points =o3d.utility.Vector3dVector(np.stack( centroids2, axis=0 ))
    lines_o3d=o3d.geometry.LineSet.create_from_point_cloud_correspondences(centroids1_o3d, centroids2_o3d,matching_centroids)
    
    # o3d.visualization.draw_geometries(matches+[cf,lines_o3d])
    # o3d.visualization.draw_geometries(patches1+patches2+[lines_o3d])
    return output
    
def find_knn_interpolation(patch1_orig,patch2,t,r,k=3):

    patch1=o3d.geometry.PointCloud()
    copy_pc(patch1,patch1_orig)
    patch1.rotate(r,False)
    patch1.translate(t, True)
    
    nbrs_knn= NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nbrs_knn.fit(np.asarray(patch2.points))
    dist_inter_patch, _ = nbrs_knn.kneighbors(np.asarray(patch2.points), return_distance=True)
    thresh=np.max(dist_inter_patch[:,1])
    dist, indices = nbrs_knn.kneighbors(np.asarray(patch1.points), return_distance=True)
    mask=dist[:,0]<thresh
    indices=np.vstack([np.arange(NUMBER_OF_NODES_IN_GRAPH),indices.T]).T
    dist = np.sum(dist,1,keepdims=True)
    dist_inv=1/dist
    dist_inv=np.round_(dist_inv[mask],3)
    indices=indices[mask]

    return indices,dist_inv


def find_rand_matching_patches(patches1,patches2,t,r,cluster_size=NUMBER_OF_NODES_IN_GRAPH):
    # cf=o3d.geometry.TriangleMesh.create_coordinate_frame()
    not_found=True
    while not_found:
        fixed_patches = np.random.randint(0, len(patches1))
        if np.asarray(patches1[fixed_patches].points).shape[0] is not cluster_size:
            continue
        
        random_5_numbers=np.random.permutation(NUMBER_OF_NODES_IN_GRAPH)[0:5]
        random_5_points = patches1[fixed_patches].select_down_sample(list(random_5_numbers))
        random_5_points_t = o3d.geometry.PointCloud()
        random_5_points_t.points = o3d.utility.Vector3dVector(np.asarray(random_5_points.points))
        
        random_5_points_t.rotate(r,False)
        random_5_points_t.translate(t)
        
        matching_lines=o3d.geometry.LineSet.create_from_point_cloud_correspondences(random_5_points, random_5_points_t,[(0,0),(1,1),(2,2),(3,3),(4,4)])
        random_5_points_t_np=np.asarray(random_5_points_t.points)
        nbrs_knn= NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
        
        nbrs_knn.fit(random_5_points_t_np)
        
        # o3d.visualization.draw_geometries(patches1 + patches2 + [matching_lines,cf])

        dist, _ = nbrs_knn.kneighbors(random_5_points_t_np, return_distance=True)
        max_dist_5_points=np.max(dist)
        for patch_tar in patches2:
            patch_tar_np=np.asarray(patch_tar.points)
            
            dist, _ = nbrs_knn.kneighbors(patch_tar_np, return_distance=True)
            max_dist=np.mean(np.max(dist,1))
            if max_dist<max_dist_5_points:
                match_patch=patch_tar
                not_found=False
                break
                
    # o3d.visualization.draw_geometries([patches1[fixed_patches],patch_tar,matching_lines,cf])

    return [matching_lines,patches1[fixed_patches], patch_tar]

def patch_and_match(pc1,pc2,t=np.zeros([3,1],dtype=float),r=np.identity(3,dtype=float),cluster_size=NUMBER_OF_NODES_IN_GRAPH,num=5):
    patch_tuples=[]
    for i in range(num):
        patches1=create_random_patches(pc1,cluster_size=cluster_size,random=True)
        patches2=create_random_patches(pc2,cluster_size=cluster_size,random=True)
        patch_tuples.append(find_rand_matching_patches(patches,patches2,t,r,cluster_size=cluster_size))

    return patch_tuples


def permute_pointcloud(points, fearture_index):

    output = o3d.geometry.PointCloud()
    copy_pc(output, points)

    indexes = np.random.permutation(len(points.points))

    output.colors = o3d.utility.Vector3dVector(np.asarray(points.colors)[[indexes], :].squeeze())
    output.normals = o3d.utility.Vector3dVector(np.asarray(points.normals)[[indexes], :].squeeze())
    output.points = o3d.utility.Vector3dVector(np.asarray(points.points)[[indexes], :].squeeze())

    new_index = list(indexes).index(fearture_index)

    return output, new_index