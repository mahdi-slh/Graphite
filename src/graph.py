"""Th functions needed to create graph."""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
import torch.nn.functional as F
import cv2
import copy
from params import *
from pointcloud_utils import *
from visualize_graph import visualize_graph

from sklearn.neighbors import RadiusNeighborsTransformer, NearestNeighbors
from augmentations import *
from scipy.spatial.transform import Rotation as R
from pylab import cross,dot,inv,det,transpose
from sklearn import decomposition

exp_alpha = 0.5  # for the exp function
k = 7 # min number of neighbors
nbrs_knn_general = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
nbrs_knn= NearestNeighbors(n_neighbors=6, algorithm='ball_tree')

def get_edges_and_adjancey_matrix_from_indices(indices):
    """Create 2xE and NxN edge and adjency matrix from list of lenght of number of neighbors for each node."""
    number_of_nodes = indices.shape[0]
    number_of_edges = 0

    for i in range(indices.shape[0]):
        number_of_edges += len(indices[i])

    edges = np.zeros([2, number_of_edges], dtype=int)
    adjancey_matrix = np.zeros([number_of_nodes, number_of_nodes], dtype=int)

    counter = 0
    for i in range(number_of_nodes):
        for j in range(len(indices[i])):
            edges[:, counter] = [i, indices[i][j]]
            adjancey_matrix[i, indices[i][j]] = 1
            counter += 1

    return edges, adjancey_matrix

def find_optimom_distance_threshold(positions, feature_index):


    # nbrs_knn_general.fit(positions)
    # dist, _ = nbrs_knn_general.kneighbors(positions, return_distance=True)

    # if(feature_index > -1):
    #     dist_final = dist[feature_index, k//2 + 1]
    # else:
    #     dist_final = np.mean(dist[:, k//2 + 1])

    # print(dist_final)
    if NUMBER_OF_NODES_IN_GRAPH==225:
        return 0.18
    elif NUMBER_OF_NODES_IN_GRAPH==81:
        return 0.4

def get_neighbour_indices(positions, distance=0, knn=True):

    if(knn == False):

        rnt = RadiusNeighborsTransformer(mode='distance', radius=distance)
        rnt_result = rnt.fit(positions)
        result = rnt_result.radius_neighbors(positions, return_distance=False)

        return result

    if(knn == True):
        nbrs_knn.fit(positions)
        result = nbrs_knn.kneighbors(positions, return_distance=False)

        return result

def get_node_labels(adjancey_matrix, feature_index, number_of_nodes):

    if(feature_index is None):
        return np.zeros(number_of_nodes)

    dist_matrix = shortest_path(adjancey_matrix)
    distances = dist_matrix[feature_index, :].squeeze()
    labels = np.expand_dims([np.exp(-distances[xi] * exp_alpha) for xi in range(0, number_of_nodes)], axis=1)

    return labels


def find_matrix_vec2vec(U,V):
    W=cross(U,V)
    A=np.array([U,W,cross(U,W)]).T
    B=np.array([V,W,cross(V,W)]).T
    if not det(A)==0:
        return dot(B,inv(A))
    else:
        return dot(B,transpose(A))


def normalize_patch(patch):
    # coor=o3d.geometry.TriangleMesh.create_coordinate_frame()
    positions = np.asarray(patch.points)
    positions_orig=copy.deepcopy(positions)
    center=np.mean(positions,0)
    patch_temp=o3d.geometry.PointCloud()
    copy_pc(patch_temp,patch)
    patch.translate(-center)

    
    

    scale=np.max(np.abs(positions))
    positions=positions/scale
    pca=decomposition.PCA(3)

    pca.fit(positions)
    positions = pca.transform(positions)

    bumpyness=np.clip(np.min(pca.explained_variance_ratio_)/np.max(pca.explained_variance_ratio_),0.0,0.5)*2
    # if np.asarray(patch.normals).shape[0]==0:
    #     patch.estimate_normals()
    #     patch.normalize_normals()
    # normals = np.asarray(patch.normals)
    # normals_avg=np.mean(normals,0)
    # normals_avg=normals_avg/np.linalg.norm(normals_avg)
    # # rot = R.from_rotvec(np.pi/2 * normals_avg)

    # bu=np.pi/2 *normals_avg
    # au=[np.pi/2,0,0]
    # # Ra_b=np.array([[bu[0]*au[0], bu[0]*au[1], bu[0]*au[2]], [bu[1]*au[0], bu[1]*au[1], bu[1]*au[2]], [bu[2]*au[0], bu[2]*au[1], bu[2]*au[2]] ])
    # Ra_b=find_matrix_vec2vec(au,bu)
    # # Ra_b=R.from_matrix(Ra_b)
    # patch.rotate(Ra_b,False)
    
    # positions = np.asarray(patch.points)
    scale=np.max(np.abs(positions))
    positions=positions/scale

    # colors = np.asarray(patch.colors)
    patch_norm=o3d.geometry.PointCloud()
    patch_norm.points=o3d.utility.Vector3dVector(positions)
    # patch_norm.colors=o3d.utility.Vector3dVector(colors)
    patch_norm.estimate_normals()
    patch_norm.normalize_normals()
    patch_norm.orient_normals_to_align_with_direction([0,0,1])
    # normals = np.abs(np.asarray(patch_norm.normals))
    # print(np.mean(normals,0))
    # o3d.visualization.draw_geometries([coor,patch_temp,patch_norm])
    return patch_norm,positions_orig,bumpyness
    

def visualize_torch_graphs(graph_torch1,graph_torch2):

    coor=o3d.geometry.TriangleMesh.create_coordinate_frame()

    nodes_np1=graph_torch1.x.cpu().detach().numpy()[:,3:]#graph_torch1.positions.cpu().detach().numpy()
    norm_np1=graph_torch1.x.cpu().detach().numpy()[:,0:3]
    points_o3d1=o3d.geometry.PointCloud()
    points_o3d1.points=o3d.utility.Vector3dVector(nodes_np1)
    points_o3d1.normals=o3d.utility.Vector3dVector(norm_np1)
    # points_o3d1.estimate_normals()
    points_o3d1.paint_uniform_color([0,1,0])
    col=np.asarray(points_o3d1.colors)
    # col[torch.argmax(graph_torch1.y).cpu().numpy(),:]=[0,0,0]
    col[0,:]=[0,0,0]
    points_o3d1.colors=o3d.utility.Vector3dVector(col)

    nodes_np2=graph_torch2.x.cpu().detach().numpy()[:,3:]
    norm_np2=graph_torch2.x.cpu().detach().numpy()[:,0:3]
    points_o3d2=o3d.geometry.PointCloud()
    points_o3d2.points=o3d.utility.Vector3dVector(nodes_np2)
    points_o3d2.normals=o3d.utility.Vector3dVector(norm_np2)
    # points_o3d2.estimate_normals()
    points_o3d2.paint_uniform_color([1,0.0,0])

    col=np.asarray(points_o3d2.colors)
    # col[torch.argmax(graph_torch2.y).cpu().numpy(),:]=[0,0,0]
    col[0,:]=[0,0,0]
    points_o3d2.colors=o3d.utility.Vector3dVector(col)

    o3d.visualization.draw_geometries([points_o3d1,coor])
    o3d.visualization.draw_geometries([points_o3d2,coor])
    o3d.visualization.draw_geometries([points_o3d1,points_o3d2,coor])

def create_graph(points_inp, feature_index, false_color=False, augment=False,center=np.asarray([-1,-1,-1])):

    # if(augment):
    #     points=random_perturb(points,coef=0.01)
    points=copy.deepcopy(points_inp)
    points,orig_positions,confidence=normalize_patch(points)

    positions = np.asarray(points.points)
    colors = np.asarray(points.normals)
    normals = np.abs(np.asarray(points.normals))

    # if(false_color):
    #     colors = np.asarray(points.normals)

    #ball
    # print(np.max(positions,0)-np.min(positions,0))
    
    distance = find_optimom_distance_threshold(positions, feature_index)
    
    indices = get_neighbour_indices(positions, distance, knn=False)
    #for 3D edge attribs
    # positions_distance_1 = squareform(pdist(positions, lambda u, v: np.abs(u[0]-v[0])))
    # positions_distance_2 = squareform(pdist(positions, lambda u, v: np.abs(u[1]-v[1])))
    # positions_distance_3 = squareform(pdist(positions, lambda u, v: np.abs(u[2]-v[2])))
    # positions_distance = np.stack((positions_distance_1, positions_distance_2,positions_distance_3), axis=-1)

    #1D edge attribs 
    positions_distance = squareform(pdist(positions)) 

    # #knn
    # nbrs = NearestNeighbors(n_neighbors = 6, algorithm='ball_tree').fit(positions)
    # indices = nbrs.kneighbors(positions, return_distance=False)

    edges, adjancey_matrix = get_edges_and_adjancey_matrix_from_indices(indices)
    edge_distances = np.array([positions_distance[edges[0, i], edges[1, i]] for i in range(0, edges.shape[1])])
    labels = get_node_labels(adjancey_matrix, feature_index, positions.shape[0])

    edge_attr = torch.tensor(edge_distances, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.int64)
    # print(edge_index.shape)
    

    colors = torch.tensor(colors, dtype=torch.float32)
    normals = torch.tensor(normals, dtype=torch.float32)
    positions = torch.tensor(positions, dtype=torch.float32)
    confidence=torch.tensor(np.expand_dims(confidence,(0,1)), dtype=torch.float32)

    x = torch.cat((normals, positions), 1)
    # x=normals
    y = torch.tensor(labels, dtype=torch.float32)

    orig_positions = torch.tensor(orig_positions, dtype=torch.float32)

    #edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_attr = distance / (edge_attr + distance)

    if center[0]==-1 and center[1]==-1:
        return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, positions=orig_positions,confidence=confidence)
    else:
        center=torch.tensor(center, dtype=torch.float32)
        return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, positions=orig_positions,confidence=confidence,center=center)