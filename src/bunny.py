import open3d as o3d
import numpy as np
from augmentations import *
import csv

from sklearn.neighbors import NearestNeighbors
# from pointcloud_utils import *
import itertools

NUMBER_OF_NODES_IN_GRAPH=81
ROOT2BUNNY='../../../dataset/bunny/'

def create_random_patches(points, cluster_size=81, number_patches=120,random=False):

    points_np=np.asarray(points.points)
    number_points=points_np.shape[0]
    subsample_rate=number_points//(number_patches)

    if random:
        randperm=np.random.permutation(number_points)
        centroids_np=np.asarray(points.select_down_sample(randperm[0:number_patches]).points)
    else:
        centroids_np=np.asarray(points.uniform_down_sample(subsample_rate).points)

    nbrs_knn= NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    nbrs_knn.fit(centroids_np)
    labels = nbrs_knn.kneighbors(points_np, return_distance=False)
    labels=labels[:,0]
    patches=[]
    centroids_o3d=o3d.geometry.PointCloud()
    centroids_o3d.points=o3d.utility.Vector3dVector(centroids_np)
    for patch_ctr in range(number_patches):
        selection=list(np.where(labels==patch_ctr)[0])
        patch=points.select_down_sample(selection)
        sub_sample=np.random.permutation(len(selection))
        patch=patch.select_down_sample(list(sub_sample[:cluster_size]))
        patch.colors=o3d.utility.Vector3dVector(np.repeat(np.random.rand(1,3),cluster_size,0))
        patches.append(patch)


    return patches,centroids_o3d



class PatchVisualizer:
    def __init__(self,patches):
        self.patches=patches
        self.focus=0
    def add_patches(self,vis):
        vis.clear_geometries()
        vis.add_geometry(self.patches[self.focus])
        if self.focus== len(self.patches)-1:
            self.focus=0
        else:
            self.focus+=1
            
        return False


class BunnyLoader:
    def __init__(self,root):
        self.root=root
        self.scans=[]
        self.translations=[]
        self.quaternions=[]
        with open('%sdata/bun.conf'%root, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                if len(row)>0 and row[0]=='bmesh':
                    self.scans.append(row[1])
                    self.translations.append([row[2],row[3],row[4]])
                    self.quaternions.append([row[8],row[5],row[6],row[7]])
    
    def get_item(self,index):

        point_cloud=o3d.io.read_point_cloud('%sdata/%s'%(self.root,self.scans[index]))
        point_cloud.estimate_normals()
        point_cloud.normalize_normals()
        point_cloud.colors=o3d.utility.Vector3dVector(np.asarray(point_cloud.normals)*0.5+0.5)

        pose=[self.translations[index],self.quaternions[index]]
        return point_cloud


if __name__=='__main__':

    loader= BunnyLoader(ROOT2BUNNY)        
    bun= loader.get_item(0)
   # o3d.visualization.draw_geometries([bun])
    #apply augmentations
    bun1=random_perturb(bun,coef=0.000001)
    b1_sub=bun1.voxel_down_sample(0.001)
    bun2=random_perturb(bun,coef=0.000001)
    # bun=color_jitter(bun,0.2)
    # bun=color_noise(bun,0.2)
    bun2,[r,t]=random_transform(bun2,rotate=True,translate=True,translate_amp=0.2)
    

    patches,c=create_random_patches(bun1,cluster_size=NUMBER_OF_NODES_IN_GRAPH,random=True)
    b1_sub.paint_uniform_color([0,0,0])
    o3d.visualization.draw_geometries([b1_sub])
    o3d.visualization.draw_geometries(patches)
    o3d.visualization.draw_geometries([c])
    patches2=create_random_patches(bun2,cluster_size=NUMBER_OF_NODES_IN_GRAPH,random=True)
    # results=find_closest_matching_patches(patches,patches2,t,r,20)
    #create patches
    # matches=patch_and_match(bun1,bun2,t,r,cluster_size=NUMBER_OF_NODES_IN_GRAPH,num=5)
    # patches=create_random_patches(bun1,cluster_size=NUMBER_OF_NODES_IN_GRAPH,number_patches=200,random=True)
    # patches2=create_random_patches(bun2,cluster_size=NUMBER_OF_NODES_IN_GRAPH,number_patches=200,random=True)
    # [lines,patch1,patch2]=find_matching_patches(patches,patches2,t,r)
    
    #patch augmentations
    #patches=random_subsample(patches,50,fixed=False)
    
    # vis=o3d.visualization.Visualizer()
    # func=vis.add_geometry(bun,reset_bounding_box=False)
    # visu=PatchVisualizer(patches)
    # key_to_callback = {}
    # key_to_callback[ord("K")] = visu.add_patches
    matches_all_list=[matches[i] for i in range(5)]
    flatten = list(itertools.chain(*matches_all_list))

    #o3d.visualization.draw_geometries(patches+patches2+[lines])
    o3d.visualization.draw_geometries([bun1,bun2] +flatten)
    # o3d.visualization.draw_geometries_with_key_callbacks([bun],key_to_callback)
    