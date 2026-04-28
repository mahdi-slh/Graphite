import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from torch.nn import Sequential, ModuleList, Linear, ReLU, Module, ELU, BatchNorm1d,Conv1d
from torch_scatter import scatter_mean, scatter_min, scatter_max, scatter_add
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models import GraphUNet
import torch.nn.functional as F
from torch_geometric.nn import max_pool_x,TAGConv,SGConv
import numpy as np

from params import NUMBER_OF_NODES_IN_GRAPH
from collections import OrderedDict
from params import *

class EdgeModel(torch.nn.Module):

    def __init__(self, edge_attr_sz_in, edge_attr_sz_out, node_attr_sz):

        super(EdgeModel, self).__init__()

        # feature_in = edge_attr_sz_in + 2 * node_attr_sz

        self.conv1 = torch.nn.Conv1d(edge_attr_sz_in, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, edge_attr_sz_out)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        x = torch.unsqueeze(edge_attr, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return x

class NodeModel(torch.nn.Module):

    def __init__(self, node_attr_sz_in, node_attr_sz_out, edge_attr_sz):

        super(NodeModel, self).__init__()

        self.reshape_value = node_attr_sz_out * 8

        self.conv1 = torch.nn.Conv1d(node_attr_sz_in + edge_attr_sz, node_attr_sz_out, 1)
        self.conv2 = torch.nn.Conv1d(node_attr_sz_out, node_attr_sz_out * 2, 1)
        self.conv3 = torch.nn.Conv1d(node_attr_sz_out * 2, node_attr_sz_out * 8, 1)
        self.fc1 = nn.Linear(node_attr_sz_out * 8, node_attr_sz_out * 4)
        self.fc2 = nn.Linear(node_attr_sz_out * 4, node_attr_sz_out * 2)
        self.fc3 = nn.Linear(node_attr_sz_out * 2, node_attr_sz_out)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(node_attr_sz_out)
        self.bn2 = nn.BatchNorm1d(node_attr_sz_out * 2)
        self.bn3 = nn.BatchNorm1d(node_attr_sz_out * 8)
        self.bn4 = nn.BatchNorm1d(node_attr_sz_out * 4)
        self.bn5 = nn.BatchNorm1d(node_attr_sz_out * 2)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        row, col = edge_index
        x = torch.cat([x[col], edge_attr], dim=1)
        x = torch.unsqueeze(x, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, self.reshape_value)

        x = scatter_mean(x, row, dim=0)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return x

class Net(torch.nn.Module):
    def __init__(self, edge_attr_in_first=3, node_attr_in_first=3):

        super(Net, self).__init__()
        
        self.softmax = torch.nn.Softmax(dim=1)
        self.ml_shared = get_metalayer(edge_attr_in_first, edge_attr_in_first * 2, node_attr_in_first, FEATURE_SIZE // 4)
        self.ml_d = ModuleList()
        self.ml_l = ModuleList()

        for i in range(2):
            
            if i==0:
                node_attr_sz_in = FEATURE_SIZE // 4
                node_attr_sz_out= FEATURE_SIZE // 2
                edge_attr_sz_in = edge_attr_in_first * 2 
                edge_attr_sz_out = edge_attr_in_first * 4

            if i==1:
                node_attr_sz_in = FEATURE_SIZE // 2
                node_attr_sz_out= FEATURE_SIZE
                edge_attr_sz_in = edge_attr_in_first * 4
                edge_attr_sz_out = edge_attr_in_first * 8

            self.ml_d.append(get_metalayer(edge_attr_sz_in, edge_attr_sz_out, node_attr_sz_in, node_attr_sz_out))
            self.ml_l.append(get_metalayer(edge_attr_sz_in, edge_attr_sz_out, node_attr_sz_in, node_attr_sz_out))

    def forward(self, data):
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x, edge_attr, _ = self.ml_shared(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        x_d, edge_attr_d = x, edge_attr
        x_l, edge_attr_l = x, edge_attr
        
        for meta_layer in self.ml_d:
            x_d, edge_attr_d, _ = meta_layer(x=x_d, edge_index=edge_index, edge_attr=edge_attr_d, batch=batch)
        for meta_layer in self.ml_l:
            x_l, edge_attr_l, _ = meta_layer(x=x_l, edge_index=edge_index, edge_attr=edge_attr_l, batch=batch)

        x_d = torch.unsqueeze(x_d, 1)
        x_d, _ = scatter_max(x_d, batch, dim=0)
        x_d = torch.squeeze(x_d, 1)
        x_d = F.normalize(x_d, dim=1)

        x_l = torch.unsqueeze(x_l, 1)
        x_l = torch.nn.functional._max_pool1d(x_l, FEATURE_SIZE)
        x_l = torch.squeeze(x_l, 1)
        x_l = x_l.view([len(data.x)//NUMBER_OF_NODES_IN_GRAPH, NUMBER_OF_NODES_IN_GRAPH])
        x_l = self.softmax(x_l)

        return {'probabilities': x_l, 'descriptors': x_d}

# https://github.com/phil-hawkins/OGREnet-v2/blob/ee21e5224fe18dbfe1cf49e86cdbf070687da899/ogrenetv2/model.py

def get_metalayer(edge_attr_sz_in, edge_attr_sz_out, node_attr_sz_in, node_attr_sz_out):
    return MetaLayer(
        EdgeModel(
            edge_attr_sz_in=edge_attr_sz_in, 
            edge_attr_sz_out=edge_attr_sz_out, 
            node_attr_sz=node_attr_sz_in), 
        NodeModel(
            node_attr_sz_in=node_attr_sz_in, 
            node_attr_sz_out=node_attr_sz_out, 
            edge_attr_sz=edge_attr_sz_out), None)

def init_model(path, settings):
    """Build the Graphite network and optionally load a checkpoint.

    Architecture is controlled by ``settings.arch`` (default: ``'netae'``).
    ``multi_gpu=1`` only wraps the network in ``nn.DataParallel``; it no
    longer changes the underlying architecture (which the original code
    silently did, breaking checkpoint compatibility between single- and
    multi-GPU runs).
    """
    arch = getattr(settings, 'arch', 'netae').lower()
    if arch in ('netae', 'ae'):
        model = NetAE()
    elif arch in ('net', 'metalayer'):
        model = Net()
    else:
        raise ValueError("Unknown arch: {!r} (expected 'netae' or 'net')".format(arch))

    if getattr(settings, 'multi_gpu', 0) == 1:
        model = nn.DataParallel(model)

    if path is not None:
        model.load_state_dict(torch.load(path))

    if settings.cuda:
        model.to(torch.cuda.current_device())

    return model





class NetAE(torch.nn.Module):
    def __init__(self, edge_attr_in_first=3, node_attr_in_first=3):

        super(NetAE, self).__init__()
        
        self.TAG1=TAGConv(6,8,1)
        self.TAG2=TAGConv(8,16,2)
        self.TAG3=TAGConv(16,32,3)
        self.FCD1=nn.Linear(32,32)
        self.FCD2=nn.Linear(32,32)

        self.TAG4=TAGConv(32,16,3)
        self.TAG5=TAGConv(16,8,2)
        self.TAG6=TAGConv(8,1,1)

        self.FC1=nn.Linear(8,8)
        self.FC2=nn.Linear(8,1)
        self.FC3=nn.Linear(NUMBER_OF_NODES_IN_GRAPH,1)

        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, data):
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x=self.TAG1(x,edge_index,edge_attr)
        x=self.TAG2(x,edge_index,edge_attr)
        x=self.TAG3(x,edge_index,edge_attr)

        # x=self.TAG3F(x,edge_index)

        x_d = torch.unsqueeze(x, 1)
        x_d, _ = scatter_max(x_d, batch, dim=0)
        x_d=self.FCD1(x_d)
        x_d=self.FCD2(x_d)
        x_d = torch.squeeze(x_d, 1)
        x_d = F.normalize(x_d, dim=1)
        # x=torch.relu(self.TAG4F(x,edge_index))


        x=torch.relu(self.TAG4(x,edge_index,edge_attr))
        x=torch.relu(self.TAG5(x,edge_index,edge_attr))

        conf=torch.relu(self.FC2(torch.relu(self.FC1(x))))
        conf = conf.view([len(data.x)//NUMBER_OF_NODES_IN_GRAPH, NUMBER_OF_NODES_IN_GRAPH])
        conf=torch.relu(self.FC3(conf))
        x=torch.relu(self.TAG6(x,edge_index,edge_attr))


        # x_l = torch.unsqueeze(x, 1)
        # x_l = torch.nn.functional._max_pool1d(x_l, FEATURE_SIZE)
        # x_l = torch.squeeze(x_l, 1)
        x_l = x.view([len(data.x)//NUMBER_OF_NODES_IN_GRAPH, NUMBER_OF_NODES_IN_GRAPH])
        x_l = self.softmax(x_l)
        

        # x_d = torch.unsqueeze(x_d, 1)
        # x_d, _ = scatter_max(x_d, batch, dim=0)
        # x_d = torch.squeeze(x_d, 1)
        # x_d = F.normalize(x_d, dim=1)

        # x_l = torch.unsqueeze(x_l, 1)
        # x_l = torch.nn.functional._max_pool1d(x_l, FEATURE_SIZE)
        # x_l = torch.squeeze(x_l, 1)
        # x_l = x_l.view([len(data.x)//NUMBER_OF_NODES_IN_GRAPH, NUMBER_OF_NODES_IN_GRAPH])
        # x_l = self.softmax(x_l)

        return {'probabilities': x_l, 'descriptors': x_d,'confidence':conf}