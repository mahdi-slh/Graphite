import networkx as nx
from matplotlib import pylab
import matplotlib.pyplot as plt
import random

from graphdataset import GraphDataset
from utils import GRAPH_DATASET


def save_graph(graph,file_name):
    #initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos)
    nx.draw_networkx_edges(graph,pos)
    nx.draw_networkx_labels(graph,pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.savefig(file_name)
    pylab.close()
    del fig

graph_dataset = GraphDataset(root=GRAPH_DATASET)
G = nx.Graph()

idx = random.randint(0, len(graph_dataset))

graph = graph_dataset[idx]

for i in range(graph.edge_index.shape[1]):
    u = graph.edge_index[0][i].item()
    v = graph.edge_index[1][i].item()
    G.add_edge(u, v, weight=graph.edge_attr[i])

color_map = []
for y in graph.y:
    if y < 0.9:
        color_map.append('blue')
    else: color_map.append('red')  

nx.draw(G, node_color=color_map)
save_graph(G, "simple_path_II.png")
nx.draw(G, node_color=color_map, with_labels=True, edge_cmap=graph.edge_attr)
plt.savefig("simple_path_II.png")