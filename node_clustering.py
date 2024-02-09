import os.path as osp
import torch
import torch.nn as nn
from torch_geometric.datasets import IMDB
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops
from sklearn.metrics.cluster import normalized_mutual_info_score
import sys
from MSE import vanilla_2D_MSE_mini
from RSSE import vanilla_2D_RSSE_mini
from SE import vanilla_2D_SE_mini
from utils import decode
import networkx as nx
import numpy as np

def get_data(dataset = 'IMDB'):
    path = osp.join(osp.dirname(osp.realpath(__file__)), './data/' + dataset)
    metapaths = [[('movie', 'actor'), ('actor', 'movie')], \
        [('movie', 'director'), ('director', 'movie')]]
    transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True, \
        drop_unconnected_node_types = True)
    dataset = IMDB(path, transform=transform)
    data = dataset[0]
    return data

def get_adj(data, num_nodes, key, mode = 'MSE'):
    if mode in ['RSSE', 'SE']:
        homo_data = data.to_homogeneous()
        homo_edge = homo_data.edge_index
        homo_edge, _ = remove_self_loops(homo_edge)
        homo_edge = torch.transpose(homo_edge, 0, 1)
        homo_edge = homo_edge.numpy().tolist()
        g = nx.Graph()
        g.add_nodes_from([i for i in range(num_nodes)])
        g.add_edges_from(homo_edge)
        homo_adj = nx.to_numpy_array(g)
        return homo_adj

    all_A = []
    metapaths = list(data['metapath_dict'].keys())
    for mp in metapaths:
        edge_mp = data[mp].edge_index
        edge_mp, _ = remove_self_loops(edge_mp)
        edge_mp = torch.transpose(edge_mp, 0, 1)
        edge_mp = edge_mp.numpy().tolist()
        g = nx.Graph()
        g.add_nodes_from([i for i in range(num_nodes)])
        g.add_edges_from(edge_mp)
        mp_adj = nx.to_numpy_array(g)
        all_A.append(mp_adj)

    adj = np.stack(all_A, axis=0)
    return adj

def run(mode = 'MSE'):
    key = 'movie'
    data = get_data()
    labels_true = data[key].y.tolist()
    num_nodes = len(labels_true)

    A = get_adj(data = data, num_nodes = num_nodes, key = key, mode = mode)

    if mode == 'MSE':
        MSE_1d, comms, MSE_2d = vanilla_2D_MSE_mini(A)
    elif mode == 'RSSE':
        RSSE_1d, comms, RSSE_2d = vanilla_2D_RSSE_mini(A)
    elif mode == 'SE':
        SE_1d, comms, SE_2d = vanilla_2D_SE_mini(A)

    labels_pred = decode(comms)

    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    print('nmi: ', '%6.4f'%nmi)

    return

if __name__ == "__main__":
    run(mode = 'MSE')
    

