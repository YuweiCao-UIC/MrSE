import numpy as np
import networkx as nx
from MSE import hier_2D_MSE_mini
from RSSE import hier_2D_RSSE_mini
from SE import hier_2D_SE_mini
import math
import random
from random import shuffle
import os

def sparsity_to_BA_num_edges(target_sparsity, num_nodes):
  num_edges = 1
  sparsity = (num_edges*num_nodes-1)/(num_nodes*num_nodes)
  while sparsity < target_sparsity:
    num_edges += 1
    sparsity = (num_edges*num_nodes-1)/(num_nodes*num_nodes)
  return num_edges

def sparcification(G, target_sparsity):
    num_node = G.number_of_nodes()
    edgelist = list(G.edges())
    original_num_edges = len(edgelist)
    original_sparsity = original_num_edges/(num_node*num_node)
    if original_sparsity <= target_sparsity:
        return G
    target_num_edges = math.ceil(target_sparsity*num_node*num_node)
    shuffle(edgelist)
    sparsified_edgelist = edgelist[:target_num_edges]
    sparsified_G = nx.Graph()
    sparsified_G.add_nodes_from([i for i in range(num_node)])
    sparsified_G.add_edges_from(sparsified_edgelist)
    return sparsified_G

def generate_single_relation_BA_graph(num_nodes, target_sparsity):
    num_edges = sparsity_to_BA_num_edges(target_sparsity, num_nodes)
    G = nx.barabasi_albert_graph(num_nodes, num_edges)
    if num_edges == 1:
        G = sparcification(G, target_sparsity)
        A = nx.to_numpy_array(G)
    return A

def generate_multi_relation_BA_graph(num_nodes, target_sparsity, num_relations):
    A = []
    for _ in range(num_relations):
        num_edges = sparsity_to_BA_num_edges(target_sparsity, num_nodes)
        G = nx.barabasi_albert_graph(num_nodes, num_edges)
        if num_edges == 1:
          G = sparcification(G, target_sparsity)
        adj = nx.to_numpy_array(G)
        A.append(adj)
      
    A_2d = np.sum(A, axis = 0)
    A_3d = np.stack(A, axis = 0)
    return A_2d, A_3d

def compare_delta_MSE_RSSE_SE():
    num_nodes = 500
    sparsity = 0.99
    num_relations = 2
    A_2d, A_3d = generate_multi_relation_BA_graph(num_nodes, 1 - sparsity, num_relations)

    MSE_1d, MSE_comms, MSE_2d = hier_2D_MSE_mini(A_3d)
    delta_MSE = 100*(MSE_1d - MSE_2d)/MSE_1d
    print('Delta MSE: %4.2f'%delta_MSE)

    RSSE_1d, RSSE_comms, RSSE_2d = hier_2D_RSSE_mini(A_2d)
    delta_RSSE = 100*(RSSE_1d - RSSE_2d)/RSSE_1d
    print('Delta RSSE: %4.2f'%delta_RSSE)

    SE_1d, SE_comms, SE_2d = hier_2D_SE_mini(A_2d)
    delta_SE = 100*(SE_1d - SE_2d)/SE_1d
    print('Delta SE: %4.2f'%delta_SE)

    return

if __name__ == "__main__":
    compare_delta_MSE_RSSE_SE()