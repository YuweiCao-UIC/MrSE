import networkx as nx
import numpy as np
import math
import copy
from networkx.algorithms import cuts
from itertools import chain

class SE:
  def __init__(self, A):
    '''
    # A = (a_{i2,i1}), where a_{i2,i1} \in {0, 1} for i2, i1 = 1, ..., n. n is the total number of objects, i.e., nodes. 
    # (i2, i1) stands for ((t-1)-th state, t-th state), i.e., (X_{t-1}, X_t).
    # a_{i2,i1} = 1 indicates that there is an edge starts from i2 and points to i1.
    '''
    self.A = A
    self.graph = nx.stochastic_graph(nx.from_numpy_array(A, create_using=nx.DiGraph), weight='weight')
    self.vol = self.get_g_vol()
    self.division = {}  # {comm1: [node11, node12, ...], comm2: [node21, node22, ...], ...}
    self.struc_data = {}  # {comm1: [vol1, cut1, community_node_SE, leaf_nodes_SE, neighbor_comms1], comm2:[vol2, cut2, community_node_SE, leaf_nodes_SE, neighbor_comms2]，... }
    self.struc_data_2d = {} # {(comm1, comm2): [vol_after_merge, cut_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], (comm1, comm3): [], ...}

  def get_g_vol(self):
    return cuts.volume(self.graph, self.graph.nodes, weight = 'weight')
  
  def get_cut(self, comm):
    comm_set = {n for n in comm if n in self.graph}
    # all in_edges to the nodes in comm
    all_in_edges = self.graph.in_edges(nbunch = comm_set, data = 'weight')
    # in_edges from nodes out of comm to the nodes in comm
    cut_in_edges = (e for e in all_in_edges if e[0] not in comm_set)
    return sum(weight for u, v, weight in cut_in_edges)

  def get_volume(self, comm):
    in_degrees = self.graph.in_degree(nbunch = comm, weight = 'weight')
    return sum(weight for node, weight in in_degrees)

  def calc_1dSE(self):
    SE = 0
    for n in self.graph.nodes:
      d = self.graph.in_degree(n, weight = 'weight')
      if d > 0:
        SE += - (d / self.vol) * math.log2(d / self.vol)
    return SE

  def calc_2dSE(self):
    SE = 0
    for comm in self.division.values():
      g = self.get_cut(comm)
      v = self.get_volume(comm)
      SE += - (g / self.vol) * math.log2(v / self.vol)
      for node in comm:
        d = self.graph.in_degree(node, weight = 'weight')
        SE += - (d / self.vol) * math.log2(d / v)
    return SE

  def update_struc_data(self):
    for vname in self.division.keys():
      comm = self.division[vname]
      volume = self.get_volume(comm)
      cut = self.get_cut(comm)
      neighbor_comms = []
      for node in comm:
        # out neighbors:
        for k,v in self.division.items():
          if k != vname and k not in neighbor_comms:
            for end_node in v:
              if self.A[node, end_node] !=  0:
                neighbor_comms.append(k)
                break
        # in neighbors:
        for k,v in self.division.items():
          if k != vname and k not in neighbor_comms:
            for start_node in v:
              if self.A[start_node, node] !=  0:
                neighbor_comms.append(k)
                break
      neighbor_comms = list(set(neighbor_comms))

      if volume == 0:
        vSE = 0
      else:
        vSE = - (cut / self.vol) * math.log2(volume / self.vol)
      vnodeSE = 0
      for node in comm:
        d = self.graph.in_degree(node, weight = 'weight')
        if d != 0:
          vnodeSE -= (d / self.vol) * math.log2(d / volume)
      self.struc_data[vname] = [volume, cut, vSE, vnodeSE, neighbor_comms]
  
  def update_struc_data_2d(self):
    all_comms = list(self.division.keys())
    all_comms.sort()
    for v1 in all_comms:
      neighbor_comms = self.struc_data[v1][4]
      for v2 in neighbor_comms:
        if v1 < v2:
          k = (v1, v2)
          comm_merged = self.division[v1] + self.division[v2]
          gm = self.get_cut(comm_merged)
          vm = self.struc_data[v1][0] + self.struc_data[v2][0]
          if self.struc_data[v1][0] == 0 or self.struc_data[v2][0] == 0:
            vmSE = self.struc_data[v1][2] + self.struc_data[v2][2]
            vmnodeSE = self.struc_data[v1][3] + self.struc_data[v2][3]
          else:
            vmSE = - (gm / self.vol) * math.log2(vm / self.vol)
            vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0]/ self.vol) * math.log2(self.struc_data[v1][0] / vm) + \
              self.struc_data[v2][3] - (self.struc_data[v2][0]/ self.vol) * math.log2(self.struc_data[v2][0] / vm)
          self.struc_data_2d[k] = [vm, gm, vmSE, vmnodeSE]

  def update_division_MinSE(self):
    
    def Mg_operator(v1, v2):
      v1SE = self.struc_data[v1][2] 
      v1nodeSE = self.struc_data[v1][3]

      v2SE = self.struc_data[v2][2]
      v2nodeSE = self.struc_data[v2][3]

      k = (v1, v2)
      vm, gm, vmSE, vmnodeSE = self.struc_data_2d[k]
      delta_SE = vmSE + vmnodeSE - (v1SE + v1nodeSE + v2SE + v2nodeSE)
      return delta_SE
    
    while True:
      delta_SE = 99999
      vm1 = None
      vm2 = None
      all_comms = list(self.division.keys())
      all_comms.sort()
      for v1 in all_comms:
        neighbor_comms = self.struc_data[v1][4]
        for v2 in neighbor_comms:
          if v1 < v2:
            new_delta_SE = Mg_operator(v1, v2)
            if new_delta_SE < delta_SE:
              delta_SE = new_delta_SE
              vm1 = v1
              vm2 = v2

      if delta_SE < 0:
        # change the tree structure: Merge v1 & v2 -> v1
        for node in self.division[vm2]:
          self.graph.nodes[node]['comm'] = vm1
        self.division[vm1] += self.division[vm2]
        self.division.pop(vm2)

        volume = self.struc_data[vm1][0] + self.struc_data[vm2][0]
        cut = self.get_cut(self.division[vm1])
        neighbor_comms = set(self.struc_data[vm1][4] + self.struc_data[vm2][4])
        neighbor_comms.remove(vm2)
        neighbor_comms = list(neighbor_comms)
        vmSE = - (cut / self.vol) * math.log2(volume / self.vol)
        vmnodeSE = self.struc_data[vm1][3] - (self.struc_data[vm1][0]/ self.vol) * math.log2(self.struc_data[vm1][0] / volume) + \
          self.struc_data[vm2][3] - (self.struc_data[vm2][0]/ self.vol) * math.log2(self.struc_data[vm2][0] / volume)
        self.struc_data[vm1] = [volume, cut, vmSE, vmnodeSE, neighbor_comms]

        vm2_neighbors = self.struc_data[vm2][4]
        vm2_neighbors.remove(vm1)
        if vm2 in vm2_neighbors:
          vm2_neighbors.remove(vm2)
        for node in vm2_neighbors:
          node_neighbors = self.struc_data[node][4]
          node_neighbors.remove(vm2)
          node_neighbors.append(vm1)
          self.struc_data[node][4] = list(set(node_neighbors))

        self.struc_data.pop(vm2)

        struc_data_2d_new = {}
        for k in self.struc_data_2d.keys():
          if k[0] == vm2 or k[1] == vm2:
            v = [k[0], k[1], vm1]
            v.remove(vm2)
            v = list(set(v))
            if len(v) < 2:
              continue
            v.sort()
            v1 = v[0]
            v2 = v[1]
            comm_merged = self.division[v1] + self.division[v2]
            gm = self.get_cut(comm_merged)
            vm = self.struc_data[v1][0] + self.struc_data[v2][0]
            if self.struc_data[v1][0] == 0 or self.struc_data[v2][0] == 0:
              vmSE = self.struc_data[v1][2] + self.struc_data[v2][2]
              vmnodeSE = self.struc_data[v1][3] + self.struc_data[v2][3]
            else:
              vmSE = - (gm / self.vol) * math.log2(vm / self.vol)
              vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0]/ self.vol) * math.log2(self.struc_data[v1][0] / vm) + \
                self.struc_data[v2][3] - (self.struc_data[v2][0]/ self.vol) * math.log2(self.struc_data[v2][0] / vm)
            struc_data_2d_new[(v1, v2)] = [vm, gm, vmSE, vmnodeSE]
          elif k[0] == vm1 or k[1] == vm1:
            v1 = k[0]
            v2 = k[1]
            comm_merged = self.division[v1] + self.division[v2]
            gm = self.get_cut(comm_merged)
            vm = self.struc_data[v1][0] + self.struc_data[v2][0]
            if self.struc_data[v1][0] == 0 or self.struc_data[v2][0] == 0:
              vmSE = self.struc_data[v1][2] + self.struc_data[v2][2]
              vmnodeSE = self.struc_data[v1][3] + self.struc_data[v2][3]
            else:
              vmSE = - (gm / self.vol) * math.log2(vm / self.vol)
              vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0]/ self.vol) * math.log2(self.struc_data[v1][0] / vm) + \
                self.struc_data[v2][3] - (self.struc_data[v2][0]/ self.vol) * math.log2(self.struc_data[v2][0] / vm)
            struc_data_2d_new[k] = [vm, gm, vmSE, vmnodeSE]
          else:
            struc_data_2d_new[k] = self.struc_data_2d[k]
        self.struc_data_2d = struc_data_2d_new
      else:
        break
   
  def init_division(self):
    self.division = {}
    for node in self.graph.nodes:
      new_comm = node
      self.division[new_comm] = [node]
      self.graph.nodes[node]['comm'] = new_comm

def vanilla_2D_SE_mini(A, division = None):
  '''
  # A = (a_{i2,i1}), where a_{i2,i1} \in {0, 1} for i2, i1 = 1, ..., n. n is the total number of objects, i.e., nodes. 
  # (i2, i1) stands for ((t-1)-th state, t-th state), i.e., (X_{t-1}, X_t).
  # a_{i2,i1} = 1 indicates that there is an edge starts from i2 and points to i1.
  '''
  seg = SE(A)
  SE_1d = seg.calc_1dSE()

  if division is None:
    seg.init_division()
  else:
    seg.division = division

  seg.update_struc_data()
  seg.update_struc_data_2d()
  seg.update_division_MinSE()
  comms = seg.division

  SE_2d = 0
  for vname in seg.division.keys():
    SE_2d += seg.struc_data[vname][2]
    SE_2d += seg.struc_data[vname][3]
  assert math.isclose(SE_2d, seg.calc_2dSE())

  return SE_1d, comms, SE_2d

def hier_2D_SE_mini(A, n = 100):
  '''
  # A = (a_{i2,i1}), where a_{i2,i1} \in {0, 1} for i2, i1 = 1, ..., n. n is the total number of objects, i.e., nodes. 
  # (i2, i1) stands for ((t-1)-th state, t-th state), i.e., (X_{t-1}, X_t).
  # a_{i2,i1} = 1 indicates that there is an edge starts from i2 and points to i1.
  '''
  n_clusters = A.shape[0]

  if n >= n_clusters:
    SE_1d, comms, SE_2d = vanilla_2D_SE_mini(A)
    return SE_1d, comms, SE_2d

  seg = SE(A)
  SE_1d = seg.calc_1dSE()

  all_comms = [[i] for i in range(n_clusters)]
  all_sub_comms = [all_comms[i*n: min((i+1)*n, len(all_comms))] for i in range(math.ceil(len(all_comms)/n))]
  while True:
    #print('all_comms', all_comms)
    last_all_comms = copy.deepcopy(all_comms)
    all_comms = []
    for sub_comms in all_sub_comms:
      split = list(chain(*sub_comms))
      #print(' split', split)
      sub_A = A[np.ix_(split, split)]
      sub_seg = SE(sub_A)
      sub_seg.division = split2division(split, sub_comms)
      sub_seg.update_struc_data()
      sub_seg.update_struc_data_2d()
      sub_seg.update_division_MinSE()
      temp = division2split(split, sub_seg.division.values())
      temp.sort()
      all_comms += temp
    if len(all_sub_comms) == 1:
      break
    all_comms.sort()
    if last_all_comms == all_comms:
      n *= 2
    all_sub_comms = [all_comms[i*n: min((i+1)*n, len(all_comms))] for i in range(math.ceil(len(all_comms)/n))]
  seg.division = {i:cluster for i, cluster in enumerate(all_comms)}
  seg.update_struc_data()
  seg.update_struc_data_2d()
  SE_2d = 0
  for vname in seg.division.keys():
    SE_2d += seg.struc_data[vname][2]
    SE_2d += seg.struc_data[vname][3]

  return SE_1d, seg.division, SE_2d

def division2split(split, division_values):
  division2split_map = {d_idx: s_idx for d_idx, s_idx in enumerate(split)}
  return [[division2split_map[d_idx] for d_idx in cluster] for cluster in division_values]

def split2division(split, sub_comms):
  split2division_map = {s_idx: d_idx for d_idx, s_idx in enumerate(split)}
  sub_comms = [[split2division_map[s_idx] for s_idx in cluster] for cluster in sub_comms]
  return {i:cluster for i, cluster in enumerate(sub_comms)}

def test_SE():
  '''
  Test SE.
  '''
  # This example tensor comes from https://towardsdatascience.com/pagerank-algorithm-fully-explained-dc794184b4af
  # A = (a_{i2,i1}), where a_{i2,i1} \in {0, 1} for i2, i1 = 1, ..., n. n is the total number of objects, i.e., nodes. 
  # (i2, i1) stands for ((t-1)-th state, t-th state), i.e., (X_{t-1}, X_t).
  # a_{i2,i1} = 1 indicates that there is an edge starts from i2 and points to i1.
  A = np.array([[0, 0, 1, 1, 1], [0, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 1, 0, 0, 0], [1, 1, 1, 0, 0]])
  
  seg = SE(A)
  print('1D SE: ', seg.calc_1dSE())
  seg.init_division()
  print('Initial 2D SE: ', seg.calc_2dSE())

  SE_1d, comms, SE_2d = vanilla_2D_SE_mini(A)
  print('Minimized 2D SE: ', SE_2d)
  print('Detected communities: ', comms)

  return

if __name__ == "__main__":
    test_SE()
  