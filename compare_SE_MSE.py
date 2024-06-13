import networkx as nx
import numpy as np
import math
import copy
from networkx.algorithms import cuts

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
      #print(g, self.vol, v)
      if v != 0:
        SE += - (g / self.vol) * math.log2(v / self.vol)
      for node in comm:
        d = self.graph.in_degree(node, weight = 'weight')
        #print(d, self.vol, v)
        if d != 0:
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
        
        print('current 2dSE: ', self.calc_2dSE(), 'current communities: ', self.division)
      else:
        break


  def update_division_MinSE_hier(self, n):
    def Mg_operator(v1, v2):
      v1SE = self.struc_data[v1][2] 
      v1nodeSE = self.struc_data[v1][3]

      v2SE = self.struc_data[v2][2]
      v2nodeSE = self.struc_data[v2][3]

      k = (v1, v2)
      vm, gm, vmSE, vmnodeSE = self.struc_data_2d[k]
      delta_SE = vmSE + vmnodeSE - (v1SE + v1nodeSE + v2SE + v2nodeSE)
      return delta_SE
    
    all_comms = list(self.division.keys())
    all_comms.sort()
    comms_splits = [all_comms[s:min(s+n, len(all_comms))] for s in range(0, len(all_comms), n)]
    while True:
      #print('\nnum comms: ', len(all_comms))
      #print('num subgraphs: ', len(comms_splits))
      for i, comms in enumerate(comms_splits):
        #print('processing subgraph ', i+1)
        while True:
          delta_SE = 99999
          vm1 = None
          vm2 = None
          for v1 in comms:
            neighbor_comms = [each for each in self.struc_data[v1][4] if each in comms]
            for v2 in neighbor_comms:
              if v1 < v2:
                new_delta_SE = Mg_operator(v1, v2)
                if new_delta_SE < delta_SE:
                  delta_SE = new_delta_SE
                  vm1 = v1
                  vm2 = v2
          
          if delta_SE < 0:
            # change the tree structure: Merge v1 & v2 -> v1

            comms.remove(vm2)

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
          
      if len(comms_splits) == 1:
        break
      
      last_all_comms = copy.deepcopy(all_comms)
      all_comms = list(self.division.keys())
      all_comms.sort()
      if last_all_comms == all_comms:
        n *= 2
      comms_splits = [all_comms[s:min(s+n, len(all_comms))] for s in range(0, len(all_comms), n)]
    return
      
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

def hier_2D_SE_mini(A, division = None, n = 100):
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
  seg.update_division_MinSE_hier(n = n)
  comms = seg.division

  SE_2d = 0
  for vname in seg.division.keys():
    SE_2d += seg.struc_data[vname][2]
    SE_2d += seg.struc_data[vname][3]
  assert math.isclose(SE_2d, seg.calc_2dSE())

  return SE_1d, comms, SE_2d

def multirank(A,
  alpha=0.85,
  max_iter=100,
  tol=1.0e-6,
  weight="weight",):
  """
  A = (a_{j1,i1,i2}), where a_{j1,i1,i2} \in R for 
    ik = 1, ..., m, 
    k = 1, 2, 
    and j1 = 1, ..., n.
    m is the total number of objects, i.e., nodes.  
    n is the total number of relations.
    (i1, i2) stands for (t-th state, (t-1)-th state), i.e., (X_t, X_{t-1}).
    j1 stands for the index of the relation.
    Shape of A: n * m * m.
  """
  # Shape of O: n * m * m (same as A).
  # o_{j1,i1,i2} = P(X_t = i1|X_{t-1} = i2, Y_t = j1)
  S_O = np.sum(A, axis = 1, keepdims = True) # sum up over i1, the (t-1)-th state.
  #print('S_O:\n', S_O)
  Q_O = np.copy(S_O)
  Q_O[Q_O == 0] += 1 # prevent dividing by zero
  #print('Q_O:\n', Q_O)
  O = np.divide(A, Q_O)
  #print('O:\n', O)
  S_O = np.squeeze(S_O, axis=1)
  #print('S_O:\n', S_O)

  # Shape of R: n * m * m (same as A).
  # r_{j1,i1,i2} = P(Y_t = j1|X_t = i1, X_{t-1} = i2)
  S_R = np.sum(A, axis = 0, keepdims = True)
  #print('S_R:\n', S_R)
  Q_R = np.copy(S_R)
  Q_R[Q_R == 0] += 1
  #print('Q_R:\n', Q_R)
  R = np.divide(A, Q_R)
  #print('R:\n', R)
  S_R = np.squeeze(S_R, axis=0)
  #print('S_R:\n', S_R)
 
  # Initialize the multirank vectors.
  n = A.shape[0] # the total number of relations
  m = A.shape[1] # the total number of objects, i.e., nodes
  #print(n, ' relations and ', m, ' objects.')
  # x is a 1d array of shape (m,). x contains the multirank values of all objects.
  # y is a 1d array of shape (n,). y contains the multirank values of all relations.
  x = np.repeat(1.0 / m, m)
  y = np.repeat(1.0 / n, n)
  
  # Set up personalization vector (introduced for primitivity adjustment) and 
  # dangling weights (introduced for stochasticity adjustment).
  # Note that the original MultiRank paper assumes irreducibility (definition 1 in https://hkumath.hku.hk/~mng/mng_files/report2.pdf),
  # which not necessarily holds. We therefore applies primitivity adjustment using p_x.
  # Stochasticity Adjustment and primitivity adjustment are explained in: 
  # https://moz.com/blog/training-the-random-surfer-two-important-adjustments-to-the-early-pagerank-model
  p_x = np.repeat(1.0 / m, m)
  dangling_weights_x = p_x
  dangling_weights_y =  np.repeat(1.0 / n, n)

  # (S_O == 0) shape: n * m. Each row indicates which of the current objects (i2) never go into the next objects (i1).
  # O_sto_adjust: stochasticity adjustment term. Shape: n * m * m. Adding O_sto_adjust to O makes sure 
  # that for any (j1, i2), the probabilities of reaching different i1 = 1, ..., m sum up to 1.
  #print((S_O == 0))
  #print(np.expand_dims((S_O == 0), axis = 1))
  #print(np.expand_dims(dangling_weights_x, axis = 0).T)
  O_sto_adjust = np.expand_dims((S_O == 0), axis = 1) * np.expand_dims(dangling_weights_x, axis = 0).T
  #print('O_sto_adjust:\n', O_sto_adjust)
  O = O + O_sto_adjust
  #print('O:\n', O)
  #assert (np.sum(O, axis = 1) == 1).all()
  assert np.isclose(np.sum(O, axis = 1), 1).all()

  # (S_R == 0) shape: m * m. The (i1, i2)-th element in (S_R == 0) equals to 1 indicating that
  # i2 (the current object) can't reach i1 (the next object) under any relation.
  # R_sto_adjust: stochasticity adjustment term. Shape: n * m * m. Adding R_sto_adjust to R makes sure 
  # that for any (i1, i2), the probability of reaching i1 from i2 under all relations sum up to 1.
  #print(np.repeat(np.expand_dims((S_R == 0), axis = 0), n, axis = 0))
  R_sto_adjust = np.repeat(np.expand_dims((S_R == 0), axis = 0), n, axis = 0) * np.expand_dims(dangling_weights_y, axis = (1, 2))
  #print('R_sto_adjust:\n', R_sto_adjust)
  R = R + R_sto_adjust
  #print('R:\n', R)
  #assert (np.sum(R, axis = 0) == 1).all()
  assert np.isclose(np.sum(R, axis = 0), 1).all()

  # power iteration: make up to max_iter iterations
  for iteration in range(max_iter):
    #print('\nIteration: ', iteration)
    xlast = x
    ylast = y

    # =====================  Calculate new multirank values of the objects  =======================
    x = alpha * O @ x + (1 - alpha) * p_x
    #print(x)

    # Sum up the relations to get the final, new multirank values of the objects, 
    # assuming that the multirank value of each relation is none-zero.
    x = y @ x 
    #print(x)
    #print(sum(x))
    if not math.isclose(sum(x), 1):
      x = x / np.linalg.norm(x, ord=1)
    assert math.isclose(sum(x), 1)

    # =====================  Calculate new multirank values of the relations  =======================
    # Sum up all (i1, i2) to get the final, new multirank values of the relations. 
    y = R @ x @ x
    #print(y)
    if not math.isclose(sum(y), 1):
      y = y / np.linalg.norm(y, ord=1)
    assert math.isclose(sum(y), 1)

    # check convergence, l1 norm
    err = np.absolute(x - xlast).sum() + np.absolute(y - ylast).sum()
    if err < (m + n) * tol:
      object_list = [i for i in range(m)]
      relation_list = [i for i in range(n)]
      return dict(zip(object_list, map(float, x))), dict(zip(relation_list, map(float, y))), O
  
  raise nx.PowerIterationFailedConvergence(max_iter)

class MSE:
  def __init__(self, A):
    """
    A = (a_{j1,i1,i2}), where a_{j1,i1,i2} \in R for 
      ik = 1, ..., m, 
      k = 1, 2, 
      and j1 = 1, ..., n.
      m is the total number of objects, i.e., nodes.  
      n is the total number of relations.
      (i1, i2) stands for (t-th state, (t-1)-th state), i.e., (X_t, X_{t-1}).
      j1 stands for the index of the relation.
      Shape of A: n * m * m.
    """
    self.A = A
    # o_{j1,i1,i2} = P(X_t = i1|X_{t-1} = i2, Y_t = j1)
    self.multirank_object, self.multirank_relation, self.O = multirank(A)
    print('self.O: ', self.O)
    print('self.multirank_object: ', self.multirank_object)
    print('self.multirank_relation: ', self.multirank_relation)
    self.multirank_object_values = np.array(list(self.multirank_object.values()))
    self.graph_nodes = list(self.multirank_object.keys())
    self.graph_nodes.sort()
    self.division = {}  # {comm1: [node11, node12, ...], comm2: [node21, node22, ...], ...}
    self.struc_data = {}  # {comm1: [n_to_comm1, p_v1, p_g1, community_node_SE, leaf_nodes_SE, neighbor_comms1], comm2:[n_to_comm2, p_v2, p_g2, community_node_SE, leaf_nodes_SE, neighbor_comms2]，... }
    self.struc_data_2d = {} # {(comm1, comm2): [n_to_comm_after_merge, p_v_after_merge, p_g_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], (comm1, comm3): [], ...}
    # self.trans_matrix: transition matrix. Shape: m * m, where m is the total number of objects, i.e., nodes.
    # self.trans_matrix_{i1,i2} = P(X_t = i1|X_{t-1} = i2)
    #                           = \sum_j1(P(X_t = i1|X_{t-1} = i2, Y_t = j1) * P(Y_t = j1))
    self.trans_matrix = np.sum(self.O * np.expand_dims(np.array(list(self.multirank_relation.values())), axis = (1, 2)), axis = 0)
    #print('self.trans_matrix:\n', self.trans_matrix)
    assert np.allclose(np.sum(self.trans_matrix, axis = 0), 1)
    
  def calc_1dSE(self):
    SE = - sum(self.multirank_object_values * np.log2(self.multirank_object_values))
    return SE

  def update_struc_data(self):
    #[n_to_comm1, p_v1, p_g1, community_node_SE, leaf_nodes_SE, neighbor_comms]
    for vname in self.division.keys():
      comm = self.division[vname]
      outside = [n for n in self.graph_nodes if n not in comm]
      # n_to_comm is a 1d array of shape (m,), where m is the total number of objects, i.e., nodes.
      n_to_comm = np.sum(self.trans_matrix[comm, :], axis = 0)
      mapped_A = np.sum(self.A, axis=0)
      neighbor_comms = []
      for node in comm:
        # out neighbors:
        for k,v in self.division.items():
          if k != vname and k not in neighbor_comms:
            for end_node in v:
              if mapped_A[end_node, node] !=  0:
                neighbor_comms.append(k)
                break
        # in neighbors:
        for k,v in self.division.items():
          if k != vname and k not in neighbor_comms:
            for start_node in v:
              if mapped_A[node, start_node] !=  0:
                neighbor_comms.append(k)
                break
      neighbor_comms = list(set(neighbor_comms))
      
      # p_g: probability of entering comm (from outside), i.e., p(outside -> comm)
      p_g = sum((self.multirank_object_values * n_to_comm)[outside])
      # p_v: probability that 'being in comm' happens
      p_v = sum(self.multirank_object_values[comm])
      # vSE, SE of the current community node
      vSE = - p_g * math.log2(p_v)
      # vnodeSE, the sum of the SEs of the singleton nodes whose parent is the current community
      vnodeSE = - sum((self.multirank_object_values * np.log2(self.multirank_object_values / p_v))[comm])
      self.struc_data[vname] = [n_to_comm, p_v, p_g, vSE, vnodeSE, neighbor_comms]
  
  def update_struc_data_2d(self):
    # [n_to_comm_after_merge, p_v_after_merge, p_g_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge]
    all_comms = list(self.division.keys())
    all_comms.sort()
    for v1 in all_comms:
      neighbor_comms = self.struc_data[v1][5]
      for v2 in neighbor_comms:
        if v1 < v2:
          k = (v1, v2)
          n_to_comm_m = self.struc_data[v1][0] + self.struc_data[v2][0]
          p_v_m = self.struc_data[v1][1] + self.struc_data[v2][1]

          comm_m = self.division[v1] + self.division[v2]
          outside = [n for n in self.graph_nodes if n not in comm_m]
          p_g_m = sum((self.multirank_object_values * n_to_comm_m)[outside])

          vSE_m = - p_g_m * math.log2(p_v_m)

          # the two equations below are equivalent. The second one saves some calculations.
          #vnodeSE_m = - sum((self.multirank_object_values * np.log2(self.multirank_object_values / p_v_m))[comm_m])
          vnodeSE_m = self.struc_data[v1][4] - self.struc_data[v1][1] * math.log2(self.struc_data[v1][1] / p_v_m) + \
            self.struc_data[v2][4] - self.struc_data[v2][1] * math.log2(self.struc_data[v2][1] / p_v_m)

          self.struc_data_2d[k] = [n_to_comm_m, p_v_m, p_g_m, vSE_m, vnodeSE_m]

  def update_division_MinSE(self):
    
    def Mg_operator(v1, v2):
      v1SE = self.struc_data[v1][3] 
      v1nodeSE = self.struc_data[v1][4]

      v2SE = self.struc_data[v2][3]
      v2nodeSE = self.struc_data[v2][4]

      k = (v1, v2)
      n_to_comm_m, p_v_m, p_g_m, vSE_m, vnodeSE_m = self.struc_data_2d[k]
      delta_SE = vSE_m + vnodeSE_m - (v1SE + v1nodeSE + v2SE + v2nodeSE)
      return delta_SE
    
    while True:
      delta_SE = 99999
      vm1 = None
      vm2 = None
      all_comms = list(self.division.keys())
      all_comms.sort()
      for v1 in all_comms:
        neighbor_comms = self.struc_data[v1][5]
        for v2 in neighbor_comms:
          if v1 < v2:
            new_delta_SE = Mg_operator(v1, v2)
            if new_delta_SE < delta_SE:
              delta_SE = new_delta_SE
              vm1 = v1
              vm2 = v2

      if delta_SE < 0:
        # change the tree structure: Merge v1 & v2 -> v1
        self.division[vm1] += self.division[vm2]
        self.division.pop(vm2)

        n_to_comm = self.struc_data[vm1][0] + self.struc_data[vm2][0]
        p_v = self.struc_data[vm1][1] + self.struc_data[vm2][1]

        comm = self.division[vm1]
        outside = [n for n in self.graph_nodes if n not in comm]
        p_g = sum((self.multirank_object_values * n_to_comm)[outside])

        neighbor_comms = set(self.struc_data[vm1][5] + self.struc_data[vm2][5])
        neighbor_comms.remove(vm2)
        neighbor_comms = list(neighbor_comms)

        vSE = - p_g * math.log2(p_v)

        # the two equations below are equivalent. The second one saves some calculations.
        #vnodeSE = - sum((self.multirank_object_values * np.log2(self.multirank_object_values / p_v))[comm])
        vnodeSE = self.struc_data[vm1][4] - self.struc_data[vm1][1] * math.log2(self.struc_data[vm1][1] / p_v) + \
          self.struc_data[vm2][4] - self.struc_data[vm2][1] * math.log2(self.struc_data[vm2][1] / p_v)
        self.struc_data[vm1] = [n_to_comm, p_v, p_g, vSE, vnodeSE, neighbor_comms]

        vm2_neighbors = self.struc_data[vm2][5]
        vm2_neighbors.remove(vm1)
        if vm2 in vm2_neighbors:
          vm2_neighbors.remove(vm2)
        for node in vm2_neighbors:
          node_neighbors = self.struc_data[node][5]
          node_neighbors.remove(vm2)
          node_neighbors.append(vm1)
          self.struc_data[node][5] = list(set(node_neighbors))

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

            n_to_comm_m = self.struc_data[v1][0] + self.struc_data[v2][0]
            p_v_m = self.struc_data[v1][1] + self.struc_data[v2][1]

            comm_m = self.division[v1] + self.division[v2]
            outside = [n for n in self.graph_nodes if n not in comm_m]
            p_g_m = sum((self.multirank_object_values * n_to_comm_m)[outside])

            vSE_m = - p_g_m * math.log2(p_v_m)

            # the two equations below are equivalent. The second one saves some calculations.
            #vnodeSE_m = - sum((self.multirank_object_values * np.log2(self.multirank_object_values / p_v_m))[comm_m])
            vnodeSE_m = self.struc_data[v1][4] - self.struc_data[v1][1] * math.log2(self.struc_data[v1][1] / p_v_m) + \
              self.struc_data[v2][4] - self.struc_data[v2][1] * math.log2(self.struc_data[v2][1] / p_v_m)
            
            struc_data_2d_new[(v1, v2)] = [n_to_comm_m, p_v_m, p_g_m, vSE_m, vnodeSE_m]
          elif k[0] == vm1 or k[1] == vm1:
            v1 = k[0]
            v2 = k[1]

            n_to_comm_m = self.struc_data[v1][0] + self.struc_data[v2][0]
            p_v_m = self.struc_data[v1][1] + self.struc_data[v2][1]

            comm_m = self.division[v1] + self.division[v2]
            outside = [n for n in self.graph_nodes if n not in comm_m]
            p_g_m = sum((self.multirank_object_values * n_to_comm_m)[outside])

            vSE_m = - p_g_m * math.log2(p_v_m)

            # the two equations below are equivalent. The second one saves some calculations.
            #vnodeSE_m = - sum((self.multirank_object_values * np.log2(self.multirank_object_values / p_v_m))[comm_m])
            vnodeSE_m = self.struc_data[v1][4] - self.struc_data[v1][1] * math.log2(self.struc_data[v1][1] / p_v_m) + \
              self.struc_data[v2][4] - self.struc_data[v2][1] * math.log2(self.struc_data[v2][1] / p_v_m)
            
            struc_data_2d_new[k] = [n_to_comm_m, p_v_m, p_g_m, vSE_m, vnodeSE_m]
          else:
            struc_data_2d_new[k] = self.struc_data_2d[k]
        self.struc_data_2d = struc_data_2d_new

        print('current 2dSE: ', self.calc_2dSE(), 'current communities: ', self.division)

      else:
        break

  def update_division_MinSE_hier(self, n):
    def Mg_operator(v1, v2):
      v1SE = self.struc_data[v1][3] 
      v1nodeSE = self.struc_data[v1][4]

      v2SE = self.struc_data[v2][3]
      v2nodeSE = self.struc_data[v2][4]

      k = (v1, v2)
      n_to_comm_m, p_v_m, p_g_m, vSE_m, vnodeSE_m = self.struc_data_2d[k]
      delta_SE = vSE_m + vnodeSE_m - (v1SE + v1nodeSE + v2SE + v2nodeSE)
      return delta_SE
    
    all_comms = list(self.division.keys())
    all_comms.sort()
    comms_splits = [all_comms[s:min(s+n, len(all_comms))] for s in range(0, len(all_comms), n)]
    while True:
      #print('\nnum comms: ', len(all_comms))
      #print('num subgraphs: ', len(comms_splits))
      for i, comms in enumerate(comms_splits):
        #print('processing subgraph ', i+1)
        while True:
          delta_SE = 99999
          vm1 = None
          vm2 = None
          for v1 in comms:
            neighbor_comms = [each for each in self.struc_data[v1][5] if each in comms]
            for v2 in neighbor_comms:
              if v1 < v2:
                new_delta_SE = Mg_operator(v1, v2)
                if new_delta_SE < delta_SE:
                  delta_SE = new_delta_SE
                  vm1 = v1
                  vm2 = v2
          
          if delta_SE < 0:
            # change the tree structure: Merge v1 & v2 -> v1
            comms.remove(vm2)

            self.division[vm1] += self.division[vm2]
            self.division.pop(vm2)

            n_to_comm = self.struc_data[vm1][0] + self.struc_data[vm2][0]
            p_v = self.struc_data[vm1][1] + self.struc_data[vm2][1]

            comm = self.division[vm1]
            outside = [n for n in self.graph_nodes if n not in comm]
            p_g = sum((self.multirank_object_values * n_to_comm)[outside])

            neighbor_comms = set(self.struc_data[vm1][5] + self.struc_data[vm2][5])
            neighbor_comms.remove(vm2)
            neighbor_comms = list(neighbor_comms)

            vSE = - p_g * math.log2(p_v)

            # the two equations below are equivalent. The second one saves some calculations.
            #vnodeSE = - sum((self.multirank_object_values * np.log2(self.multirank_object_values / p_v))[comm])
            vnodeSE = self.struc_data[vm1][4] - self.struc_data[vm1][1] * math.log2(self.struc_data[vm1][1] / p_v) + \
              self.struc_data[vm2][4] - self.struc_data[vm2][1] * math.log2(self.struc_data[vm2][1] / p_v)
            self.struc_data[vm1] = [n_to_comm, p_v, p_g, vSE, vnodeSE, neighbor_comms]

            vm2_neighbors = self.struc_data[vm2][5]
            vm2_neighbors.remove(vm1)
            if vm2 in vm2_neighbors:
              vm2_neighbors.remove(vm2)
            for node in vm2_neighbors:
              node_neighbors = self.struc_data[node][5]
              node_neighbors.remove(vm2)
              node_neighbors.append(vm1)
              self.struc_data[node][5] = list(set(node_neighbors))

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

                n_to_comm_m = self.struc_data[v1][0] + self.struc_data[v2][0]
                p_v_m = self.struc_data[v1][1] + self.struc_data[v2][1]

                comm_m = self.division[v1] + self.division[v2]
                outside = [n for n in self.graph_nodes if n not in comm_m]
                p_g_m = sum((self.multirank_object_values * n_to_comm_m)[outside])

                vSE_m = - p_g_m * math.log2(p_v_m)

                # the two equations below are equivalent. The second one saves some calculations.
                #vnodeSE_m = - sum((self.multirank_object_values * np.log2(self.multirank_object_values / p_v_m))[comm_m])
                vnodeSE_m = self.struc_data[v1][4] - self.struc_data[v1][1] * math.log2(self.struc_data[v1][1] / p_v_m) + \
                  self.struc_data[v2][4] - self.struc_data[v2][1] * math.log2(self.struc_data[v2][1] / p_v_m)
                
                struc_data_2d_new[(v1, v2)] = [n_to_comm_m, p_v_m, p_g_m, vSE_m, vnodeSE_m]
              elif k[0] == vm1 or k[1] == vm1:
                v1 = k[0]
                v2 = k[1]

                n_to_comm_m = self.struc_data[v1][0] + self.struc_data[v2][0]
                p_v_m = self.struc_data[v1][1] + self.struc_data[v2][1]

                comm_m = self.division[v1] + self.division[v2]
                outside = [n for n in self.graph_nodes if n not in comm_m]
                p_g_m = sum((self.multirank_object_values * n_to_comm_m)[outside])

                vSE_m = - p_g_m * math.log2(p_v_m)

                # the two equations below are equivalent. The second one saves some calculations.
                #vnodeSE_m = - sum((self.multirank_object_values * np.log2(self.multirank_object_values / p_v_m))[comm_m])
                vnodeSE_m = self.struc_data[v1][4] - self.struc_data[v1][1] * math.log2(self.struc_data[v1][1] / p_v_m) + \
                  self.struc_data[v2][4] - self.struc_data[v2][1] * math.log2(self.struc_data[v2][1] / p_v_m)
                
                struc_data_2d_new[k] = [n_to_comm_m, p_v_m, p_g_m, vSE_m, vnodeSE_m]
              else:
                struc_data_2d_new[k] = self.struc_data_2d[k]
            self.struc_data_2d = struc_data_2d_new
          else:
            break
          
      if len(comms_splits) == 1:
        break
      
      last_all_comms = copy.deepcopy(all_comms)
      all_comms = list(self.division.keys())
      all_comms.sort()
      if last_all_comms == all_comms:
        n *= 2
      comms_splits = [all_comms[s:min(s+n, len(all_comms))] for s in range(0, len(all_comms), n)]

    return
      
  def calc_2dSE(self):
    SE = 0
    for comm in self.division.values():
      outside = [n for n in self.graph_nodes if n not in comm]
      # n_to_comm is a 1d array of shape (m,), where m is the total number of objects, i.e., nodes.
      n_to_comm = np.sum(self.trans_matrix[comm, :], axis = 0)
      # p_g: probability of entering comm (from outside), i.e., p(outside -> comm)
      p_g = sum((self.multirank_object_values * n_to_comm)[outside])
      # p_v: probability that 'being in comm' happens
      p_v = sum(self.multirank_object_values[comm])
      # vSE, SE of the current community node
      vSE = - p_g * math.log2(p_v)
      SE += vSE
      # vnodeSE, the sum of the SEs of the singleton nodes whose parent is the current community
      vnodeSE = - sum((self.multirank_object_values * np.log2(self.multirank_object_values / p_v))[comm])
      SE += vnodeSE
    return SE

  def init_division(self):
    self.division = {}
    for node in self.graph_nodes:
      new_comm = node
      self.division[new_comm] = [node]
 
def vanilla_2D_MSE_mini(A, division = None):
  seg = MSE(A)
  MSE_1d = seg.calc_1dSE()  

  if division is None:
    seg.init_division()
  else:
    seg.division = division

  seg.update_struc_data()
  seg.update_struc_data_2d()
  seg.update_division_MinSE()
  comms = seg.division

  MSE_2d = 0
  for vname in seg.division.keys():
    MSE_2d += seg.struc_data[vname][3]
    MSE_2d += seg.struc_data[vname][4]
  assert math.isclose(MSE_2d, seg.calc_2dSE())

  return MSE_1d, comms, MSE_2d

def hier_2D_MSE_mini(A, n = 100):
  n_relations, n_clusters = A.shape[0], A.shape[1]

  if n >= n_clusters:
    MSE_1d, comms, MSE_2d = vanilla_2D_MSE_mini(A)
    return MSE_1d, comms, MSE_2d

  seg = MSE(A)
  MSE_1d = seg.calc_1dSE()

  all_comms = [[i] for i in range(n_clusters)]
  all_sub_comms = [all_comms[i*n: min((i+1)*n, len(all_comms))] for i in range(math.ceil(len(all_comms)/n))]
  while True:
    #print('all_comms', all_comms)
    last_all_comms = copy.deepcopy(all_comms)
    all_comms = []
    for sub_comms in all_sub_comms:
      split = list(chain(*sub_comms))
      #print(' split', split)
      sub_A = A[np.ix_([i for i in range(n_relations)], split, split)]
      sub_seg = MSE(sub_A)
      sub_seg.division = split2division_0504(split, sub_comms)
      sub_seg.update_struc_data()
      sub_seg.update_struc_data_2d()
      sub_seg.update_division_MinSE()
      all_comms += division2split_0504(split, sub_seg.division.values())
    if len(all_sub_comms) == 1:
      break
    all_comms.sort()
    if last_all_comms == all_comms:
      n *= 2
    all_sub_comms = [all_comms[i*n: min((i+1)*n, len(all_comms))] for i in range(math.ceil(len(all_comms)/n))]
  
  seg.division = {i:cluster for i, cluster in enumerate(all_comms)}
  seg.update_struc_data()
  seg.update_struc_data_2d()
  MSE_2d = 0
  for vname in seg.division.keys():
    MSE_2d += seg.struc_data[vname][3]
    MSE_2d += seg.struc_data[vname][4]

  return MSE_1d, seg.division, MSE_2d

def test_MSE():
  '''
  Test MSE.
  '''
  # This example tensor comes from Figure 1 of the MultiRank paper (https://hkumath.hku.hk/~mng/mng_files/report2.pdf).
  # 0th dimension: relation, 1st dimension: end node, 2nd dimension: start node
  A = np.array(
    [[[0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0],
      [0, 1, 0, 0, 1],
      [0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0]],
     [[0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0],
      [0, 0, 0, 1, 0],
      [0, 0, 0, 0, 1],
      [1, 0, 0, 0, 0]],
     [[0, 1, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [1, 0, 0, 1, 0],
      [0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0]]])

  seg = MSE(A)
  print('1D MSE: ', seg.calc_1dSE())
  seg.init_division()
  print('Initial 2D MSE: ', seg.calc_2dSE())

  MSE_1d, comms, MSE_2d = vanilla_2D_MSE_mini(A)
  print('Minimized 2D MSE: ', MSE_2d)
  print('Detected communities: ', comms)
  
  return

def test_SE_1():
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

def test_SE_2():
  A = np.array(
    [[0, 1, 1, 0, 1],
      [1, 0, 1, 0, 0],
      [1, 1, 0, 1, 1],
      [0, 0, 1, 0, 1],
      [1, 0, 1, 1, 0]])
  seg = SE(A)
  print('1D SE: ', seg.calc_1dSE())
  seg.init_division()
  print('Initial 2D SE: ', seg.calc_2dSE())

  SE_1d, comms, SE_2d = vanilla_2D_SE_mini(A)
  print('Minimized 2D SE: ', SE_2d)
  print('Detected communities: ', comms)

def test_SE_3():
  A = np.array(
     [[0, 1, 0, 0, 0],
      [1, 0, 1, 0, 0],
      [1, 1, 0, 1, 1],
      [0, 0, 0, 0, 1],
      [1, 0, 0, 0, 0]])
  seg = SE(A)
  print('1D SE: ', seg.calc_1dSE())
  seg.init_division()
  print('Initial 2D SE: ', seg.calc_2dSE())

  SE_1d, comms, SE_2d = vanilla_2D_SE_mini(A)
  print('Minimized 2D SE: ', SE_2d)
  print('Detected communities: ', comms)

def test_SE_r1():
  A = np.array(
     [[0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0],
      [0, 1, 0, 0, 1],
      [0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0]])
  seg = SE(A)
  print('1D SE: ', seg.calc_1dSE())
  seg.init_division()
  print('Initial 2D SE: ', seg.calc_2dSE())

  SE_1d, comms, SE_2d = vanilla_2D_SE_mini(A)
  print('Minimized 2D SE: ', SE_2d)
  print('Detected communities: ', comms)

def test_SE_r2():
  A = np.array(
     [[0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0],
      [0, 0, 0, 1, 0],
      [0, 0, 0, 0, 1],
      [1, 0, 0, 0, 0]])
  seg = SE(A)
  print('1D SE: ', seg.calc_1dSE())
  seg.init_division()
  print('Initial 2D SE: ', seg.calc_2dSE())

  SE_1d, comms, SE_2d = vanilla_2D_SE_mini(A)
  print('Minimized 2D SE: ', SE_2d)
  print('Detected communities: ', comms)

def test_SE_r3():
  A = np.array(
     [[0, 1, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [1, 0, 0, 1, 0],
      [0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0]])
  seg = SE(A)
  print('1D SE: ', seg.calc_1dSE())
  seg.init_division()
  print('Initial 2D SE: ', seg.calc_2dSE())

  SE_1d, comms, SE_2d = vanilla_2D_SE_mini(A)
  print('Minimized 2D SE: ', SE_2d)
  print('Detected communities: ', comms)

if __name__ == "__main__":
    #test_MSE()
    #test_SE_1()
    #test_SE_2()
    #test_SE_3()
    #test_SE_r1()
    #test_SE_r2()
    test_SE_r3()