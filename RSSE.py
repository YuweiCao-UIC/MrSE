import networkx as nx
import numpy as np
import math
import copy
import scipy as sp
from itertools import chain

def pagerank_scipy(G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,):
    """Returns the PageRank of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    alpha : float, optional
      Damping parameter for PageRank, default=0.85.

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key some subset of graph nodes and personalization value each of those.
      At least one personalization value must be non-zero.
      If not specified, a nodes personalization value will be zero.
      By default, a uniform distribution is used.

    max_iter : integer, optional
      Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
      Error tolerance used to check convergence in power method solver.

    nstart : dictionary, optional
      Starting value of PageRank iteration for each node.

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified) This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.

    Returns
    -------
    pagerank : dictionary
       Dictionary of nodes with PageRank as value

    Examples
    --------
    >>> from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_scipy
    >>> G = nx.DiGraph(nx.path_graph(4))
    >>> pr = _pagerank_scipy(G, alpha=0.9)

    Notes
    -----
    The eigenvector calculation uses power iteration with a SciPy
    sparse matrix representation.

    This implementation works with Multi(Di)Graphs. For multigraphs the
    weight between two nodes is set to be the sum of all edge weights
    between those nodes.

    See Also
    --------
    pagerank

    Raises
    ------
    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
       The PageRank citation ranking: Bringing order to the Web. 1999
       http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf
    """

    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]
    # TODO: csr_array
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A # Each row in A sums up to 1

    # initial vector
    if nstart is None:
        x = np.repeat(1.0 / N, N)
    else:
        x = np.array([nstart.get(n, 0) for n in nodelist], dtype=float)
        x /= x.sum()

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()
    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = np.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x @ A + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        if err < N * tol:
            return dict(zip(nodelist, map(float, x)))
    raise nx.PowerIterationFailedConvergence(max_iter)

class RSSE:
  def __init__(self, A):
    '''
    # A = (a_{i2,i1}), where a_{i2,i1} \in {0, 1} for i2, i1 = 1, ..., n. n is the total number of objects, i.e., nodes. 
    # (i2, i1) stands for ((t-1)-th state, t-th state), i.e., (X_{t-1}, X_t).
    # a_{i2,i1} = 1 indicates that there is an edge starts from i2 and points to i1.
    '''
    self.A = A
    self.graph = nx.stochastic_graph(nx.from_numpy_array(A, create_using=nx.DiGraph), weight='weight')
    self.pagerank = pagerank_scipy(self.graph) # {0: p0, 1: p1, ...}
    self.division = {}  # {comm1: [node11, node12, ...], comm2: [node21, node22, ...], ...}
    self.struc_data = {}  # {comm1: [n_to_comm1, p_v1, p_g1, community_node_SE, leaf_nodes_SE, neighbor_comms1], comm2:[n_to_comm2, p_v2, p_g2, community_node_SE, leaf_nodes_SE, neighbor_comms2]，... }
    self.struc_data_2d = {} # {(comm1, comm2): [n_to_comm_after_merge, p_v_after_merge, p_g_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], (comm1, comm3): [], ...}

  def calc_1dSE(self):
    SE = 0
    for n in self.graph.nodes:
      SE += - (self.pagerank[n]) * math.log2(self.pagerank[n])
    return SE

  def update_struc_data(self):
    #[n_to_comm1, p_v1, p_g1, community_node_SE, leaf_nodes_SE, neighbor_comms]
    for vname in self.division.keys():
      comm = self.division[vname]
      outside = self.graph.nodes - comm
      n_to_comm = nx.adjacency_matrix(self.graph).tocsc()[:, comm]
      # n_to_comm shape: n*1. The i-th row is the probability of node i -> a node in comm
      n_to_comm = n_to_comm.tocsr().sum(1)
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

      p_g = 0 # probability of entering comm (from outside), i.e., p(outside -> comm)
      for n in outside:
        p_g += self.pagerank[n] * n_to_comm[n] # p(n -> comm) = p(n) * p(comm|n)
      
      p_v = 0 # probability that 'being in comm' happens
      for n in comm:
        p_v += self.pagerank[n]
      
      vSE = - p_g * math.log2(p_v)

      vnodeSE = 0
      for n in comm:
        vnodeSE += - self.pagerank[n] * math.log2(self.pagerank[n] / p_v)
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
          outside = self.graph.nodes - comm_m
          p_g_m = 0 
          for n in outside:
            p_g_m += self.pagerank[n] * n_to_comm_m[n] # p(n -> comm_m) = p(n) * p(comm_m|n)

          vSE_m = - p_g_m * math.log2(p_v_m)

          vnodeSE_m = 0
          for n in comm_m:
            vnodeSE_m += - self.pagerank[n] * math.log2(self.pagerank[n] / p_v_m)
          
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
        for node in self.division[vm2]:
          self.graph.nodes[node]['comm'] = vm1
        self.division[vm1] += self.division[vm2]
        self.division.pop(vm2)

        n_to_comm = self.struc_data[vm1][0] + self.struc_data[vm2][0]
        p_v = self.struc_data[vm1][1] + self.struc_data[vm2][1]

        comm = self.division[vm1]
        outside = self.graph.nodes - comm
        p_g = 0 
        for n in outside:
          p_g += self.pagerank[n] * n_to_comm[n] # p(n -> comm) = p(n) * p(comm|n)

        neighbor_comms = set(self.struc_data[vm1][5] + self.struc_data[vm2][5])
        neighbor_comms.remove(vm2)
        neighbor_comms = list(neighbor_comms)

        vSE = - p_g * math.log2(p_v)

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
            outside = self.graph.nodes - comm_m
            p_g_m = 0 
            for n in outside:
              p_g_m += self.pagerank[n] * n_to_comm_m[n] # p(n -> comm_m) = p(n) * p(comm_m|n)

            vSE_m = - p_g_m * math.log2(p_v_m)

            vnodeSE_m = self.struc_data[v1][4] - self.struc_data[v1][1] * math.log2(self.struc_data[v1][1] / p_v_m) + \
              self.struc_data[v2][4] - self.struc_data[v2][1] * math.log2(self.struc_data[v2][1] / p_v_m)
            
            struc_data_2d_new[(v1, v2)] = [n_to_comm_m, p_v_m, p_g_m, vSE_m, vnodeSE_m]
          elif k[0] == vm1 or k[1] == vm1:
            v1 = k[0]
            v2 = k[1]

            n_to_comm_m = self.struc_data[v1][0] + self.struc_data[v2][0]
            p_v_m = self.struc_data[v1][1] + self.struc_data[v2][1]

            comm_m = self.division[v1] + self.division[v2]
            outside = self.graph.nodes - comm_m
            p_g_m = 0 
            for n in outside:
              p_g_m += self.pagerank[n] * n_to_comm_m[n] # p(n -> comm_m) = p(n) * p(comm_m|n)

            vSE_m = - p_g_m * math.log2(p_v_m)

            vnodeSE_m = self.struc_data[v1][4] - self.struc_data[v1][1] * math.log2(self.struc_data[v1][1] / p_v_m) + \
              self.struc_data[v2][4] - self.struc_data[v2][1] * math.log2(self.struc_data[v2][1] / p_v_m)
            
            struc_data_2d_new[k] = [n_to_comm_m, p_v_m, p_g_m, vSE_m, vnodeSE_m]
          else:
            struc_data_2d_new[k] = self.struc_data_2d[k]
        self.struc_data_2d = struc_data_2d_new
      else:
        break
   
  def calc_2dSE(self):
    SE = 0
    for comm in self.division.values():
      outside = self.graph.nodes - comm
      n_to_comm = nx.adjacency_matrix(self.graph).tocsc()[:, comm]
      # n_to_comm shape: n*1. The i-th row is the probability of node i -> a node in comm
      n_to_comm = n_to_comm.tocsr().sum(1) 

      p_g = 0 # probability of entering comm (from outside), i.e., p(outside -> comm)
      for n in outside:
        p_g += self.pagerank[n] * n_to_comm[n, 0] # p(n -> comm) = p(n) * p(comm|n)
      
      p_v = 0 # probability that 'being in comm' happens
      for n in comm:
        p_v += self.pagerank[n]
      
      SE += - p_g * math.log2(p_v)
      for n in comm:
        SE += - self.pagerank[n] * math.log2(self.pagerank[n] / p_v)

    return SE

  def init_division(self):
    self.division = {}
    for node in self.graph.nodes:
      new_comm = node
      self.division[new_comm] = [node]
      self.graph.nodes[node]['comm'] = new_comm

def vanilla_2D_RSSE_mini(A, division = None):
  '''
  # A = (a_{i2,i1}), where a_{i2,i1} \in {0, 1} for i2, i1 = 1, ..., n. n is the total number of objects, i.e., nodes. 
  # (i2, i1) stands for ((t-1)-th state, t-th state), i.e., (X_{t-1}, X_t).
  # a_{i2,i1} = 1 indicates that there is an edge starts from i2 and points to i1.
  '''
  seg = RSSE(A)
  RSSE_1d = seg.calc_1dSE()

  if division is None:
    seg.init_division()
  else:
    seg.division = division
  seg.update_struc_data()
  seg.update_struc_data_2d()
  seg.update_division_MinSE()
  comms = seg.division
  
  RSSE_2d = 0
  for vname in seg.division.keys():
    RSSE_2d += seg.struc_data[vname][3]
    RSSE_2d += seg.struc_data[vname][4]
  assert math.isclose(RSSE_2d, seg.calc_2dSE())

  return RSSE_1d, comms, RSSE_2d

def hier_2D_RSSE_mini(A, n = 100):
  '''
  # A = (a_{i2,i1}), where a_{i2,i1} \in {0, 1} for i2, i1 = 1, ..., n. n is the total number of objects, i.e., nodes. 
  # (i2, i1) stands for ((t-1)-th state, t-th state), i.e., (X_{t-1}, X_t).
  # a_{i2,i1} = 1 indicates that there is an edge starts from i2 and points to i1.
  '''
  n_clusters = A.shape[0]

  if n >= n_clusters:
    RSSE_1d, comms, RSSE_2d = vanilla_2D_RSSE_mini(A)
    return RSSE_1d, comms, RSSE_2d

  seg = RSSE(A)
  RSSE_1d = seg.calc_1dSE()

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
      sub_seg = RSSE(sub_A)
      sub_seg.division = split2division(split, sub_comms)
      sub_seg.update_struc_data()
      sub_seg.update_struc_data_2d()
      sub_seg.update_division_MinSE()
      all_comms += division2split(split, sub_seg.division.values())
    if len(all_sub_comms) == 1:
      break
    all_comms.sort()
    if last_all_comms == all_comms:
      n *= 2
    all_sub_comms = [all_comms[i*n: min((i+1)*n, len(all_comms))] for i in range(math.ceil(len(all_comms)/n))]
  
  seg.division = {i:cluster for i, cluster in enumerate(all_comms)}
  seg.update_struc_data()
  seg.update_struc_data_2d()
  RSSE_2d = 0
  for vname in seg.division.keys():
    RSSE_2d += seg.struc_data[vname][3]
    RSSE_2d += seg.struc_data[vname][4]

  return RSSE_1d, seg.division, RSSE_2d

def division2split(split, division_values):
  division2split_map = {d_idx: s_idx for d_idx, s_idx in enumerate(split)}
  return [[division2split_map[d_idx] for d_idx in cluster] for cluster in division_values]

def split2division(split, sub_comms):
  split2division_map = {s_idx: d_idx for d_idx, s_idx in enumerate(split)}
  sub_comms = [[split2division_map[s_idx] for s_idx in cluster] for cluster in sub_comms]
  return {i:cluster for i, cluster in enumerate(sub_comms)}

def test_RSSE():
  '''
  Test RSSE.
  '''
  # This example tensor comes from https://towardsdatascience.com/pagerank-algorithm-fully-explained-dc794184b4af
  # A = (a_{i2,i1}), where a_{i2,i1} \in {0, 1} for i2, i1 = 1, ..., n. n is the total number of objects, i.e., nodes. 
  # (i2, i1) stands for ((t-1)-th state, t-th state), i.e., (X_{t-1}, X_t).
  # a_{i2,i1} = 1 indicates that there is an edge starts from i2 and points to i1.
  A = np.array([[0, 0, 1, 1, 1], [0, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 1, 0, 0, 0], [1, 1, 1, 0, 0]])
  
  seg = RSSE(A)
  print('1D RSSE: ', seg.calc_1dSE())
  seg.init_division()
  print('Initial 2D RSSE: ', seg.calc_2dSE())

  RSSE_1d, comms, RSSE_2d = vanilla_2D_RSSE_mini(A)
  print('Minimized 2D RSSE: ', RSSE_2d)
  print('Detected communities: ', comms)

  return

if __name__ == "__main__":
  test_RSSE()
  