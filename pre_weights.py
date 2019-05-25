import networkx as nx
import numpy as np
from itertools import combinations

def number_of_common_neighbors(g):
    N = g.number_of_nodes()
    cn = np.zeros((N, N))
    for (u, v), (i, j) in zip(combinations(g.nodes(), 2),
                              combinations(range(N), 2)):
        cn[i, j] = len(list(nx.common_neighbors(g, u, v)))
    return cn + cn.T


def external_degree(g):
    degree = np.array(g.degree())[:, 1]
    N = g.number_of_nodes()
    adj = nx.to_numpy_array(g)
    cn = number_of_common_neighbors(g)
    d = np.repeat(degree.reshape(1, -1), degree.shape[0], axis=0)
    ext_degree = (d - cn - 1) * adj
    return ext_degree.T

def RA1_weights(g):
    degree = np.array(g.degree())[:, 1]
    di, dj = np.meshgrid(degree, degree)
    cn = number_of_common_neighbors(g)
    return (di + dj + di * dj) / (1 + cn)


def RA2_weights(g):
    ei = external_degree(g)
    cn = number_of_common_neighbors(g)
    return (ei + ei.T + ei * ei.T) / (1 + cn)


def EBC_weights(g):
    w = nx.edge_betweenness_centrality(g)
    edges = [(u, v, w[(u,v)]) for u, v in w]
    _g = g.copy()
    _g.add_weighted_edges_from(edges)


    return nx.to_numpy_array(_g)
