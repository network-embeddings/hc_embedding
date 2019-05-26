import networkx as nx
import numpy as np
from itertools import combinations


def number_of_common_neighbors(g):
    """
    Calculates the number of neighbors shared by each pair of nodes in a graph.

    Args:
        g: networkx.Graph

    Returns: numpy.ndarray
        (N, N) array, each value is the number of common neighbors between nodes i and j.
    """
    N = g.number_of_nodes()
    cn = np.zeros((N, N))
    for (u, v), (i, j) in zip(combinations(g.nodes(), 2),
                              combinations(range(N), 2)):
        cn[i, j] = len(list(nx.common_neighbors(g, u, v)))
    return cn + cn.T


def external_degree(g):
    """
    Calculates the external degree of a node with respect to each node in the graph.

    An edge counts towards the external degree of node i w.r.t. node j if:
        - The edge is not connected to node j
        - The edge is not connected to a common neighbor of i and j

    Args:
        g: networkx.Graph

    Returns: numpy.ndarray
        (N, N) array, each value is the external degree of node i w.r.t. node j.
    """
    degree = np.array(g.degree())[:, 1]
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
