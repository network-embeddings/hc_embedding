import networkx as nx
import numpy as np
from itertools import product

def number_of_common_neighbors(g):
    N = g.number_of_nodes()
    weights = []
    for e in g.edges():
        u, v = e[0], e[1]
        w = len(list(nx.common_neighbors(g, u, v)))
        weights.append((u, v, w))
        print(u, v, w)

    _g = g.copy()
    _g.add_weighted_edges_from(weights)

    return nx.to_numpy_array(_g)


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

N = 10
avgk = 3
g = nx.gnp_random_graph(N, avgk/(N - 1))

import matplotlib.pyplot as plt
nx.draw(g, with_labels=True)
plt.show()

print(number_of_common_neighbors(g))
print(nx.to_numpy_array(g))