import networkx as nx
import numpy as np

def number_of_common_neighbors(g):
    N = g.number_of_nodes()
    cn = np.zeros((N, N))
    for i, u in enumerate(g.nodes()):
        for j, v in enumerate(g.nodes()):
            cn[i, j] = len(list(nx.common_neighbors(g, u, v)))
    return cn


def external_degree(g):
    degree = np.array(g.degree())[:, 1]
    N = g.number_of_nodes()
    adj = nx.to_numpy_array(g)
    cn = number_of_common_neighbors(g)
    d = np.repeat(degree.reshape(1, -1), degree.shape[0], axis=0)
    ext_degree = (d - cn - 1) * adj
    return ext_degree.T

def norm_angles(coords):
    N = coords.shape[0]

    angles = np.arctan(coords[:, 1], coords[:, 0])
    angles[(coords[:, 0] > 0) * (coords[:, 1] > 0)] += np.pi
    angles[(coords[:, 0] > 0) * (coords[:, 1] < 0)] -= np.pi
    return angles

def RA1_weights(g):
    degree = np.array(g.degree())[:, 1]
    di, dj = np.meshgrid(degree, degree)
    cn = number_of_common_neighbors(g)
    return (di + dj + di * dj) / (1 + cn)


def RA2_weights(g):
    ei = external_degree(g)
    return (ei + ei.T + ei * ei.T) / (1 + cn)


def EBC_weights(g):
    N = g.number_of_nodes()
    weights = np.zeros((N, N))

    for u in g.nodes():
        for v in g.neighbors(u):
            sp = list(nx.all_shortest_paths(g, u, v))
            for p in sp:
                for _i in range(len(p) - 1):
                    weights[p[_i], p[_i - 1]] += 1 / len(sp)

    return weights