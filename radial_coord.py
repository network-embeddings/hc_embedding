import math
import numpy as np


def radial_coord(G, beta, zeta=1):
    """
    Nodes are first sorted i = 1,2, ... N according to their degree
    (descending).

    Then r_i is calculated by the formula in pg. 3

    r_i = \frac{2}{\zeta} ( \beta ln i + (1 - \beta) ln N )

    Args:
        G: networkx.Graph
        zeta: float, sqrt(-K), where K is the curature of the hyperbolic space.
        beta: float, Exponent of the power-law degree distribution.

    Returns:

    """
    sorted_node_ids = [i for i, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)]

    for i, n in enumerate(sorted_node_ids):
        r = 2 / zeta * (beta * math.log(i) + (1 - beta) * math.log(len(G)))
        yield n, r


def radial_coord_deg(degrees, beta, zeta=1):
    """
    Nodes are first sorted i = 1,2, ... N according to their degree
    (descending).

    Then r_i is calculated by the formula in pg. 3

    r_i = \frac{2}{\zeta} ( \beta ln i + (1 - \beta) ln N )

    Args:
        degrees: numpy.ndarray, Degree of each node in some graph.
        beta: float, Exponent of the power-law degree distribution.
        zeta: float, sqrt(-K), where K is the curature of the hyperbolic space.

    Returns: numpy.ndarray
        Radial coordinate for each node in a graph.
    """
    inds = np.argsort(degrees)
    return 2 / zeta * (beta * np.log(inds) + (1 - beta) * np.log(len(inds)))
