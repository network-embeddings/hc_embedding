import math

def radial_coord(G, zeta, beta):
    """
        Nodes are first sorted i = 1,2, ... N according to their degree
        (descending).

        Then r_i is calculated by the formula in pg. 3

        r_i = \frac{2}{\zeta} ( \beta ln i + (1 - \beta) ln N )
    """
    sorted_node_ids = [i for i, k in sorted(G.degree, key=lambda x: x[1], reverse=True)]

    for i, n in sorted_node_ids:
        r = 2/zeta * ( beta * math.log(i) + (1 - beta) math.log(len(G)))
        yield i, r

