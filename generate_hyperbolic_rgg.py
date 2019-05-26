import logging
import argparse
import networkx as nx
import numpy as np

def random_position_hyperbolic(R):
    """Random uniform position on a circle of radius R."""
    position = np.array([0., 0.])
    position[0] = np.random.rand() * np.pi * 2.     # theta
    position[1] = np.arccosh(np.random.rand() * (np.cosh(R) - 1) + 1)     # r
    return position


def distance_hyperbolic(x, y):
    """Hyperbolic distance between two points."""
    if (x != y).all():
        cosh_ans = np.cosh(x[1]) * np.cosh(y[1])
        cosh_ans -= np.sinh(x[1]) * np.sinh(y[1]) * np.cos(x[0] - y[0])
        ans = np.arccosh(cosh_ans)
    else:
        ans = 0
    return ans


def hyperbolic_random_graph(n, R):
    """RGG in circular hyperbolic space of radius R."""
    g = nx.Graph()
    for i in range(n):
        g.add_node(i)
        x = random_position_hyperbolic(R)
        g.node[i]["position"] = x
        for j in range(i):
            y = g.node[j]["position"]
            if distance_hyperbolic(x, y) < R and np.random.rand() < 1:
                g.add_edge(i, j)

    return g

def from_polar(pos):
    theta = pos[0]
    r = pos[1]

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def generate_save(n, avgk, filename):
    R = 2 * np.log(8 * n / (np.pi * avgk))

    g = hyperbolic_random_graph(n, R)

    pos = nx.get_node_attributes(g, 'position')
    pos = np.array([from_polar(pos[i]) for i in pos])

    nx.write_edgelist(g, filename + '_edgelist.txt')
    np.savetxt(filename + '_position.txt', pos)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='generate a random geometric graph in hyperbolic dist.')
    parser.add_argument('num_nodes', metavar='num_nodes', type=int,
                    help='number of nodes.')
    parser.add_argument('avgk', metavar='avgk', type=float,
                    help='average degree of the network.')
    parser.add_argument('filename', metavar='filename', type=str,
                    help='Name of file where edgelist and positions are saved.')
    args = parser.parse_args()
    logging.info(args)
    generate_save(args.num_nodes, args.avgk, args.filename)
