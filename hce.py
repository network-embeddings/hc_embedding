""" Hyperbolic Coalescent Embeddings

input: an edge list

output: a table with the each node's coordinate

"""


import argparse
import logging

import matplotlib.pyplot as plt
import networkx as nx
from sklearn import manifold

import angular_coords
import pl_exponent_fit
import pre_weights
import radial_coord


def get_parser():
    parser = argparse.ArgumentParser(
        description='Coalescent embedding in the hyperbolic space.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'infile',
        metavar='edgelist',
        type=str,
        help='An input network file containing an edge list.',
    )

    parser.add_argument(
        '-s'
        '--save',
        type=bool,
        default=True,
        help='Toggles saving of the visualized network.',
    )
    parser.add_argument(
        '-o',
        '--output_path',
        type=str,
        default='./hce_network.png',
        help='Location where the network visualization should be saved, can also control the output format.',
    )
    parser.add_argument(
        '-d',
        '--display',
        type=bool,
        default=True,
        help='Toggles the display of the visualized network.',
    )

    return parser


def hce(G, pre_weighting='RA1', embedding=None, angular='EA'):
    """
    Computes a hyperbolic coalescent embedding of a given graph.

    Args:
        G: networkx.Graph
        pre_weighting: str, Determines the features that are passed to the dimensionality reduction method.
        embedding: Object, An embedding model that implements the fit_transform method (like sklearn.manifold.SpectralEmbedding).
        angular: str, Determines the method used to create the angular coordinates for the final embedding.

    Returns: numpy.ndarray
        (N, 2) array that contains the spatial (x, y) coordinates of the embedded network.
    """
    weight_func = getattr(pre_weights, f'{pre_weighting}_weights')
    embedding_model = embedding or manifold.SpectralEmbedding()
    angular_func = getattr(angular_coords, f'{angular}_coords')

    weights = weight_func(G)
    embedded_weights = embedding_model.fit_transform(weights)

    coords = angular_func(embedded_weights)

    beta = 1 / (pl_exponent_fit.get_pl_exponent(G) - 1)
    radii = radial_coord.radial_coord_deg(G, beta)

    coords = coords * radii[..., None]
    return {node: coord for node, coord in zip(G.nodes, coords)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = get_parser().parse_args()
    logging.info(args)

    G = nx.read_edgelist(args.edgelist)

    fig, ax = plt.subplots()
    nx.draw(
        G,
        pos=hce(G),
        ax=ax,
    )

    if args.save:
        plt.savefig(args.output_path)

    if args.display:
        plt.show()

    plt.close()
