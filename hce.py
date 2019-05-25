""" Hyperbolic Coalescent Embeddings

input: an edge list

output: a table with the each node's coordinate

"""

import angular_coords
import pl_exponent_fit
import pre_weights
import radial_coord
from sklearn import manifold


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

    beta = pl_exponent_fit.get_pl_exponent(G)
    radii = radial_coord.radial_coord_deg(G, beta)

    return coords * radii


if __name__ == "__main__":
    # TODO: Add CLI logic here, should load the graph, pass it to hce, and then plot the result.
    pass
