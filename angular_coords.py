import numpy as np


def CA_coords(coords):
    """
    Args:
        coords: numpy.ndarray, A (N, 2) array of coordinates.

    Returns: numpy.ndarray
        A (N, 2) array of rescaled coordinates that fall on the unit circle.
    """
    return np.array(coords) / np.linalg.norm(coords, ord=2, axis=-1, keepdims=True)


def EA_coords(coords):
    """

    Args:
        coords: numpy.ndarray, A (N, 2) array of coordinates.

    Returns: numpy.ndarray
        A (N, 2) array of rescaled coordinates that fall on the unit circle and are equidistantly spaced.
    """
    zero_angle_vec = np.array([1, 0]).reshape((1, 2))
    coords = CA_coords(coords)

    angles = angle_between(coords, zero_angle_vec)
    inds = np.argsort(angles, axis=0)

    rescaled = np.arange(len(coords))[inds]
    rescaled = rescaled * 2 * np.pi / len(coords)

    return np.hstack((np.cos(rescaled), np.sin(rescaled)))


def angle_between(vecs, baseline):
    """
    Computes the angle between each vector in vecs and baseline, which is assumed to be a single vector.

    Adapted from:
        https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793

    Args:
        vecs: numpy.ndarray, (N, M) array.
        baseline: numpy.ndarray, (1, M) array that defines a reference point.

    Returns: numpy.ndarray
        (N, 1) array of angles.
    """
    vecs = CA_coords(vecs)
    baseline = CA_coords(baseline)
    return np.arccos(np.clip(vecs @ baseline.T, -1.0, 1.0))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Draw some random points from the unit square and center them
    x = np.random.random((100, 2))
    x -= x.mean(axis=0)

    coords1 = CA_coords(x)
    coords2 = EA_coords(x)

    plt.figure()
    plt.scatter(coords1[:, 0], coords1[:, 1])

    plt.figure()
    plt.scatter(coords2[:, 0], coords2[:, 1])

    plt.show()
