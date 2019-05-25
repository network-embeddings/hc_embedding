import numpy as np


def circular_adjustment(coords):
    """
    Args:
        coords: numpy.ndarray, A (N, 2) array of coordinates.

    Returns: numpy.ndarray
        A (N, 2) array of rescaled coordinates that fall on the unit circle.
    """
    return np.array(coords) / np.linalg.norm(coords, ord=2, axis=-1, keepdims=True)


def equidistant_adjustment(coords):
    """

    Args:
        coords: numpy.ndarray, A (N, 2) array of coordinates.

    Returns: numpy.ndarray
        A (N, 2) array of rescaled coordinates that fall on the unit circle and are equidistantly spaced.
    """
    zero_angle_vec = np.array([1, 0]).reshape((1, 2))
    coords = circular_adjustment(coords)

    angles = angle_between(coords, zero_angle_vec)
    inds = np.argsort(angles, axis=0)

    rescaled = np.arange(len(coords))[inds]
    rescaled = rescaled * 2 * np.pi / len(coords)

    return np.hstack((np.cos(rescaled), np.sin(rescaled)))


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'.

    Borrowed from:
        https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = circular_adjustment(v1)
    v2_u = circular_adjustment(v2)
    return np.arccos(np.clip(v1_u @ v2_u.T, -1.0, 1.0))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Draw some random points from the unit square and center them
    x = np.random.random((100, 2))
    x -= x.mean(axis=0)

    coords1 = circular_adjustment(x)
    coords2 = equidistant_adjustment(x)

    plt.figure()
    plt.scatter(coords1[:, 0], coords1[:, 1])

    plt.figure()
    plt.scatter(coords2[:, 0], coords2[:, 1])

    plt.show()
