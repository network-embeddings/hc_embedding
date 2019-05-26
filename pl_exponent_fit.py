import numpy as np
import powerlaw


def get_pl_exponent(g):
    degree = np.array(g.degree())[:, 1]
    results = powerlaw.Fit(degree)
    return results.power_law.alpha
