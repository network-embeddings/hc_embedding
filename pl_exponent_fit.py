import numpy as np
import plfit


def get_pl_exponent(g):
    degree = np.array(g.degree())[:, 1]
    myplfit = plfit.plfit(degree)
    x = plfit.plfit.discrete_best_alpha(myplfit, finite=False, verbose=0)
    return x[0]
