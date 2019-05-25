import networkx as nx
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import plfit



def get_pl_exponent(g):
    degree = np.array(g.degree())[:, 1]
    myplfit = plfit.plfit(degree)
    x = plfit.plfit.discrete_best_alpha(myplfit, finite=False, verbose=0)
    return x[0]