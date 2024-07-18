import numpy as np
import networkx as nx


def eff_conductance(G):
    return np.linalg.pinv(nx.laplacian_matrix(G)).toarray()


