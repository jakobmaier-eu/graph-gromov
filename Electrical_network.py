import numpy as np
import networkx as nx


def eff_resistance(G):
    return np.linalg.pinv((nx.laplacian_matrix(G)).toarray())

def eff_conductance(G):
    L = eff_resistance(G)
    n, m = L.shape
    for i in range(n):
        for j in range(m):
            L[i, j] = 1/L[i, j]
    return L

