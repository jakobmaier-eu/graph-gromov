import numpy as np
import networkx as nx


def eff_conductance(G, delta):
    L = np.linalg.pinv((nx.laplacian_matrix(G)).toarray())
    n, m = L.shape
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if L[i, j] != 0:
                R[i, j] = 1/L[i, j]
            else:
                R[i, j] = delta
    return R



