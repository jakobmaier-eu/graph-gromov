import numpy as np
import networkx as nx


def eff_conductance(G, delta):
    Lplus = np.linalg.pinv((nx.laplacian_matrix(G)).toarray())
    n, m = Lplus.shape
    C =  np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Rij = Lplus[i, i] + Lplus[j, j] - Lplus[i, j] - Lplus[j, i]
            if Rij != 0:
                C[i, j] = 1/Rij
            else:
                C[i, j] = delta
    return C



