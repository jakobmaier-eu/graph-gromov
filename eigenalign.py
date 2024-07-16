import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment #Hungarian algorithm

#Onaran-Villar 2017

#compute the alignment network matrix O(n**4)
def align_matrix(G, Gp, s1, s2, s3):
    n = nx.number_of_nodes(G)
    m = nx.number_of_nodes(Gp)
    if n != m:
        raise RuntimeError("Aligning two graphs with different number of nodes.")
    A = np.zeros((n*n, n*n))
    for i in range(n):
        for j in range(n):
            ij = G.has_edge((i,j))
            for k in range(n):
                for l in range(n):
                    kl = Gp.has_edge((k, l))
                    if ij and kl:
                        A[i, j, k, l] = s1
                    elif not(ij) and not(kl):
                        A[i, j, k, l] = s2
                    else:
                        A[i, j, k, l] = s3
    return A

#compute v the top eigenvector of A (iterated power ?)
#maybe useless
def iter_power(A):
    n, m = A.shape
    v = np.random.rand(1, n)
    v = v/np.sqrt(np.linalg.norm(v))
    v_mem = v
    for k in range(100):
        v_mem = v
        v = A * v
        v = v/np.sqrt(np.linalg.norm(v))
    return v, np.inner(v_mem, v)/np.linalg.norm(v_mem)


#Stand-by for today
def eigenalign(G, Gp, s1, s2, s3):
    A = align_matrix(G, Gp, s1, s2, s3)
    eigval, eigvect = np.linalg.eignh(A) #ascending order
    eigval = reversed(eigval)
    eigvect = reversed(eigvect)
    a, b = linear_sum_assignment(a)
    return -1