import numpy as np
import networkx as nx
from functions import *
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

def align_matrix(G, Gp):
    n = nx.number_of_nodes(G)
    m = nx.number_of_nodes(Gp)
    if n != m:
        raise RuntimeError("Aligning two graphs with different number of nodes.")
    A = np.zeros((n,n, n,n))
    for i in range(n):
        for j in range(n):
            ij = G.has_edge(i,j)
            for k in range(n):
                for l in range(n):
                    kl = Gp.has_edge(k, l)
                    if ij and kl:
                        A[i, j, k, l] = 1/(Gp.degree(k) * Gp.degree(l))
    return A

def isorank(G, Gp):
    n = nx.Graph.number_of_nodes(G)
    A = align_matrix(G, Gp)
    eigvect, _ = np.linalg.eigh(A)
    l = len(eigvect)
    v = eigvect[l - 1] 
    A_eq, b_eq =  bistochastic_equality_constraints(n)
    return project_to_perm(np.reshape(v, (n, n)), A_eq, b_eq)