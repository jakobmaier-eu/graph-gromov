import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment #Hungarian algorithm
from functions import *

#Onaran-Villar 2017

#compute the alignment network matrix O(n**4)
def align_matrix(G, Gp, s1, s2, s3):
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
    v = np.random.rand(n) #ISSUES HERE !!!
    v = v/np.sqrt(np.linalg.norm(v))
    v_mem = v
    for k in range(100):
        v_mem = v
        v = A.dot(v)
        v = v/np.sqrt(np.linalg.norm(v))
    return v, np.dot(v_mem, v)/np.linalg.norm(v_mem)


def eigenalign(G, Gp, s1, s2, s3):
    n = nx.Graph.number_of_nodes(G)
    A = align_matrix(G, Gp, s1, s2, s3)
    eigvect, _ = np.linalg.eigh(A)
    l = len(eigvect)
    v = eigvect[l - 1]
    A_eq, b_eq =  bistochastic_equality_constraints(n)
    return project_to_perm(np.reshape(v, (n, n)), A_eq, b_eq)

#Eig1 algorithm from Ganassali's thesis

def find_permutation(v1, v2):
    n = len(v1)
    P = np.zeros((n,n))
    for i in range(n):
        j = 0
        while v1[i] != v2[j]:
            j = j+1
        P[i, j] = 1
    return P
            

def eig1(G, Gp):
    A = (nx.adjacency_matrix(G)).toarray()
    Ap = (nx.adjacency_matrix(Gp)).toarray()
    v, _ = iter_power(A)
    vp, _ = iter_power(Ap)
    #idea is to use "inegalite de rearrangement"
    v_max = sorted(v)
    vp_max = sorted(vp)
    P = find_permutation(v, v_max)
    Pp = find_permutation(vp, vp_max)
    Pi_plus = Pp.T @ P
    vp_max = sorted(-vp)
    Pp = find_permutation(-vp, vp_max)
    Pi_moins = Pp.T @ P
    if np.trace(np.matmul(A, Pi_plus @ Ap @ Pi_plus.T)) >= np.trace(np.matmul(A, Pi_moins @ Ap @ Pi_moins.T)):
        return Pi_plus
    else:
        return Pi_moins
