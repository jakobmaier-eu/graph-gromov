import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def d(A, B):
    return np.sqrt(np.linalg.norm(A - B))

def K(C, epsilon):
    K = C
    n, m = C.shape
    for i in range(n):
        for j in range(m):
            K[i][j] = np.exp(C[i][j]/epsilon)
    return K

def div(a, b):
    n = len(a)
    m = len(b)
    S = []
    if n != m:
        raise RuntimeError("Vectors a and b have non-compatible shape.")
    for i in range(n):
        S.append(a[i]/b[i])
    return S
    
    

## Sinkhorn algorithm from Rioux2024
def sinkhorn(a, b, C, epsilon, threshold, max_iteration):
    n = len(a)
    m = len(b)
    N0, N1 = C.shape
    if n != N0 or m != N1: 
        raise RuntimeError("Vectors a, b and matrice C have non-compatible shape.")
    u = 1/n * np.transpose(np.ones((1, n))) #Taille a revoir ?
    k = 1
    P = K(C, epsilon)
    v = b
    dist = d(np.transpose(P) * np.ones((1, n)),  b) 
    while k < max_iteration +1 and dist > threshold:
        v = div(b,(np.transpose(P)).dot(u))
        u = div(a,(P.dot(v)))
        P = np.diag(u) * P * np.diag(v)
        dist = d(np.transpose(P) * np.ones((1, n)) ,  b)
        k = k + 1
    return P


#x y atoms of measures, P oracle from Sinkhorn
def gradient(x, y, A, P):
    N = len(x)
    M = len(y)#useless ?
    S = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            S[i, j] = x[i] * y[j] * P[i, j]
    return 64* A - 32 * S

#useless ?
def min_wise(m, M): #min(m, 1) * M
    n = len(M)
    m = len(M[0])
    S = M
    for i in range(n):
        for j in range(m):
            S[i, j] = min(m, 1) * M[i, j]
    return S

def min_wise_abs(m, M): #min(m, 1) * M
    n = len(M)
    m = len(M[0])
    S = M
    for i in range(n):
        for j in range(m):
            S[i, j] = np.sign(M[i, j]) * min(np.sign(M[i, j]) * m * abs(M[i, j]), 1)
    return S
            
#x and y support of measures, C0 in DM, P is an oracle from Sinkhorn
def inexact_gradient_method(x, y, k_max, M, C0, L, P): #fonction L-smooth
    C = C0
    k = 1
    A = C
    G = gradient(x, y, A, P) #gradient
    while k < k_max: #Voir les conditions de fin de boucle
        gamma = k/(4*L)
        tau = 2/(k + 2)
        B = min(1, M/(2 * np.linalg.norm(A - (1/(2*L)) *G )) ) * (A - (1/(2*L)*G))
        C = min(1, M/(2 * np.linalg.norm(C - gamma *G )) ) * (C - gamma *G)
        B = M/2 * min_wise_abs(2/M, A - (1/(2*L)) *G)
        C = M/2 * min_wise_abs(2/M, C - gamma * G)
        A = tau * C + (1 - tau) * B
        G = gradient(x, y, A, P)
        k = k + 1
    return B