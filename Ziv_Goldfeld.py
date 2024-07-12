import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def d(A, B):
    return np.sqrt(np.linalg.norm(A - B))

def K(C, epsilon):
    K = C
    for i in range(len(C)):
        for j in range(len(C[0])):
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
    N0 = len(C)
    N1 = len(C[0])
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

#Graphs for tests

def sample_Erdos_Renyi(n, p, seed=123):
    G = nx.erdos_renyi_graph(n, p, seed, directed=False)
    return G, nx.adjacency_matrix(G)

p = 0.3
n = 10
epsilon = 0.1
threshold = 0.1
max_iteration = 100

G1, A1 = sample_Erdos_Renyi(n, p)
G2, A2 = sample_Erdos_Renyi(n, p)

nx.draw_networkx(G1)#useless ?
plt.show()
nx.draw_networkx(G2)
plt.show()

#Cost function ?
C = np.random.randint(10, size=(n, n))
mes = 1/n * np.transpose( np.ones((1, n)) )
C = (C + C.T)/2
P = sinkhorn(mes, mes, C, epsilon, threshold, max_iteration)

print(P)


#plot in a good way ?


##Adaptive gradient method with inexact oracle from Rioux 2024

k_max = 100
M = 10 ##Calculer M ?

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
def inexact_gradient_method(x, y, C0, L, P): #fonction L-smooth
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
