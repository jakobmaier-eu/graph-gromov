import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random as rd
from curvature import *


#Using https://github.com/aidos-lab/curvature-filtrations/tree/main for the curvature

def position_list(L, e):
    for i in range(len(L)):
        if L[i] == e:
            return i
    raise RuntimeError('No element e in L')

def subsampling(G, ntilde, Curv): #Better with S connected !!!
    n = nx.number_of_nodes(G)
    A = nx.adjacency_matrix(G)
    v0 = rd.randint(0, n - 1)
    v1 = rd.choice(list(nx.neighbors(G, v0)))
    S = [v1]
    vtm = v0
    vt = v1
    while len(S) < ntilde:
        N = G.edges([vtm, vt])
        N = [ (x,y) for (x,y) in N if not(x in S and y in S) or x ==y ]
        if len(N) == 0:
            break
        L = [Curv[position_list(N, e)] for e in N]
        i = np.argmax(L)
        n1, n2 = (N[i])
        et = -1
        if n1 == vtm or n1 == vt:
            et = n2
            vtm = n1
        else:
            et = n1
            vtm = n2
        S.append(et)
        vt = et
    return S, G.subgraph(S).copy()

def add_nodes(G, k, p):
    Gp = nx.convert_node_labels_to_integers(G)
    n = nx.number_of_nodes(Gp)
    for i in range(n, n + k):
        Gp.add_node(i)
        for j in range(i):
            if np.random.rand() < p:
                Gp.add_edge(i, j)
    return Gp

def how_many_edges_left(G, S): #Dans le sens de combien reste t-il
    l = 0
    for i in S:
        l += G.degree[i]
    return l

def noisy_curvature_graph(G, ntilde, curv="resistance"):
    if curv=="resistance": Curv= resistance_curvature(G)
    if curv=="ollivier-ricci": Curv= ollivier_ricci_curvature(G)
    S, Gs = subsampling(G, ntilde, Curv)
    print(Gs)
    print("S = ", S)
    k = nx.number_of_nodes(G) - len(S)
    l = nx.number_of_edges(G) - nx.number_of_edges(Gs)
    print("k = ", k)
    print("l =",l)
    Gs = add_nodes(Gs, k, p) #HAS TO BE CHANGED
    return Gs

#some tests
n = 50
p = 0.5
G = nx.erdos_renyi_graph(n, p)
print("G = ", G)

Gp = noisy_curvature_graph(G, 30)
print("Gp = ",Gp)
nx.draw(Gp)
plt.show()