import numpy as np
import matplotlib.pyplot as plt
from munkres import Munkres
from functions import *
from frank_wolfe import *
from scipy import optimize
from functools import partial #to avaluate a function partially
import networkx as nx

#ISSUES BECAUSE MUNKRES USES LEN

#Parameters
n = 10
p = 0.3
epsilon = 0.1
A_eq, b_eq = bistochastic_equality_constraints(n)

def sample_Erdos_Renyi(n, p):
    G = nx.erdos_renyi_graph(n, p,  directed=False)
    return G, nx.adjacency_matrix(G)

G0, A0 = sample_Erdos_Renyi(n,p)
G1, A1 = sample_Erdos_Renyi(n,p)

#Functions

def f(A, B, P): #Function to optimize
    return - np.matrix.trace(A * P * B.T * P.T)

def optim(A, B, P, Q, alpha): #In order to partially evaluate
    print(P)
    print(Q)
    return f(A, B, P + alpha * Q)
# non plus P une permutatio

#FAQ = Frank-Wolfe with the Hungarian algorithm so...
def faq(A, B, epsilon):
    n1, n2 = A.shape
    m1, m2 = B.shape
    if n1 != m1 or n1 != n2 or m2 != m1:
        raise RuntimeError("Matrices A and B must be squared and have the same dimention")
    P = 1/n1 * np.ones((n1, n1)) #We can also take another starting point
    print(P)
    k = 1
    G = np.ones((n1,n1))
    Q = np.ones((n1,n1))
    alpha = 0
    while np.linalg.norm(G) >= epsilon: #stop criteria to find ?
        #compute the gradient of f
        G = -A * P * B.T - (A.T) * P * B
        #hungarian algorithm
        m = Munkres()
        Q = m.compute((G.T).toarray()) #Not the good renvoi !!
        #Frank-Wolfe step-size
        g = partial(optim, A, B, P, Q)
        alpha = optimize.minimize(g, 0) #to find the minimal value
        P = P + alpha * Q
    return P #project !!!!!

print((A0).toarray())
print((A1).toarray())
print(faq(A0, A1, epsilon))