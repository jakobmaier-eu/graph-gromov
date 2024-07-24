import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize_scalar
import networkx as nx

def project_to_perm(M, A_eq, b_eq): # Projects a matrix onto the set of permutations
    n = len(M)
    Mflat = -M.flatten()
    result = linprog(Mflat, A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)]*(n**2))
    P = result.x.reshape(n, n)
    return P

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def greedy_to_perm(M):
    n = len(M)
    P = M
    for i in range(n):
        m = argmax(M[i])
        for j in range(n):
            P[i, j] = 0
            P[j, m] = 0
        P[i, m] = 1
    return P

def f_diff(D, A, B):
    return np.linalg.norm(A@D- D@B, ord='fro')**2

def grad_f_diff(D, A, B):
    return 2 * (A@A@D + D@B@B - 2*A@D@B)

def f_minustrace(D, A, B):
    return -np.trace(A@ D @ B.T @ D.T)

def grad_f_minustrace(D, A, B): # negative because we minimize
    return - 2 * A@D@B

def overlap(P1, P2): # potentially need to rescale for plot
    n1, n2 = P1.shape
    m1, m2 = P2.shape
    n = n1
    return (1/n)*np.trace(P1.T @ P2)

def is_permutation(M): # checks if matrix M is a permutation
    n = len(M)
    if M.shape != (n,n): return False
    for i in range(n):
        if sum(M[i,:]) != 1: return False
        if sum(M[:,i]) != 1: return False
    if not (M.T@M == np.eye(n)).all(): return False
    return True

def matrix_heatmap(M): # paints a matrix M
    plt.imshow(M, cmap='hot')
    plt.show()

def bistochastic_equality_constraints(n):
    # Need to change this when working with different marginals than bistochastic matrices.
    A_eq = np.zeros((2 * n, n ** 2))
    for i in range(n):
        A_eq[i, i * n : (i + 1) * n] = 1  # Constraint 1: Sum of each row of S equals 1
        A_eq[n + i, i::n] = 1  # Constraint 2: Sum of each column of S equals 1
    b_eq = np.ones(2 * n)
    return A_eq, b_eq

def rand_perm_matrix(n):
    pistar = np.random.permutation(n)
    Pstar = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if pistar[i] == j:
                Pstar[i,j] = 1
    return Pstar


def plot_alignment(G, Gprime, A):
    n1, n2 = A.shape
    m1 = nx.Graph.number_of_nodes(G)
    m2 = nx.Graph.number_of_nodes(Gprime)
    if n1 != m1 or n2 != m2:
        raise RuntimeError("Aligning G and Gprime with a wrong A matrix.")
    for i in range(n1):
        plt.plot(0, 2*i, 'bo')
    for i in range(m2):
        plt.plot(max(n1,m1)/2, 2*i, 'bo')
    for i in range(n1):
        for j in range(m2):
            if A[i, j] != 0:
                plt.plot([0, max(n1,m1)/2],[ 2*i, 2*j], linewidth = A[i,j] * 2, color='red')
    plt.plot()

def auxiliaire_norm_plot(L):
    n = len(L)
    m = len(L[0])
    R = []
    for i in range(m):
        Li = []
        for t in range(n):
            Li.append(L[t][i])
        R.append(Li)
    return R


def norm_plot(L, Label, x, p, s): #L[n, M] = matrice of the distances for all n=edges  
    n = len(L)
    m = len(Label)
    R = auxiliaire_norm_plot(L)
    for i in range(m):
        plt.plot(x, R[i], label=Label[i])
    plt.xlabel('n = number of edges')
    plt.ylabel('|| A - PBtP||')
    plt.legend()
    plt.title('p = ' + str(p) + ', s = ' + str(s))
    plt.plot()

#trying something
def ihara(G):
    n = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)
    A = nx.adjacency_matrix(G)
    A = A.toarray()
    D = np.diag(np.diag(A))
    def F(t):
        return 1/((1 - t**2)**(m - n) * np.linalg.det(np.eye(n) - t*A + (D - np.eye(n)) * t**2))
    return F