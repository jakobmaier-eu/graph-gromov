import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize_scalar

def project_to_perm(M, A_eq, b_eq): # Projects a matrix onto the set of permutations
    n = len(M)
    Mflat = -M.flatten()
    result = linprog(Mflat, A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)]*(n**2))
    P = result.x.reshape(n, n)
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