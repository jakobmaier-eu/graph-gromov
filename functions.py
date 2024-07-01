import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize_scalar

def ortho_projection(A):
    U, _, VT = np.linalg.svd(A, full_matrices=True)
    return U@VT

def perm_projection(M, A_eq, b_eq):
    n = len(M)
    Mflat = -M.flatten()
    result = linprog(Mflat, A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)]*(n**2))
    P = result.x.reshape(n, n)
    if False: # Next lines don't contribute
        print("Trace with P = "+str(np.trace(P.T@M)))
        P_alt = rand_perm_matrix(n)
        print("Trace with P_alt = "+str(np.trace(P_alt.T@M)))
    return P

def ortho_from_perm(P, X, Y):
    return ortho_projection(Y @ P @ X.T)

def perm_from_ortho(Q, X, Y, A_eq, b_eq):
    # return np.transpose(perm_projection(X.T @ Q.T @ Y, A_eq, b_eq))
    return perm_projection(Y.T @ Q @ X, A_eq, b_eq)


def f(D, X, Y):
    return np.linalg.norm(D @ X.T @ X - Y.T @ Y @ D, ord='fro')

def grad_f(D, X, Y):
    return 2*( D @ X.T @ X @ X.T @ X - 2 * Y.T @ Y @ D @ X.T @ X + Y.T @ Y @ Y.T @ Y @ D)

def f_maxtrace(D, X, Y):
    return -np.trace(X.T @ X @ D.T @ Y.T @ Y @ D)

def grad_f_maxtrace(D, X, Y): # negative because we minimize
    return - 2 * Y.T @ Y @ D @ X.T @ X

def empirical_loss(X, Y, P, Q):
    return np.linalg.norm(Q@X - Y@P, ord='fro')

def L2ortho_squared(Q, Qprime):
    return np.linalg.norm(Q - Qprime, ord='fro')**2

def L2perm_squared(P, Pprime, X):
    n = len(P)
    return (1/n) * np.linalg.norm(X @ P.T - X @ Pprime.T, ord='fro')**2

def overlap(P, Pprime):
    n = len(P)
    return (1/n)*np.trace(P.T @ Pprime)

def is_permutation(M):
    n = len(M)
    if M.shape != (n,n): return False
    for i in range(n):
        if sum(M[i,:]) != 1: return False
        if sum(M[:,i]) != 1: return False
    if not (M.T@M == np.eye(n)).all(): return False
    return True

def matrix_heatmap(M):
    plt.imshow(M, cmap='hot')
    plt.show()

def equality_constraints(n):
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

def initialise_XYPstarQstar(n, d, sigma, seed=154):
    np.random.seed(seed)
    X = np.zeros((d, n))
    Z = np.zeros((d, n))
    for i in range(n):
        xi = np.random.multivariate_normal(mean = np.zeros(d), cov = np.eye(d))
        X[:,i] = xi
        zi = np.random.multivariate_normal(mean = np.zeros(d), cov = np.eye(d))
        Z[:,i] = zi

    # Sample orthogonal matrix:
    Helper = np.random.randn(d, d)
    Qstar, _ = scipy.linalg.qr(Helper)

    Pstar=rand_perm_matrix(n)
    Pstar=np.eye(n) # wlog, currrently
    Y = Qstar@X@Pstar.T + sigma*Z
    return X, Y, Pstar, Qstar


def initialise_XYPstarQstar_old(n, d, sigma, plot=False, seed=123):
    np.random.seed(seed)
    X = np.zeros((d, n))
    Z = np.zeros((d, n))
    for i in range(n):
        xi = np.random.multivariate_normal(mean = np.zeros(d), cov = np.eye(d))
        X[:,i] = xi
        zi = np.random.multivariate_normal(mean = np.zeros(d), cov = np.eye(d))
        Z[:,i] = zi

    # Sample orthogonal matrix:
    Helper = np.random.randn(d, d)
    Qstar, _ = scipy.linalg.qr(Helper)

    # Sample permutation and convert to matrix
    pistar = np.random.permutation(n)
    Pstar = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if pistar[i] == j:
                Pstar[i,j] = 1
    Y = Qstar@X@Pstar.T + sigma*Z
    if plot:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(X[:,0], X[:,1], X[:,2], color="green");
        ax.scatter3D(Y[:,0], Y[:,1], Y[:,2], color="blue");
        plt.show()
    return X, Y, Pstar, Qstar


#------ Simultaneous FW ------#
def F(Q, P, X, Y):
    return np.linalg.norm(Q@X - Y@P)**2

def minus_grad_F(Q, P, X, Y):
    return - (Q@X - Y@P)@X.T , + Y.T@(Q@X - Y@P) 