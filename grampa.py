import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from functions import *
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment
#According to Fan-Mao-Wu-Xu 2019

def grampa(A, B, eta):
    n1, n2 = A.shape
    m1, m2 = B.shape
    if n1 != m1 or n1 != n2 or m1 != m2: 
        raise RuntimeError("Aligning matrices A and B of different shape.")
    
    n = n1
    
    eigvalA, eigvectA = np.linalg.eigh(A)
    eigvalB, eigvectB = np.linalg.eigh(B)
    
    J = np.ones((n, n))
    K = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            K = K + 1 / (eta**2 + (eigvalA[i]- eigvalB[j])**2) * (np.outer(eigvectA[:, i], eigvectA[:, i])) @ J @ (np.outer(eigvectB[:, j], eigvectB[:, j]))
    
    row_ind, col_ind = linear_sum_assignment(-K)
    P = np.zeros((n, n))
    P[row_ind, col_ind] = 1
    
    return P

def grampa_matlab(A, B, eta):
    """
    GRAMPA: Graph Matching by Pairwise Eigen-Alignments
    
    Parameters:
    - A: the (centered or uncentered) adjacency matrix of the first graph
    - B: the (centered or uncentered) adjacency matrix of the second graph
    - eta: the regularization parameter (eta > 0)
    
    Returns:
    - P: the permutation matrix P such that P @ A @ P.T is matched to B
    """
    n = A.shape[0]
    
    # Compute the eigenvalues and eigenvectors
    lambda_vals, U = eig(A)
    mu_vals, V = eig(B)
    
    lambda_vals = np.real(lambda_vals)
    mu_vals = np.real(mu_vals)
    U = np.real(U)
    V = np.real(V)
    
    # Compute the similarity matrix
    lambda_diff = (lambda_vals[:, None] - mu_vals[None, :]) ** 2
    coeff = 1.0 / (lambda_diff + eta ** 2)
    coeff = coeff * (U.T @ np.ones((n, n)) @ V)
    X = U @ coeff @ V.T
    
    # Solve the linear sum assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(-X)
    
    # Create the permutation matrix
    P = np.zeros((n, n))
    P[row_ind, col_ind] = 1
    
    return P