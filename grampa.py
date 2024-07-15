import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from functions import *

#According to Fan-Mao-Wu-Xu 2019

def grampa(A, B, eta):
    n1, n2 = A.shape
    m1, m2 = B.shape
    n = n1
    if n1 != m1 or n1 != n2 or m1 != m2: 
        raise RuntimeError("Aligning matrices A and B of different shape.")
    eigvalA, eigvectA = np.linalg.eigh(A)
    eigvalB, eigvectB = np.linalg.eigh(B)
    A_eq, b_eq = bistochastic_equality_constraints(n)
    J = np.ones((n,n))
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K= K + 1/(eta**2 + eigvalA[i]**2 + eigvalB[j]** 2) * eigvectA[i] * eigvalA[i].T * J * eigvectB[j] * eigvalB[j].T
    return project_to_perm(K, A_eq, b_eq)