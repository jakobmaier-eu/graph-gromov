import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from functions import *

def FW_simultaneous(X, Y, opt_steps, A_eq, b_eq):
    d, n = X.shape
    R = np.zeros((d,d))
    D = (1/n)*np.ones((n, n))
    step = 0
    while True:
        grad_R, grad_D = minus_grad_F(R, D, X, Y)
        R_hat = grad_R/ np.linalg.norm(grad_R, ord=2)
        D_hat = perm_projection(grad_D, A_eq, b_eq)
        R_dir = R_hat - R
        D_dir = D_hat - D
        gamma = minimize_scalar(
            lambda gamma: F(R + gamma*R_dir, D + gamma*D_dir, X, Y), 
            bounds=(0,1), 
            method='bounded'
            ).x
        R = R + gamma*R_dir
        D = D + gamma*D_dir
        step += 1
        if step==opt_steps: break
    return R, D