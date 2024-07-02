import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from functions import *

def frank_wolfe(A, B, no_opt_steps, stepsize="linesearch", Pstar=None, Dinit=None, method="mindiff", plot=False):
    n = A.shape()[0]
    m = B.shape()[0]
    if n != m: 
        raise RuntimeError("Aligning matrices A and B of different shape.")
    A_eq, b_eq = bistochastic_equality_constraints(n)
    if method=="minustrace": f = f_minustrace; grad_f = grad_f_minustrace
    if method=="diff": f = f_diff; grad_f = grad_f_diff
    if Dinit == None: D_init = (1/n)*np.ones((n, n))
    D = Dinit.copy()
    step = 0
    if isinstance(stepsize, float): gamma = stepsize
    if plot: dists_D_Dinit = []; errs_FW_goal = []; overlaps_D_Pstar = []
    while True:
        if plot:
            dists_D_Dinit.append(np.linalg.norm(D - Dinit, ord='fro'))
            errs_FW_goal.append(f(D, A, B))
            overlaps_D_Pstar.append(overlap(D, Pstar))
        G = -grad_f(D, A, B)
        G_proj = project_to_perm(G, A_eq, b_eq)
        Direction = G_proj - D
        if stepsize=="linesearch": gamma = minimize_scalar(
                lambda gamma:f(D + gamma*Direction, A, B), 
                bounds=(0,1), 
                method='bounded'
                ).x
        D = D + gamma * Direction
        step += 1
        if step==no_opt_steps: break
    if plot:
        xs = range(step)
        plt.plot(xs, dists_D_Dinit, label="||D - Dinit||")
        plt.plot(xs, errs_FW_goal, label="f(D)")
        plt.plot(xs, overlaps_D_Pstar, label="ov(D, Pstar)")
        plt.legend()
        plt.title("FW with "+"n="+str(n)
                  +", #steps="+str(step)+", gamma="+str(stepsize))
        plt.show()
    return D
