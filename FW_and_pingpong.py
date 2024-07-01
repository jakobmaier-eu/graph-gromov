import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from functions import *

def Qinit(X, Y, Pstar, Qstar, sigma, opt_steps, A_eq, b_eq, stepsize, plot_opt):
    Ux, Sigx, _ = np.linalg.svd(X@X.T, full_matrices=False)
    Uy, Sigy, _ = np.linalg.svd(Y@Y.T, full_matrices=False)
    print(Sigx)
    print(Sigy)
    return Uy@Ux.T

def frank_wolfe(X, Y, Pstar, Qstar, sigma, opt_steps, A_eq, b_eq, stepsize, method="", plot=False):
    if method=="maxtrace": f_here = f_maxtrace; grad_f_here = grad_f_maxtrace
    else: f_here = f; grad_f_here = grad_f
    d, n = X.shape
    Dinit = (1/n)*np.ones((n, n))
    D = Dinit.copy()
    step = 0
    if isinstance(stepsize, float): gamma = stepsize
    if plot: dists_D_Dinit = []; errs_FW_goal = []; losses_wrt_Qstar = []; scalar_prods = []
    while True:
        if plot:
            dists_D_Dinit.append(np.linalg.norm(D - Dinit, ord='fro'))
            errs_FW_goal.append(f_here(D, X, Y))
            losses_wrt_Qstar.append(empirical_loss(X, Y, D, Qstar))
        G = -grad_f_here(D, X, Y)*(1/n)
        S_hat = perm_projection(G, A_eq, b_eq)
        Direction = S_hat - D
        if stepsize=="linesearch": gamma = minimize_scalar(
                lambda gamma:f_here(D + gamma*Direction, X, Y), 
                bounds=(0,1), 
                method='bounded'
                ).x
        if plot:
            print("--------- Step "+str(step)+" ---------")
            print(gamma)
            matrix_heatmap(D)
            matrix_heatmap(G)
            matrix_heatmap(S_hat)
        D = D + gamma*Direction
        if plot: 
            scalar_prods.append(np.trace(G.T @ Direction))
        step += 1
        if step==opt_steps: break
    if plot:
        d, n = X.shape
        xs = range(step)
        plt.plot(xs, np.array(dists_D_Dinit), label="||D - Dinit||")
        plt.plot(xs, errs_FW_goal, label="f(D)")
        # plt.plot(xs, losses_wrt_Qstar, label="||Qstar X - YD||")
        # plt.plot(xs, scalar_prods, label="< -grad_f, dir >")
        plt.legend()
        plt.title("FW with " 
                  +"n="+str(n)+", d="+str(d)+", sig="+str(sigma)
                  +", #step="+str(step)+", gamma="+str(stepsize))
        plt.show()
    return D

def pingpong(PQ0, Pstar, Qstar, X, Y, A_eq, b_eq, plot=False):
    d, n = X.shape
    lastP = np.zeros((n, n))
    lastQ = np.zeros((d, d))
    pindex = 0; qindex = 0
    if len(PQ0)==n:
        P = PQ0
        Q = np.ones((d, d))
        P_turn = False
    elif len(PQ0)==d:
        P = np.ones((n, n))
        Q = PQ0
        P_turn = True
    if plot: loss_errs = []; orth_errs = []; perm_errs = []; my_xticks = []
    while not(np.linalg.norm(lastP-P) == 0 and (np.linalg.norm(lastQ-Q) == 0 or np.linalg.norm(lastQ+Q)==0)):
        if P_turn:
            pindex += 1
            lastP = P
            P = perm_from_ortho(Q, X, Y, A_eq, b_eq)
            P_turn = False
        else:
            qindex += 1
            lastQ = Q
            Q = ortho_from_perm(P, X, Y)
            P_turn = True
        if plot:
            loss_errs.append(empirical_loss(X, Y, P, Q)**2)
            perm_errs.append(L2perm_squared(P, Pstar, X))
            orth_errs.append(L2ortho_squared(Q, Qstar))
            my_xticks.append("(P"+str(pindex)+", Q"+str(qindex)+")")
    if plot:
        xs = range(len(loss_errs))
        plt.plot(xs, loss_errs, label="||QX - YX||^2")
        plt.plot(xs, orth_errs, label="||Q - Q*||^2")
        plt.plot(xs, perm_errs, label="||XP.T - XP*.T||")
        plt.xticks(xs, my_xticks)
        plt.legend()
        plt.show()
    return pindex, P, Q

def experiment(n, d, sigma, opt_algo, opt_steps="", stepsize="", method="", seed_XYZ = 23, plot_opt=False, plot_PP = False):
    # Setup
    A_eq, b_eq = equality_constraints(n)
    X, Y, Pstar, Qstar = initialise_XYPstarQstar(n, d, sigma, seed_XYZ)
    # Find P0
    PQ0 = opt_algo(X, Y, Pstar, Qstar, sigma, opt_steps, A_eq, b_eq, stepsize, method, plot_opt)
    # PingPong
    PP_steps, Pfinal, Qfinal = pingpong(PQ0, Pstar, Qstar, X, Y, A_eq, b_eq, plot_PP)
    # Report results
    algoname = opt_algo.__name__ if opt_algo.__name__ != "frank_wolfe" else "FW"
    stepsizename = "of size "+str(stepsize) if isinstance(stepsize, float) else stepsize
    methodname = "maxtr" if method == "maxtrace" else "basic"
    print("Ov=" + str(overlap(Pfinal, Pstar))
          +", perm_l2="+str("{:.2f}".format(L2perm_squared(Pfinal, Pstar, X)))
          +", ortho_l2="+str("{:.2f}".format(L2ortho_squared(Qfinal, Qstar)))
          +": "+ algoname 
          +" "+ methodname
          +" w "+str(PP_steps)+"-step PP & n="+str(n)
          +", d="+str(d)
          +", sig="+str(sigma)
          +", seed="+str(seed_XYZ)
          +", nsteps="+str(opt_steps)
          +" "+stepsizename
         )
    return X, Y, Pstar, Qstar, Pfinal, Qfinal, PQ0