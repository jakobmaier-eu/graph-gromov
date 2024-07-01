from FW_and_pingpong import *

# def experiment(n, d, sigma, opt_algo, opt_steps="", stepsize="", method="", seed_XYZ = 23, plot_opt=False, plot_PP = False):

X, Y, Pstar, Qstar, Pfinal, Qfinal, PQ0 = experiment(
    100, 10, 0, frank_wolfe, opt_steps=3, stepsize="linesearch", plot_opt=True)

for sigma in [0.01]:
    for d in [10]:
        print("----- d="+str(d)+ ", sigma="+str(sigma)+" -----")
        for opt_steps in [100]:
            n = 100
            # stepsize = 1/50
            experiment(n, d, sigma, frank_wolfe, opt_steps, 1/50 ,method = "maxtrace", plot_opt=False)
            experiment(n, d, sigma, frank_wolfe, opt_steps, 'linesearch',method = "maxtrace", plot_opt=False)