#!/usr/bin/env python

####  GPR optimization  ####


import time
import numpy as np

from scipy.optimize import minimize


def cov_kernel(X, ell, sf2):
    """ Squared Exponential Automatic Relevance Determination (SE ARD) covariance kernel K.

    # Arguments:
        X:    Inputs training data matrix of size (N x Nx) - Nx is the number of inputs to the GP.
        ell:  Length scales vector of size Nx.
        sf2:  Signal variance (scalar)
    """
    dist = 0
    n, D = X.shape
    for i in range(D):
        x = X[:, i].reshape(n, 1)
        dist = (np.sum(x**2, 1).reshape(-1, 1) + np.sum(x**2, 1) -
                2 * np.dot(x, x.T)) / ell[i]**2 + dist
    return sf2 * np.exp(-.5 * dist)


def NLL(hyper, X, Y):
    """ Negative Log Likelihood (NLL) function - objective function.

    # Arguments:
        hyper:  Vector of hyperparameters [ell_1 .. ell_Nx sf sn] - Nx is the number of inputs to the GP.
        X:      Inputs training data matrix of size (N x Nx).
        Y:      Onputs training data matrix of size (N x Ny) - Ny number of outputs.

    # Returns:
        NLL:    The negative log likelihood function (scalar)
    """

    n, D = X.shape
    ell = hyper[:D]
    sf2 = hyper[D]**2
    sn2 = hyper[D + 1]**2
    K = cov_kernel(X, ell, sf2)
    K = K + sn2 * np.eye(n)
    K = (K + K.T) * 0.5   # Make sure matrix is symmentric
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        print("K is not positive definit, adding jitter!")
        K = K + np.eye(n) * 1e-8
        L = np.linalg.cholesky(K)

    logK = 2 * np.sum(np.log(np.abs(np.diag(L))))
    invLy = np.linalg.solve(L, Y)
    alpha = np.linalg.solve(L.T, invLy)
    NLL = 0.5 * np.dot(Y.T, alpha) + 0.5 * logK
    return NLL


def train_gp(X, Y, hyper_init=None, log=False, multistart=1, optimizer_opts=None):
    """ Train hyperparameters

    Maximum likelihood estimation is used to optimize the hyperparameters of
    the Gaussian Process. Sequential Least SQuares Programming (SLSQP) is used
    to find the optimal solution.

    A uniform prior of the hyperparameters are assumed and implemented as
    limits in the optimization problem.

    NOTE: This version only support a zero-mean function.

    # Arguments:
        X: Training data matrix with inputs of size (N x Nx),
            where Nx is the number of inputs to the GP.
        Y: Training data matrix with outputs of size (N x Ny),
            with Ny number of outputs.

    # Return:
        opt: Dictionary with the optimal hyperparameters [ell_1 .. ell_Nx sf sn].
    """
    if log:
        X = np.log(X)
        Y = np.log(Y)

    N, Nx = X.shape
    Ny = Y.shape[1]

    h_ell   = Nx    
    h_sf    = 1     
    h_sn    = 1    
    num_hyp = h_ell + h_sf + h_sn 

    options = {'disp': True, 'maxiter': 10000}
    if optimizer_opts is not None:
        options.update(optimizer_opts)

    hyp_opt = np.zeros((Ny, num_hyp))
    invK = np.zeros((Ny, N, N))
    alpha = np.zeros((Ny, N))
    chol = np.zeros((Ny, N, N))

    print('\n________________________________________')
    print('# Optimizing hyperparameters (N=%d)' % N )
    print('----------------------------------------')
    for output in range(Ny):
        meanF     = np.mean(Y[:, output])
        lb        = -np.inf * np.ones(num_hyp)
        ub        = np.inf * np.ones(num_hyp)
        lb[:Nx]    = 1e-2
        ub[:Nx]    = 2e2
        lb[Nx]     = 1e-8
        ub[Nx]     = 1e2
        lb[Nx + 1] = 1e-10
        ub[Nx + 1] = 1e-2
        bounds = np.hstack((lb.reshape(num_hyp, 1), ub.reshape(num_hyp, 1)))

        if hyper_init is None:
            hyp_init = np.zeros((num_hyp))
            hyp_init[:Nx] = np.std(X, 0)
            hyp_init[Nx] = np.std(Y[:, output])
            hyp_init[Nx + 1] = 1e-5
        else:
            hyp_init = hyper_init[output, :]


        obj = np.zeros((multistart, 1))
        hyp_opt_loc = np.zeros((multistart, num_hyp))
        for i in range(multistart):
            solve_time = -time.time()
            res = minimize(NLL, hyp_init, args=(X, Y[:, output]),
                           method='SLSQP', options=options, bounds=bounds, tol=1e-15)
            obj[i] = res.fun
            hyp_opt_loc[i, :] = res.x
        solve_time += time.time()
        print("* Output %d:  %f s" % (output+1, solve_time))

        # With multistart, get solution with lowest decision function value
        hyp_opt[output, :]   = hyp_opt_loc[np.argmin(obj)]
        ell = hyp_opt[output, :Nx]
        sf2 = hyp_opt[output, Nx]**2
        sn2 = hyp_opt[output, Nx + 1]**2

        # Calculate the inverse covariance matrix
        K = cov_kernel(X, ell, sf2)
        K = K + sn2 * np.eye(N)
        K = (K + K.T) * 0.5   # Make sure matrix is symmentric
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            print("K matrix is not positive definit, adding jitter!")
            K = K + np.eye(N) * 1e-8
            L = np.linalg.cholesky(K)
        invL = np.linalg.solve(L, np.eye(N))
        invK[output, :, :] = np.linalg.solve(L.T, invL)
        chol[output] = L
        alpha[output] = np.linalg.solve(L.T, np.linalg.solve(L, Y[:, output]))
        
    print('----------------------------------------')

    opt = {}
    opt['hyper'] = hyp_opt
    opt['invK'] = invK
    opt['alpha'] = alpha
    opt['chol'] = chol
    return opt

