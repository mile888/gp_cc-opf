#!/usr/bin/env python

####  GPR approximation functions  ####

import numpy as np
import casadi as ca


# Casadi
def cov_se_ard(x, z, ell, sf2):
    """ Squared Exponential Automatic Relevance Determination (SE ARD) kernel.
    
    # Arguments:
        x:    Sample vector of training input matrix X (1 x Nx) - Nx is the number of inputs to the GP.
        z:    Unseen smaple vector (1 x Nx) - Nx is the number of inputs to the GP.
        ell:  Length scales vector of size Nx.
        sf2:  Signal variance (scalar)
    """
    
    dist = ca.sum1((x - z)**2 / ell**2)
    return sf2 * ca.MX.exp(-.5 * dist)


# Numpy
def cov_kernel(x, z, ell, sf2):
    """ Squared Exponential Automatic Relevance Determination (SE ARD) kernel.
    
    # Arguments:
        x:    Sample vector of training input matrix X (1 x Nx) - Nx is the number of inputs to the GP.
        z:    Unseen smaple vector (1 x Nx) - Nx is the number of inputs to the GP.
        ell:  Length scales vector of size Nx.
        sf2:  Signal variance (scalar)
    """
    
    dist = np.sum(((x - z)**2 / ell**2), axis=1)
    return sf2 * np.exp(-0.5 * dist)



def gp(X, Y, X_new_mean, invK, hyper, alpha=None):
    """ Gaussian Process
    
    # Arguments
        X_new_mean: Input to the GP of size (1 x Nx).
        X: Training data matrix with inputs of size (N x Nx) - Nx is the number of inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny).
        invK: Array with the inverse covariance matrices of size (Ny x N x N) - Ny number of outputs from the GP and N number of training points.
        hyper: Array with hyperparame|ters [ell_1 .. ell_Nx sf sn].
        alpha: Training data matrix with invK time outputs of size (Ny x N).

    # Returns
        mean: The estimated mean.
        var: The estimated variance
    """
    Ny = len(invK)
    N, Nx = X.shape
    
    mean  = np.zeros((Ny, 1))
    var   = np.zeros((Ny, 1))

    for output in range(Ny):
        ell = hyper[output, 0:Nx]
        sf2 = hyper[output, Nx]**2
        
        
        ks = np.zeros((N, 1))
        for i in range(N):
            ks[i] = cov_kernel(X[i, :].reshape(1,-1), X_new_mean, ell, sf2)
        kss = cov_kernel(X_new_mean, X_new_mean, ell, sf2)

        if alpha is not None:
            mean[output] =  ks.T @ alpha[output]
        else:
            mean[output] = ks.T @ invK[output, :, :] @ Y[:,output]
            
        var[output] = kss - ks.T @ invK[output, :, :] @ ks

        
    return mean, var



def gp_ME(X, inputmean, invK, hyper, alpha, chol):
    """ Gaussian Process function

    # Arguments:
        X: Training data matrix with inputs of size NxNx - Nx number of inputs to the GP.
        inputmean: Mean from the new unseen/tested data of size (1 x Nx)
        invK: Array with the inverse covariance matrices of size (Ny x N x N) - Ny number of outputs from the GP and N number of training points.
        hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn].
    
    # Returns
        mean: The estimated mean vector of size (Ny x 1)
        var:  The estimated variance vector of size (Ny x 1)
    """

    Ny = len(invK)
    X  = ca.MX(X)
    inputmean = ca.MX(inputmean)
    N, Nx = X.shape

    mean  = ca.MX.zeros(Ny, 1)
    var   = ca.MX.zeros(Ny, 1)

    for output in range(Ny):
        ell      = ca.MX(hyper[output, 0:Nx])
        sf2      = ca.MX(hyper[output, Nx]**2)
        alpha_a  = ca.MX(alpha[output])
        L_a      = ca.MX(chol[output])
        
        ks = ca.MX.zeros(N, 1)
        for i in range(N):
            ks[i] = cov_se_ard(X[i, :].T, inputmean, ell, sf2)
        kss = cov_se_ard(inputmean, inputmean, ell, sf2)
        
        v_a = ca.solve(L_a,ks)
        
        mean[output] = ks.T @ alpha_a
        var[output]  = kss - v_a.T @ v_a

    mean_func  = mean
    var_func   = var

    return mean_func, var_func





def gp_TA1(X, Y, inputmean, inputcovar, hyper, invK, alpha):
    """ Gaussian Process with 1-st Taylor Approximation

    This uses a first order taylor for the mean evaluation (a normal GP mean),
    and a first order taylor for estimating the variance.

    # Arguments:
        X: Training data matrix with inputs of size NxNx - Nx number of inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny).
        inputmean: Mean from the new unseen/tested data of size (1 x Nx)
        inputcovar: Covariance from the new unseen/tested data of size (Nx x Nx)
        invK: Array with the inverse covariance matrices of size (Ny x N x N) - Ny number of outputs from the GP and N number of training points.
        hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn].
    
    # Returns
        mean: The estimated mean vector of size (Ny x 1)
        var:  The estimated variance vector of size (Ny x 1)
    """
    X = ca.MX(X)
    Y = ca.MX(Y)
    inputmean = ca.MX(inputmean)
    inputcovar = ca.MX(inputcovar)
   

    Ny         = len(invK)
    N, Nx      = X.shape
    mean       = ca.MX.zeros(Ny, 1)
    var        = ca.MX.zeros(Ny, 1)
    v          = X - ca.repmat(inputmean.T, N, 1)

    var_TA1    = ca.MX.zeros(Ny, 1)
    d_mean     = ca.MX.zeros(Nx, 1)

    for output in range(Ny):
        ell = ca.MX(hyper[output, :Nx])
        w = 1 / ell**2
        sf2 = ca.MX(hyper[output, Nx]**2)
        iK =  ca.MX(invK[output])
        #alpha_a = iK @ Y[:,output]
        alpha_a = ca.MX(alpha[output])

        ks = ca.MX.zeros(N, 1)
        for i in range(N):
            ks[i] = cov_se_ard(X[i, :].T, inputmean, ell, sf2)
        kss = cov_se_ard(inputmean, inputmean, ell, sf2)

        invKks = iK @ ks
        mean[output] = ks.T @ alpha_a
        var[output] = kss - ks.T @ invKks
        d_mean = (w * (v * ks).T) @ alpha_a

        var_TA1[output] = var[output] + d_mean.T @ inputcovar @ d_mean

    return [mean, var_TA1]




def gp_TA2(X, Y, inputmean, inputcovar, hyper, invK, alpha):
    """ Gaussian Process with 2-nd Taylor Approximation

    This uses a first order taylor for the mean evaluation (a normal GP mean),
    and a second order taylor for estimating the variance.

    # Arguments
        X: Training data matrix with inputs of size NxNx - Nx number of inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny).
        inputmean: Mean from the new unseen/tested data of size (1 x Nx)
        inputcovar: Covariance from the new unseen/tested data of size (Nx x Nx)
        invK: Array with the inverse covariance matrices of size (Ny x N x N) - Ny number of outputs from the GP and N number of training points.
        hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn].

    # Returns
        mean: The estimated mean vector of size (Ny x 1)
        var:  The estimated variance vector of size (Ny x 1)
    """
    X = ca.MX(X)
    Y = ca.MX(Y)
    inputmean = ca.MX(inputmean)
    inputcovar = ca.MX(inputcovar)
   
    Ny         = len(invK)
    N, Nx      = X.shape
    mean       = ca.MX.zeros(Ny, 1)
    var        = ca.MX.zeros(Ny, 1)
    v          = X - ca.repmat(inputmean.T, N, 1)

    var_TA2    = ca.MX.zeros(Ny, 1)
    d_mean     = ca.MX.zeros(Nx, 1)
    dd_var     = ca.MX.zeros(Nx, Nx)

    for output in range(Ny):
        ell = ca.MX(hyper[output, :Nx])
        w = 1 / ell**2
        sf2 = ca.MX(hyper[output, Nx]**2)
        iK = ca.MX(invK[output])
        #alpha_a = iK @ Y[:,output]
        alpha_a = ca.MX(alpha[output])
       

        ks = ca.MX.zeros(N, 1)
        for i in range(N):
            ks[i] = cov_se_ard(X[i, :].T, inputmean, ell, sf2)
        kss = cov_se_ard(inputmean, inputmean, ell, sf2)

        invKks = iK @ ks
        mean[output] = ks.T @ alpha_a
        var[output] = kss - ks.T @ invKks
        d_mean = (w * (v * ks).T) @ alpha_a
        
        for i in range(Nx):
            for j in range(Nx):
                dd_var1a = 0
                dd_var1b = 0
                dd_var2 = 0
                dd_var1a += ca.transpose(v[:, i] * ks) @ iK
                dd_var1b += dd_var1a @ (v[:, j] * ks)
                dd_var2 += ca.transpose(v[:, i] * v[:, j] * ks) @ invKks
                dd_var[i,j] += -2 * w[i] * w[j] * (dd_var1b + dd_var2)                         
                
                if i==j:
                    dd_var[i, j] = dd_var[i,j] + 2 * w[i] * (kss - var[output])


        mean_mat = d_mean @ d_mean.T
        var_TA2[output] = var[output] + ca.trace(inputcovar @ (.5* dd_var + mean_mat))

    return [mean, var_TA2]




def gp_EM(X, Y, inputmean, inputcov, hyper, invK, alpha):
    """ Gaussian Process with Exact Moment Matching

    The first and second moments are used to compute the mean and covariance of the
    posterior distribution with a stochastic input distribution. This assumes a
    zero prior mean function and the squared exponential kernel.

    # Arguments
        X: Training data matrix with inputs of size NxNx - Nx number of inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny).
        inputmean: Mean from the new unseen/tested data of size (1 x Nx)
        inputcov: Covariance from the new unseen/tested data of size (Nx x Nx)
        invK: Array with the inverse covariance matrices of size (Ny x N x N) - Ny number of outputs from the GP and N number of training points.
        hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn].

    # Returns
        mean: The estimated mean vector of size (Ny x 1)
        var:  The estimated variance vector of size (Ny x 1).
    """
    
    X = ca.MX(X)
    Y = ca.MX(Y)
    inputmean = ca.MX(inputmean)
    inputcov = ca.MX(inputcov)
    
    hyper = ca.log(hyper)
    Ny     = len(invK)
    N, Nx  = X.shape
    mean  = ca.MX.zeros(Ny, 1)
    beta  = ca.MX.zeros(N, Ny)
    log_k = ca.MX.zeros(N, Ny)
    v     = X - ca.repmat(inputmean.T, N, 1)

    covariance = ca.MX.zeros(Ny, Ny)
    
    
    # mean
    for a in range(Ny):
        ell = ca.MX(hyper[a, :Nx])
        sf2 = ca.MX(hyper[a, Nx])
        iK = ca.MX(invK[a])
        beta = ca.MX(alpha[a])
        
  
        iLambda   = ca.diag(ca.exp(-2 * ell))
        S  = inputcov + ca.diag(ca.exp(2 * ell))
        iS = iLambda @ (ca.MX.eye(Nx) - ca.solve((ca.MX.eye(Nx) + (inputcov @ iLambda)), (inputcov @ iLambda)))
        
        
        T  = v @ iS                       
        c  = ca.exp(2 * sf2) / ca.sqrt(determinant(S)) \
                * ca.exp(ca.sum2(ell))
        q2 = c * ca.exp(-ca.sum2(T * v) * 0.5)
        qb = q2 * beta
        mean[a] = ca.sum1(qb)
       
        t  = ca.repmat(ca.exp(ell), N, 1)
        v1 = v / t
        log_k[:, a] = 2 * sf2 - ca.sum2(v1 * v1) * 0.5

    # covariance                           
    for a in range(Ny):
        ell_a = ca.MX(hyper[a, :Nx])
        sf2_a = ca.MX(hyper[a, Nx])
        iK_a  = ca.MX(invK[a])
        beta_a = ca.MX(alpha[a])
        
        ii = v / ca.repmat(ca.exp(2 * ell_a), N, 1)
        
        for b in range(a + 1):
            ell_b = ca.MX(hyper[b, :Nx])
            sf2_b = ca.MX(hyper[b, Nx])
            iK_b = ca.MX(invK[b])
            beta_b = ca.MX(alpha[b])
                                
            S =  (inputcov @ ca.diag(ca.exp(-2 * ell_a)
                + ca.exp(-2 * ell_b))) + ca.MX.eye(Nx)
            t = 1.0 / ca.sqrt(determinant(S))
            ij = v / ca.repmat(ca.exp(2 * ell_b), N, 1)
            Q = ca.exp(ca.repmat(log_k[:, a], 1, N)
                + ca.repmat(ca.transpose(log_k[:, b]), N, 1)
                + maha(ii, -ij, ca.solve(S, inputcov * 0.5), N))
            A = beta_a @ ca.transpose(beta_b)
            if b == a:
                A = A - iK_a
            A = A * Q
            covariance[a, b] = t * ca.sum2(ca.sum1(A))
            covariance[b, a] = covariance[a, b]
        covariance[a, a] = covariance[a, a] + ca.exp(2 * sf2_a)
    covariance = covariance - (mean @ ca.transpose(mean))
    
    var_EM = ca.diag(covariance)

    return [mean, var_EM]



def determinant(S):
    """ Determinant
    # Arguments
        S:  Covariance 
    """
    return ca.exp(ca.trace(ca.log(S)))

def maha(a1, b1, Q1, N):
    """ Mahalanobis distance """
    
    aQ =  a1 @ Q1
    bQ =  b1 @ Q1 
    K1  = ca.repmat(ca.sum2(aQ * a1), 1, N) \
            + ca.repmat(ca.transpose(ca.sum2(bQ * b1)), N, 1) \
            - 2 * (aQ @ ca.transpose(b1))
    return K1

