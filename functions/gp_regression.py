#!/usr/bin/env python

####  GPR - Gaussian Process Regression  ####

import time
import numpy as np
import pandas as pd

from gp_functions import gp, gp_ME, gp_TA1, gp_TA2, gp_EM
from gp_optimization import train_gp

class GPR:
    def __init__(self, X, Y, optimizer_opts=None, normalize=True, multistart=1, hyper=None, meta=None):
        """ Optimize GPR model.
        
        # Arguments:
            X:               Training data set with inputs of size NxNx - Nx number of inputs to the GP.
            Y:               Training data set with outputs of size (N x Ny).
            optimizer_opts:  Options of optimization solver (Sequential Least SQuares Programming (SLSQP)). 
            normalize:       Normalization of data.
            multistart:      Number of optimization.
            hyper:           Existing optimize hyperparameters.
        """
        
        X = np.array(X).copy()
        Y = np.array(Y).copy()
        
        self.__X = X
        self.__Y = Y
        self.__Ny = Y.shape[1]
        self.__Nx = X.shape[1]
        self.__N = X.shape[0] 
        self.__normalize = normalize
        
        if meta is not None:
            self.__meanY = np.array(meta['meanY'])
            self.__stdY  = np.array(meta['stdY'])
            self.__meanX = np.array(meta['meanX'])
            self.__stdX = np.array(meta['stdX'])
       

        #Optimize hyperparameters
        if hyper is None:
            self.optimize(X=X, Y=Y, opts=optimizer_opts, multistart=multistart, normalize=normalize)
        else:
            self.__hyper  = np.array(hyper['hyper'])
            self.__invK   = np.array(hyper['invK'])
            self.__alpha  = np.array(hyper['alpha'])
            self.__chol   = np.array(hyper['chol'])
            self.__hyper_length_scales   = np.array(hyper['length_scale'])
            self.__hyper_signal_variance = np.array(hyper['signal_var'])
            self.__hyper_noise_variance  = np.array(hyper['noise_var'])
     

    def optimize(self, X=None, Y=None, opts=None, multistart=1, normalize=True, 
                 warm_start=False):
        
        self.__normalize = normalize

        if normalize and X is not None:
            self.__meanY = np.mean(Y, 0)
            self.__stdY  = np.std(Y, 0)
            self.__meanX = np.mean(X, 0)
            self.__stdX  = np.std(X, 0)

        if X is not None:
            X = np.array(X).copy()
            if normalize and X is not None:
                self.__X = self.standardize(X, self.__meanX, self.__stdX)
            else:
                self.__X = X.copy()
        else:
            X = self.__X.copy()
            
        if Y is not None:
            Y = np.array(Y).copy()
            if normalize and X is not None:
                self.__Y = self.standardize(Y, self.__meanY, self.__stdY)
            else:
                self.__Y = Y.copy()
        else:
            Y = self.__Y.copy()

        if warm_start:
            hyp_init = self.__hyper
        else:
            hyp_init = None
            
    
        opt = train_gp(self.__X, self.__Y, optimizer_opts=opts, multistart=multistart, hyper_init=hyp_init)


        self.__hyper = opt['hyper']
        self.__invK  = opt['invK']
        self.__alpha = opt['alpha']
        self.__chol  = opt['chol']
        self.__hyper_length_scales   = self.__hyper[:, :self.__Nx]
        self.__hyper_signal_variance = self.__hyper[:, self.__Nx]**2
        self.__hyper_noise_variance  = self.__hyper[:, self.__Nx + 1]**2



    
    def validate(self, X_test, Y_test):
        """ Validate GP model with test data """
        
        Y_test = Y_test.copy()
        X_test = X_test.copy()
        
        N, Ny = Y_test.shape
        Nx    = X_test.shape[1]
        
        if self.__normalize:
            Y_test = self.standardize(Y_test, self.__meanY, self.__stdY)
            X_test = self.standardize(X_test, self.__meanX, self.__stdX)

        
        loss = 0
        NLP = 0

        for i in range(N):
            mean, var = gp(self.__X, self.__Y, X_test[i,:].reshape(1,-1),
                           self.__invK, self.__hyper, alpha=self.__alpha)
            
            if self.__normalize:  
                mean = self.inverse_mean(mean, self.__meanY.reshape(-1,1), self.__stdY.reshape(-1,1))
                y_test = self.inverse_mean(Y_test[i, :].reshape(-1,1), self.__meanY.reshape(-1,1), self.__stdY.reshape(-1,1))
            
            loss += (y_test - mean)**2
            
        MSE  =  loss / N
        RMSE =  np.sqrt(MSE) 
      

        print('\n________________________________________')
        print('# Validation of GP model ')
        print('----------------------------------------')
        print('* Num training samples: ' + str(self.__N))
        print('* Num test samples: ' + str(N))
        print('----------------------------------------')
        print('* Root Mean Squared Error: ')
        for i in range(Ny):
            print('\t- State %d: %f' % (i + 1, RMSE[i]))
        print('----------------------------------------')
      
        return np.array(RMSE).flatten()
    
    
    
    
    def gp_ME(self, X_test_mean):
        mean, var = gp_ME(self.__X, X_test_mean,
                                         self.__invK, self.__hyper, self.__alpha, self.__chol)
        return mean, var
    
    def gp_TA1(self, X_test_mean, X_test_covar):
        mean, var = gp_TA1(self.__X, self.__Y, X_test_mean, X_test_covar,
                                         self.__hyper, self.__invK, self.__alpha)
        return mean, var
    
    def gp_TA2(self, X_test_mean, X_test_covar):
        mean, var = gp_TA2(self.__X, self.__Y, X_test_mean, X_test_covar,
                                         self.__hyper, self.__invK, self.__alpha)
        return mean, var
    
    def gp_EM(self, X_test_mean, X_test_covar):
        mean, var = gp_EM(self.__X, self.__Y, X_test_mean, X_test_covar,
                                      self.__hyper, self.__invK, self.__alpha)
        return mean, var
    
       
    
    def predict(self, x, cov=None, gp_method='ME'):
        """ Predict future state

        # Arguments:
            x:           Input vector (Nx x 1)
            cov:         Covariance matrix of input size:
                               1. (Nx x 1) if option=True;
                               2. (Nx x Nx) if option=False;
            gp_method:   Analytical methods for uncertainty propagation:
                              'ME':   Mean Equivalence (normal GP);
                              'TA1':  1st order Tayolor Approximation;
                              'TA2':  2nd order Tayolor Approximation;
                              'EM':   1st and 2nd Expected Moments.
        """
        
        self.__cov = cov
        
        
        if self.__normalize:
            x_s = self.standardize(x, self.__meanX, self.__stdX)
            if self.__cov is not None:
                cov_s = self.standardize_cov(cov, self.__stdX)
        else:
            x_s = x
            if self.__cov is not None:
                cov_s = cov

    
        if gp_method is 'ME':
            mean, cov = self.gp_ME(x_s)
        elif gp_method is 'TA1':
            mean, cov = self.gp_TA1(x_s, cov_s)
        elif gp_method is 'TA2':
            mean, cov = self.gp_TA2(x_s, cov_s)
        elif gp_method is 'EM':
            mean, cov = self.gp_EM(x_s, cov_s)
        else:
            raise NameError('No GP method called: ' + gp_method)
            
        if self.__normalize:
            mean = self.inverse_mean(mean, self.__meanY, self.__stdY)
            cov = self.inverse_var(cov, self.__stdY)
        return mean, cov
    
    
    def gp_(self, X_test_mean):
        mean, var = gp(self.__X, self.__Y, X_test_mean,
                           self.__invK, self.__hyper, alpha=self.__alpha)
        return mean, var
        
        
    def predict_gp_mean(self, x):
        """ Predict future state

        # Arguments:
            x: Input vector (Nx x 1)
        """
        if self.__normalize:
            x_s = self.standardize(x, self.__meanX, self.__stdX)
        else:
            x_s = x
    
        mean, _ = self.gp_(x_s)
        
        if self.__normalize:
            mean = self.inverse_mean(mean, self.__meanY.reshape(-1,1), self.__stdY.reshape(-1,1))
            
        return mean
    
    
    
    
    def standardize(self, X, mean, std):
        """ Mean standardization """
        return (X - mean) / std  

    def inverse_mean(self, Y, mean, std):
        """ Inverse standardization of the mean """
        return (Y * std) + mean

    def standardize_cov(self, cov, std):
        """ Covariance standardization """
        return cov / std**2
    
    def inverse_var(self, var, std):
        """ Inverse standardization of the variance """
        return var * std**2

    
    def get_size(self):
        """ Get the size of the GP model

        # Returns:
                N:  Number of training data
                Ny: Number of outputs
                Nu: Number of control inputs
        """
        return self.__N, self.__Ny, self.__Nx
  
    
    def print_hyper_parameters(self):
        """ Print out all hyperparameters
        """
        print('\n________________________________________')
        print('# Hyper-parameters')
        print('----------------------------------------')
        print('* Num samples:', self.__N)
        print('* Ny:', self.__Ny)
        print('* Nx:', self.__Nx)
        print('* Normalization:', self.__normalize)
        for state in range(self.__Ny):
            print('----------------------------------------')
            print('* Lengthscale: ', state)
            for i in range(self.__Nx):
                print(('-- l{a}: {l}').format(a=i,l=self.__hyper_length_scales[state, i]))
            print('* Signal variance: ', state)
            print('-- sf2:', self.__hyper_signal_variance[state])
            print('* Noise variance: ', state)
            print('-- sn2:', self.__hyper_noise_variance[state])
        print('----------------------------------------')




    def to_dict(self):
        """ Store model data in a dictionary """

        gp_dict = {}
        gp_dict['X'] = self.__X.tolist()
        gp_dict['Y'] = self.__Y.tolist()
        gp_dict['hyper'] = dict(
                    hyper = self.__hyper.tolist(),
                    invK = self.__invK.tolist(),
                    alpha = self.__alpha.tolist(),
                    chol = self.__chol.tolist(),
                    length_scale = self.__hyper_length_scales.tolist(),
                    signal_var = self.__hyper_signal_variance.tolist(),
                    noise_var = self.__hyper_noise_variance.tolist()
                                )
        gp_dict['normalize'] = self.__normalize
        if self.__normalize:
            gp_dict['meta'] = dict(
                        meanY = self.__meanY.tolist(),
                        stdY = self.__stdY.tolist(),
                        meanX = self.__meanX.tolist(),
                        stdX = self.__stdX.tolist()
                                   )
        return gp_dict


    def save_model(self, filename):
        """ Save model to a json file"""
        import json
        output_dict = self.to_dict()
        with open(filename + ".json", "w") as outfile:
            json.dump(output_dict, outfile)


    @classmethod
    def load_model(cls, filename):
        """ Create a new model from file"""
        import json
        with open(filename + ".json") as json_data:
            input_dict = json.load(json_data)
        return cls(**input_dict)

