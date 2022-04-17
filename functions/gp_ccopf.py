#!/usr/bin/env python

####  GP CC-OPF  ####


import time
import numpy as np
import casadi as ca
from scipy.stats import norm

import matplotlib.pyplot as plt




class GP_CCOPF:
    def __init__(self, gp=None, gp_method=None, predefine_probability_y=None, predefine_probability_u=None,
                 R=None, R2=None, ulb=None, uub=None, xlb=None, xub=None, solver_opts=None, normalize=False,
                 X=None, Y=None, Xd=None, alpha=None, per_load=None, per_rs=None, n_load=None):
        """ Initialize and build the GP CC-OPF solver
         
        # Arguments:
            gp:           Trained GP model.
            gp_method:    Method of propagating the uncertainty.
                          Possible options:
                          'ME':  Mean equivalent approximation;
                          'TA1': 1-st order Taylor approximation;
                          'TA2': 2-nd order Taylor approximation;
                          'EM':  Exact Moment Matching.
                          
            predefine_probability_y: Measure of output variables - how far from the contrain that is allowed,
                                     P(X in constrained set) > predefine_probability_y,
                                     predefine_probability_y= 1 - probability of violation.
                                     
            predefine_probability_u: Measure of control variables - how far from the contrain that is allowed,
                                     P(X in constrained set) > predefine_probability_u,
                                     predefine_probability_u= 1 - probability of violation.
            
            R:   Diagonal input squared cost matrix.
            R2:  Diagonal input linear cost matrix.
            ulb: Lower boundry input.
            uub: Upper boundry input.
            xlb: Lower boundry state.
            xub: Upper boundry state.
            solver_opts:  Additional options to pass to the NLP solver
            normalize:    Normalization of input and output data.
            
            X:  Training input data.
            Y:  Training output data.
            Xd: Unseen input data.
            alpha: Participation factors.
            per_load: variation percentage of forcasted loads.
            per_rs:   variation percentage of forecasted renewable sources.
            n_load: Number of loads. 
           
        """
            
        build_solver_time = -time.time()
        
        N, Ny, Nu = gp.get_size()
        Nd = Xd.shape[1]
        Nu = Nu - Nd
        Nx = Nu + Nd
        
        N_load = n_load
        N_rs   = Nd - n_load 
        
        self.__N  = N
        self.__Ny = Ny
        self.__Nx = Nx
        self.__Nu = Nu
        self.__Nd = Nd 
        
        self.__X       = X    
        self.__Y       = Y
        self.__Xd      = Xd
        self.__alpha   = alpha
        
        self.__gp        = gp
        self.__normalize = normalize
        
        
        K, std_trace, std = self.input_covar(Xd, alpha, per_load, per_rs, Nu, N_load, N_rs)
        self.__Xd_std_trace = std_trace
        self.__Xd_std = std

        opti = ca.Opti()
        

        u     = opti.variable(Nu)
        alpha = opti.variable(Nu)
        mean  = opti.variable(Ny)
        var   = opti.variable(Ny)
        
        x_d   = opti.parameter(Nd)
        cov_d = opti.parameter(Nu+Nd, Nu+Nd)
        opti.set_value(x_d, Xd)
        opti.set_value(cov_d, K)
        
        
        x_t = ca.vertcat(u, x_d)
        cov_t = cov_d
        
        mean_next_pred, var_next_pred = gp.predict(x_t, cov_t, gp_method=gp_method)

        # outputs    
        opti.subject_to([mean-mean_next_pred==0, var-var_next_pred==0, var>=0])
        # alpha
        opti.subject_to([alpha>=0, ca.sum1(alpha) == 1])
        # balance
        opti.subject_to(ca.sum1(u) + ca.sum1(x_d[n_load:]) - ca.sum1(x_d[:n_load]) == 0)
        # probability constraint u
        quantile_u = norm.ppf(predefine_probability_u)
        self.__quantile_u = quantile_u
        opti.subject_to([self.constraint_u_up(u, cov_t, quantile_u, alpha)<=uub, self.constraint_u_down(u, cov_t, quantile_u, alpha)>=ulb])
        # probability constraint x
        quantile_y = norm.ppf(predefine_probability_y)
        self.__quantile_y = quantile_y
        opti.subject_to([self.constraint_x_up(mean, var, quantile_y)<=xub, self.constraint_x_down(mean, var, quantile_y)>=xlb])

        # cost
        opti.minimize(self.cost(u, alpha, cov_d, R, R2))

        # solver
        opts = {'ipopt.print_level' : 3,
                  'ipopt.mu_init' : 0.01,
                  'ipopt.tol' : 1e-5,
                  'ipopt.warm_start_init_point' : 'yes',
                  'ipopt.warm_start_bound_push' : 1e-4,
                  'ipopt.warm_start_bound_frac' : 1e-4,
                  'ipopt.warm_start_slack_bound_frac' : 1e-4,
                  'ipopt.warm_start_slack_bound_push' : 1e-4,
                  'ipopt.warm_start_mult_bound_push' : 1e-4,
                  'ipopt.max_iter' : 150,
                  'ipopt.mu_strategy' : 'adaptive',
                  'print_time' : True,
                  'verbose' : False,
                  'expand' : False}

        
        if solver_opts is not None:
            opts.update(solver_opts)

        opti.solver("ipopt", opts)
        
        build_solver_time += time.time()
        
        
        print('----------------------------------------') 
        print('# Time to build GP CCOPF solver: %f sec' % build_solver_time)
        
        
        self.__opti = opti
        
        self.__u = u
        self.__alpha = alpha
        self.__mean = mean
        self.__var = var
        
        
    def solve(self):
            self.__opti.set_initial(self.__u, self.__X[:,:self.__Nu].mean())
            self.__opti.set_initial(self.__alpha, np.ones(self.__Nu) * 0.5)
            self.__opti.set_initial(self.__mean, self.__Y.mean())
            self.__opti.set_initial(self.__var, np.ones(self.__Ny))
            
            #prediction_time = -time.time()
            sol = self.__opti.solve()
            #prediction_time += time.time()
            
            status          = sol.stats()['return_status']
            num_iter        = sol.stats()["iter_count"]
            prediction_time = sol.stats()["t_proc_total"]
            
            # Print status 
            print('Status:', status)
            print('Number of iteration:', num_iter)
            print('CPU time:', prediction_time)
        
            x_u = sol.value(self.__u)
            x_alpha = sol.value(self.__alpha)
            y_mean = sol.value(self.__mean)
            y_var = sol.value(self.__var)
            
            trace_cov = self.__Xd_std_trace
            
            print('Optimal control:\n', x_u.reshape(-1,1))
            print('Alpha:\n', x_alpha.reshape(-1,1))    
            print('Output mean:\n', y_mean.reshape(-1,1))
            print('Output variance:\n', y_var.reshape(-1,1))
            
            
            self.__x_u         = x_u.reshape(1,-1)
            self.__x_alpha     = x_alpha.reshape(1,-1)
            self.__y_mean      = y_mean.reshape(1,-1)
            self.__y_var       = y_var.reshape(1,-1)
            self.__trace_cov   = trace_cov
            
        
            return self.__x_u , self.__x_alpha, self.__y_mean, self.__y_var, self.__trace_cov, self.__Xd_std
        

    def debug(self):
        return self.__opti.debug.value
    
    def input_covar(self, x, alpha, per_load, per_rs, n_u, n_load, n_rs):
        """ Input Covariance matrix """
        
        # matrix
        x_var = np.zeros((1,n_load + n_rs))
        x_cov = np.zeros((1,n_load + n_rs))
    
        # add values
        x_var[:, :n_load] = (x[:, :n_load]*per_load)**2
        x_var[:, n_load:] = (x[:, n_load:]*per_rs)**2
    
        x_cov[:, :n_load] = (x[:, :n_load]*per_load)**2
        x_cov[:, n_load:] = (-1)*(x[:, n_load:]*per_rs)**2
    
        # create input cov
        K = np.ones((n_u,n_load + n_rs))
        K_cov = K*alpha*x_cov
        
        K_dist =np.diag(x_var.flatten())

        K_control = np.zeros((n_u,n_u))
     
        var_trace = np.sum(x_var)
        K_control = np.diag(alpha.flatten()**2 * var_trace)
        
        for i in range(n_u):
            for j in range(n_u):
                if i<j or i>j:
                    K_control[i,j] = alpha[i] * var_trace * alpha[j]
                
        # concatenate 
        K_control_cov = np.concatenate([K_control, K_cov], axis=1)
        K_cov_dist = np.concatenate([K_cov.T, K_dist], axis=1)
        K = np.concatenate([K_control_cov, K_cov_dist], axis=0)
    
        return K, var_trace, np.sqrt(x_var)


    def cost(self, u, alpha, cov_t, R, R2):
        """ Cost function """
        R = ca.MX(R) 
        R2 = ca.MX(R2)
    
        sqnorm_u = u.T @ R @ u
        linorm_u = R2.T @ u
        trace_u  = alpha.T @ (R*ca.trace(cov_t)) @ alpha
        
        return sqnorm_u + trace_u + linorm_u
     
    
    def constraint_u_up(self, u, cov_t, quantile, alpha):
        """ Build up chance constraint vectors for control inputs """
        
        quantile = ca.MX(quantile)
        
        con = u + alpha * quantile * ca.sqrt(ca.trace(cov_t))
        return con
    
    
    def constraint_u_down(self, u, cov_t, quantile, alpha):
        """ Build up chance constraint vectors for control inputs """
        
        quantile = ca.MX(quantile)
        
        con = u - alpha * quantile * ca.sqrt(ca.trace(cov_t))
        return con
    
    
    def constraint_x_up(self, mean, var, quantile):
        """ Build up chance constraint for system outputs """

        quantile = ca.MX(quantile)
        
        con = mean + quantile * np.sqrt(var)
        return con
    
    def constraint_x_down(self, mean, var, quantile):
        """ Build up chance constraint for system outputs """

        quantile = ca.MX(quantile)
        
        con = mean -  quantile * np.sqrt(var)
        return con
    
    
    def plot(self, x_names=None, u_names=None, vol_nom=345, x_w=15, x_h=30, u_w=7, u_h=15, system='IEEE9'):
        
        numcols = 2
        numrows = int(np.ceil(self.__y_mean.shape[1] / numcols))
        t = 1

        x_names = x_names
        u_names = u_names
        
        if system=='IEEE9':
            fig_x = plt.figure(figsize=(x_w, x_h))
            for i in range(self.__Ny-1):
                ax = fig_x.add_subplot(numrows, numcols, i + 1)
                if i<6:
                    ax.errorbar(t, self.__y_mean[:, i]/vol_nom, yerr= 3 * np.sqrt(self.__y_var[:, i])/vol_nom, marker='.',
                        linestyle='None', color='b', label='Voltage uncertanty prediction')
                    ax.axhline(y=1.1, xmin=0, xmax=10, color='r')
                    ax.axhline(y=0.9, xmin=0, xmax=10, color='r')
                elif i>5 and i<9:
                    ax.errorbar(t, self.__y_mean[:, i], yerr= 3 * np.sqrt(self.__y_var[:, i]), marker='.',
                        linestyle='None', color='b', label='Reactive Power uncertanty predicton')
                    ax.axhline(y=100,  xmin=0, xmax=10, color='r')
                    ax.axhline(y=-100, xmin=0, xmax=10, color='r')
                else:
                    ax.errorbar(t, self.__y_mean[:, i], yerr= 3 * np.sqrt(self.__y_var[:, i]), marker='.',
                        linestyle='None', color='b', label='Active Power uncertanty predicton')
                    ax.axhline(y=200,  xmin=0, xmax=10, color='r')
                    ax.axhline(y=-200, xmin=0, xmax=10, color='r')
                
            
                ax.set_ylabel(x_names[i])
            

            
            fig_u = plt.figure(figsize=(u_w, u_h))
            for i in range(self.__Nu):
                ax = fig_u.add_subplot(self.__Xd.shape[1], 1, i + 1)
                
                
                ax.plot(t, self.__x_u[:, i] , 'k', linewidth = 0.5, linestyle='--')
                ax.errorbar(t, self.__x_u[:, i], yerr= self.__x_alpha[:, i] * self.__quantile_u * np.sqrt(self.__trace_cov), marker='.',
                        linestyle='None', color='b', label='Optimal Control')
           
                if i==0:
                    ax.axhline(y=250, xmin=0, xmax=10, color='r')
                elif i==1:
                    ax.axhline(y=300, xmin=0, xmax=10, color='r')
                elif i==2:
                    ax.axhline(y=270, xmin=0, xmax=10, color='r')
                    
                ax.axhline(y=10, xmin=0, xmax=10, color='r')

                ax.set_ylabel(u_names[i])
                
                
        elif system=='IEEE39':
            fig_x = plt.figure(figsize=(x_w, x_h))
            for i in range(self.__Ny-1):
                ax = fig_x.add_subplot(numrows, numcols, i + 1)
                if i<29:
                    ax.errorbar(t, self.__y_mean[:, i]/vol_nom, yerr= 3 * np.sqrt(self.__y_var[:, i])/vol_nom, marker='.',
                        linestyle='None', color='b', label='Voltage uncertanty prediction')
                    ax.axhline(y=1.1, xmin=0, xmax=1, color='r')
                    ax.axhline(y=0.9, xmin=0, xmax=1, color='r')
                elif i>28 and i<39:
                    ax.errorbar(t, self.__y_mean[:, i], yerr= 3 * np.sqrt(self.__y_var[:, i]), marker='.',
                        linestyle='None', color='b', label='Reactive Power uncertanty predicton')
                    ax.axhline(y=500,  xmin=0, xmax=1, color='r')
                    ax.axhline(y=-500, xmin=0, xmax=1, color='r')
                else:
                    ax.errorbar(t, self.__y_mean[:, i], yerr= 3 * np.sqrt(self.__y_var[:, i]), marker='.',
                        linestyle='None', color='b', label='Active Power uncertanty predicton')
                    ax.axhline(y=2000,  xmin=0, xmax=1, color='r')
                    ax.axhline(y=-2000, xmin=0, xmax=1, color='r')
                
            
                ax.set_ylabel(x_names[i])
            

            
            fig_u = plt.figure(figsize=(u_w, u_h))
            for i in range(self.__Nu):
                ax = fig_u.add_subplot(self.__Xd.shape[1], 1, i + 1)
            
                ax.plot(t, self.__x_u[:, i] , 'k', linewidth = 0.5, linestyle='--')
                ax.errorbar(t, self.__x_u[:, i], yerr= self.__x_alpha[:, i] * self.__quantile_u * np.sqrt(self.__trace_cov), marker='.',
                        linestyle='None', color='b', label='Optimal Control')
           
                ax.axhline(y=10, xmin=0, xmax=1, color='r')
                ax.axhline(y=1000, xmin=0, xmax=1, color='r')
                
                ax.set_ylabel(u_names[i]) 

