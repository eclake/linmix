""" linmix -- A hierarchical Bayesian approach to linear regression with error in both X and Y.
"""

from __future__ import print_function

import numpy as np
import sys
from scipy.stats import multivariate_normal
import time
import math

def nearly_equal(a,b,sig_fig=5):
    return ( a==b or
            int(a*10**sig_fig) == int(b*10**sig_fig)
            )


def task_manager(conn):
    chain = None
    while True:
        message = conn.recv()
        if message['task'] == 'init':
            chain = Chain(**message['init_args'])
            chain.initial_guess()
        elif message['task'] == 'init_chain':
            chain.initialize_chain(message['miniter'])
        elif message['task'] == 'step':
            chain.step(message['niter'])
        elif message['task'] == 'extend':
            chain.extend(message['niter'])
        elif message['task'] == 'fetch':
            conn.send(chain.__dict__[message['key']])
        elif message['task'] == 'kill':
            break
        else:
            raise ValueError("Invalid task")

class Chain(object):
    def __init__(self, xArr, yArr, xsigArr, ysigArr, xycovArr, delta, K, nchains, nGMM_err, pi_err, rng=None):
        self.xArr = np.array(xArr, dtype=float)
        self.yArr = np.array(yArr, dtype=float)
        self.xsigArr = np.array(xsigArr, dtype=float)
        self.ysigArr = np.array(ysigArr, dtype=float)
        self.xvarArr = np.power(np.array(xsigArr, dtype=float),2)
        self.yvarArr = np.power(np.array(ysigArr, dtype=float),2)
        self.xycovArr = np.array(xycovArr, dtype=float)
#        self.x = np.array(xArr[:,0], dtype=float)
#        self.y = np.array(yArr[:,0], dtype=float)
        self.nGMM_err = nGMM_err
        self.pi_err = np.array(pi_err, dtype=float)

        if xsigArr is None:
            print('error, have not coded for this circumstance')
            sys.exit
            self.xsig = np.zeros_like(self.x)
            xycov = np.zeros_like(self.x)
        else:
            self.xsig = np.array(xsigArr[:,0], dtype=float)
        if ysigArr is None:
            print('error, have not coded for this circumstance')
            sys.exit
            self.ysig = np.zeros_like(self.y)
            xycov = np.zeros_like(self.y)
        else:
            self.ysig = np.array(ysigArr[:,0], dtype=float)
        self.wxerr = (self.xsig != 0.0)
        self.wyerr = (self.ysig != 0.0)
        self.werrs = werrs = self.wxerr & self.wyerr

        if xycovArr is None:
            print('error, have not coded for this circumstance')
            sys.exit
            self.xycov = np.zeros_like(self.x)
        else:
            self.xycov = np.array(xycovArr[:,0], dtype=float)

        self.xycorr = np.zeros_like(self.xycov)
        self.xycorr[werrs] = self.xycov[werrs] / (self.xsig[werrs] * self.ysig[werrs])
        self.xycorrArr = np.zeros_like(self.xycovArr)
        self.xycorrArr = self.xycovArr/(self.xsigArr*self.ysigArr)

        self.N = len(self.xArr[:,0])
        self.K = K
        self.nchains = nchains

        self.xvar = self.xsig**2
        self.yvar = self.ysig**2

        if delta is None:
            self.delta = np.ones((self.N), dtype=bool)
        else:
            self.delta = np.array(delta, dtype=bool)

        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        self.initialized = False

    def initial_guess(self):  # Step 1
        # For convenience
#        x = self.x
#        y = self.y
#        xycov = self.xycov
#        xvar = self.xvar
#        yvar = self.yvar
        N = self.N
        K = self.K
        x = np.zeros(N)
        y = np.zeros(N)
        xycov = np.zeros(N)
        xvar = np.zeros(N)
        yvar = np.zeros(N)
        nGMM_err = self.nGMM_err
        #assign x and y as centres of Gaussians with highest scales
        for i in range(N):
            tempIdx = np.argmax(self.pi_err[i,:])
            x[i] = self.xArr[i,tempIdx]
            y[i] = self.yArr[i,tempIdx]
            xycov[i] = self.xycovArr[i,tempIdx]
            xvar[i] = self.xvarArr[i,tempIdx]
            yvar[i] = self.yvarArr[i,tempIdx]
        
        
#        #The following now get assigned when G_err is assigned
#        x = np.zeros(N)
#        y = np.zeros(N)
#        xycov = np.zeros(N)
#        xvar = np.zeros(N)
#        yvar = np.zeros(N)
#        nGMM_err = self.nGMM_err

        pi_err = self.pi_err #the relative amplitudes of the models are fixed
        
#        if nGMM_err == 1:
#            self.G_err = np.ones(N, dtype=int)
#        else:
#            self.G_err = np.zeros((N, nGMM_err), dtype=int)
#            #just start by assigning G_err to the Gaussian with highest amplitude
#            for i in range(N):
#                maxind = np.argmax(pi_err[i])
#                self.G_err[i,maxind] = 1
#                self.x[i] = self.xArr[i,maxind]
#                self.y[i] = self.yArr[i,maxind]
#                self.xsig[i] = self.xsigArr[i,maxind]
#                self.ysig[i] = self.ysigArr[i,maxind]
#                self.xycov[i] = self.xycovArr[i,maxind]
#                self.xvar[i] = self.xsig[i]**2
#                self.yvar[i] = self.ysig[i]**2
#                self.xycorr[i] = self.xycov[i] / (self.xsig[i] * self.ysig[i])
#                #avoid negative sigsqr later on - definitely a bit of a hack :S
#                if nearly_equal(self.xycorr[i], 1., sig_fig=1e-6):
#                    print(i, self.xycorr[i], self.xycov[i])
#                    self.xycov[i] = self.xycov[i]*0.999
#                    self.xycorr[i] = self.xycov[i] / (self.xsig[i] * self.ysig[i])
#                    print(i, self.xycorr[i], self.xycov[i])

#        x = self.x
#        y = self.y
#        xsig = self.xsig
#        ysig = self.ysig
#        xycov = self.xycov
#        xvar = self.xvar
#        yvar = self.yvar
#        xycorr = self.xycorr


        # Use BCES estimator for initial guess of theta = {alpha, beta, sigsqr}
        self.beta = ((np.cov(x, y, ddof=1)[1, 0] - np.mean(xycov))
                     / (np.var(x, ddof=1) - np.mean(xvar)))
        self.alpha = np.mean(y) - self.beta * np.mean(x)
        self.sigsqr = np.var(y, ddof=1) - np.mean(yvar) - self.beta * (np.cov(x, y, ddof=1)[1, 0]
                                                                       - np.mean(xycov))
        self.sigsqr = np.max([self.sigsqr,
                              0.05 * np.var(y - self.alpha - self.beta * x, ddof=1)])

        self.mu0 = np.median(x)
        self.wsqr = np.var(x, ddof=1) - np.median(xvar)
        self.wsqr = np.max([self.wsqr, 0.01*np.var(x, ddof=1)])

        # Now get an MCMC value dispersed around above values
        X = np.ones((N, 2), dtype=float)
        X[:, 1] = x
        Sigma = np.linalg.inv(np.dot(X.T, X)) * self.sigsqr
        coef = self.rng.multivariate_normal([0, 0], Sigma)
        chisqr = self.rng.chisquare(self.nchains)
        self.alpha += coef[0] * np.sqrt(1.0/chisqr)
        self.beta += coef[1] * np.sqrt(1.0/chisqr)
        self.sigsqr *= 0.5 * N / self.rng.chisquare(0.5*N)

        # Now get the values for the mixture parameters, first do prior params
        self.mu0min = min(x)
        self.mu0max = max(x)

        mu0g = np.nan
        while not (mu0g > self.mu0min) & (mu0g < self.mu0max):
            mu0g = self.mu0 + (self.rng.normal(scale=np.sqrt(np.var(x, ddof=1) / N)) /
                               np.sqrt(self.nchains/self.rng.chisquare(self.nchains)))
        self.mu0 = mu0g

        # wsqr is the global scale
        self.wsqr *= 0.5 * N / self.rng.chisquare(0.5 * N)

        self.usqrmax = 1.5 * np.var(x, ddof=1)
        self.usqr = 0.5 * np.var(x, ddof=1)

        self.tausqr = 0.5 * self.wsqr * self.nchains / self.rng.chisquare(self.nchains, size=K)

        self.mu = self.mu0 + self.rng.normal(scale=np.sqrt(self.wsqr), size=K)

        # get initial group proportions and group labels

        pig = np.zeros(self.K, dtype=float)
        if K == 1:
            self.G = np.ones(N, dtype=int)
            self.pi = np.array([1], dtype=float)
        else:
            self.G = np.zeros((N, K), dtype=int)
            for i in range(N):
                minind = np.argmin(abs(x[i] - self.mu))
                pig[minind] += 1
                self.G[i, minind] = 1
            
            self.pi = self.rng.dirichlet(pig+1)

        self.eta = y.copy()
        self.y_ul = y.copy()
        self.xi = x.copy()
        
        self.cens = np.nonzero(np.logical_not(self.delta))[0]

        self.initialized = True

    def update_cens_y(self):  # Step 2
        todo = self.cens[:]
        while len(todo) > 0:
            self.y[todo] = self.rng.normal(loc=self.eta[todo],
                                           scale=np.sqrt(self.yvar[todo]),
                                           size=len(todo))
            todo = np.nonzero(np.logical_not(self.delta) & (self.y > self.y_ul))[0]

    def scale_gaussian_multiplication(self, meanArr, sigsqrArr, meanJoint, sigsqrJoint):
        n = len(meanArr)

        scale = np.log(1/(2*math.pi)**((n-1)/2.)*np.sqrt(sigsqrJoint/np.prod(sigsqrArr,axis=1)))\
                + -0.5*(np.sum(meanArr**2/sigsqrArr,axis=1) - meanJoint**2/sigsqrJoint)
    
        if np.isfinite(np.max(scale)) == False or np.isfinite(np.min(scale)) == False:
            print(scale.shape, scale)
#            print('sigsqrJoint: ', sigsqrJoint[0])
#            print('np.prod(sigsqrArr,axis=0): ', np.prod(sigsqrArr,axis=0)[0])
#            print(1/(2*math.pi)**(n/2.)*np.sqrt(sigsqrJoint/np.prod(sigsqrArr,axis=0))[0])
#            print('np.sum(meanArr**2/sigsqrArr,axis=0): ', np.sum(meanArr**2/sigsqrArr,axis=0)[0])
#            print('meanJoint: ', meanJoint[0])
#            print('meanJoint**2/sigsqrJoint: ', meanJoint[0]**2/sigsqrJoint[0])
#            print(-0.5*(np.sum(meanArr**2/sigsqrArr,axis=0)[0] - meanJoint[0]**2/sigsqrJoint[0]))
#            print(np.exp(-0.5*(np.sum(meanArr**2/sigsqrArr,axis=0)[0] - meanJoint[0]**2/sigsqrJoint[0])))
        return scale

    def sample_from_GMM(self, x, xsig, scale, logScale=False):
        output = np.zeros(self.N)
        if logScale:
            scale = np.exp(scale)
#            #Set all scales == 0 if they are 1000 times smaller than maximum scale
#            #Remember, his is in ln, not log10 ln(1/1000)~-6.9
#            for i in range(self.N):
#                if np.isfinite(np.max(scale[i,:])) == False:
#                    print('**', scale[i,:])
#                scale[i,:] = scale[i,:]-np.max(scale[i,:])
#                tempIdx = np.where(scale[i,:] < - 6.9)[0]
#                temp = np.exp(scale[i,:])
#                if np.isfinite(np.max(np.exp(scale[i,:]))) == False:
#                    print('**', scale[i,:],temp)
#                if len(tempIdx) > 0:
#                    temp[tempIdx] = 0.
#                scale[i,:] = temp
        prob_arr = scale/np.sum(scale,axis=0)
        tempX = np.zeros_like(output)
        tempXsig = np.zeros_like(output)
        for i in range(self.N):
            choose = np.random.multinomial(1,prob_arr[i,:])
            if np.sum(choose) != 1:
                print(prob_arr[i,:], scale[i,:], choose)
            tempIdx = np.where(choose == 1)[0][0]
            tempX[i] = x[tempIdx,i]
            tempXsig[i] = xsig[tempIdx,i]
        output = np.random.normal(loc=0,scale=1)*tempXsig + tempX
        return output

    def update_xi(self):  # Step 3
        wxerr = self.wxerr
        wyerr = self.wyerr

        sigma_xihat_i_sqr = []
        xihat_i = []
        means = np.zeros([self.N,3])
        var = np.zeros_like(means)
        logScale = np.zeros([self.N,self.nGMM_err])
        for i in range(self.nGMM_err):
            # Eqn (58)
            sigma_xihat_ik_sqr = 1.0/(1.0/(self.xvarArr[:,i] * (1.0 - self.xycorrArr[:,i]**2))[:, np.newaxis]
                        + self.beta**2 / self.sigsqr
                        + 1.0/self.tausqr)
            
            # Eqn (57)
            sigma_xihat_i_sqr.append(np.sum(self.G * sigma_xihat_ik_sqr, axis=1))
                                     
            # Eqn (56)
            xihat_xy_i = self.xArr[:,i].copy()
            xihat_xy_i[wyerr] += (self.xycovArr[:,i] / self.yvarArr[:,i] * (self.eta - self.yArr[:,i]))[wyerr]
            # Eqn (55)
            temp = np.array(sigma_xihat_i_sqr[-1])
            xihat_ik = (temp[:, np.newaxis]
                        * ((xihat_xy_i/(self.xvarArr[:,i]
                                        * (1.0 - self.xycorrArr[:,i]**2)))[:, np.newaxis]
                        + (self.beta)*(self.eta[:, np.newaxis] - self.alpha)/self.sigsqr
                        + self.mu/self.tausqr))
            # Eqn (54)
            xihat_i.append(np.sum(self.G * xihat_ik, axis=1))
                    
            #P(xi|x,y,eta)
            means[:,0] = xihat_xy_i
            var[:,0] = self.xvarArr[:,i]*(1.0 - self.xycorrArr[:,i]**2)
#            print('xihat_i,means,var,x, y, xycov,yvar,eta, xycorr, beta, alpha, sigsqr, mu, tausqr,xvar= ', xihat_i[-1][0],',',means[0,i], ',',var[0,i],',',self.xArr[0,i],',', self.yArr[0,i],',', self.xycovArr[0,i],',',self.yvarArr[0,i],',',self.eta[0],',', self.xycorrArr[0,i],',', self.beta,',', self.alpha,',', self.sigsqr,',', np.sum(np.array(self.G)*np.array(self.mu)),',', np.sum(np.array(self.G)*np.array(self.tausqr)),',',self.xvarArr[0,i])
#            sys.exit()

            
        
            #P(xi|theta,eta)
            means[:,1] = (self.eta - self.alpha)/self.beta
            var[:,1] = self.sigsqr/self.beta**2
        
            #P(xi|phi)
            means[:,2] = np.sum(self.G * self.mu, axis=1)
            var[:,2] = np.sum(self.G * self.tausqr, axis=1)
        
            #print("ERROR: Haven't taken account of pi_err (the relative amplitudes of the different Gaussians in the error GMM model)")
            logScale[:,i] = np.log(self.pi_err[:,i])+self.scale_gaussian_multiplication(means, var, xihat_i[-1], sigma_xihat_i_sqr[-1])
        
        sigma_xihat_i_sqr = np.array(sigma_xihat_i_sqr)
        xihat_i = np.array(xihat_i)

        
        #now need to sample from a GMM
        # Eqn (53)
        temp = self.sample_from_GMM(xihat_i, np.sqrt(sigma_xihat_i_sqr), logScale, logScale=True)
        self.xi[wyerr] = temp[wyerr]


    def update_eta(self):  # Step 4
        wxerr = self.wxerr
        wyerr = self.wyerr
        
        means = np.zeros([self.N,2])
        var = np.zeros_like(means)
        logScale = np.zeros([self.N,self.nGMM_err])
        sigma_etahat_i_sqr = []
        etahat_i = []
        for i in range(self.nGMM_err):
            #means[0,:] =

            etaxyvar = self.yvarArr[:,i] * (1.0 - self.xycorrArr[:,i]**2)
            etaxy = self.yArr[:,i].copy()
            etaxy[wxerr] += (self.xycovArr[:,i] / self.xvarArr[:,i] * (self.xi - self.xArr[:,i]))[wxerr]

            # Eqn (68)
            sigma_etahat_i_sqr.append(1.0/(1.0/etaxyvar + 1.0/self.sigsqr))
            # Eqn (67)
            etahat_i.append((sigma_etahat_i_sqr[-1] * (etaxy / etaxyvar
                        + (self.alpha + self.beta * self.xi) / self.sigsqr)))
        
            #P(eta|x,y,xi)
            means[:,0] = etaxy
            var[:,0] = etaxyvar
        
            #P(eta|theta,xi)
            means[:,1] = self.alpha+self.beta*self.xi
            var[:,1] = self.sigsqr
        
            logScale[:,i] = np.log(self.pi_err[:,i])+self.scale_gaussian_multiplication(means, var, etahat_i[-1], sigma_etahat_i_sqr[-1])
            #print(logScale)
            #sys.exit()
        

        sigma_etahat_i_sqr = np.array(sigma_etahat_i_sqr)
        etahat_i = np.array(etahat_i)
        
#        for i in range(self.N):
#            print(i, sigma_etahat_i_sqr[:,i])
#            print(i, logScale[:,i])
#            print(i, logScale[:,i]-np.min(logScale[:,i]))
#        self.exit()

        #now need to sample from a GMM
        # Eqn (66)
        temp = self.sample_from_GMM(etahat_i, np.sqrt(sigma_etahat_i_sqr), logScale, logScale = True)
        self.eta[wyerr] = temp[wyerr]
    
    

    

    def update_G(self):  # Step 5
        # Eqn (74)
        #print('self.tausqr: ', self.tausqr)
        #print(1.0/np.sqrt(2*np.pi*self.tausqr))
        #print('self.xi: ', self.xi[:,np.newaxis])
        #print('self.mu: ', self.mu)
        #print(np.exp(-0.5 * (self.xi[:, np.newaxis] - self.mu)**2))
        piNp = self.pi * (1.0/np.sqrt(2.0*np.pi*self.tausqr)
                          * np.exp(-0.5 * (self.xi[:, np.newaxis] - self.mu)**2 / self.tausqr))
                          
        #print(np.sum(piNp, axis=1))
        q_ki = piNp / np.sum(piNp, axis=1)[:, np.newaxis]
        # Eqn (73)
        for i in range(self.N):
            #print 'self.rng.multinomial(1, q_ki[i]): ', self.rng.multinomial(1, q_ki[i])
            self.G[i] = self.rng.multinomial(1, q_ki[i])


    def update_alpha_beta(self):  # Step 6
        X = np.ones((self.N, 2), dtype=float)
        X[:, 1] = self.xi
        # Eqn (77)
        XTXinv = np.linalg.inv(np.dot(X.T, X))
        Sigma_chat = XTXinv * self.sigsqr
        # Eqn (76)
        chat = np.dot(np.dot(XTXinv, X.T), self.eta)
        # Eqn (75)
#        print('update_alpha_beta: ', self.ichain, chat, Sigma_chat)
#        print('self.xi: ', self.xi)
#        print('XTXinv: ', XTXinv)
#        print('self.sigsqr: ', self.sigsqr)
#        print('self.eta: ', self.eta)
        self.alpha, self.beta = self.rng.multivariate_normal(chat, Sigma_chat)

    def update_sigsqr(self):  # Step 7
        # Eqn (80)
        ssqr = 1.0/(self.N-2) * np.sum((self.eta - self.alpha - self.beta * self.xi)**2)
        # Eqn (79)
        nu = self.N - 2
        # Eqn (78)
        self.sigsqr = nu * ssqr / self.rng.chisquare(nu)

    def update_pi(self):  # Step 8
        # Eqn (82)
        self.nk = np.sum(self.G, axis=0)
        # Eqn (81)
        self.pi = self.rng.dirichlet(self.nk+1)

    def update_mu(self):  # Step 9
        Gsum = np.sum(self.G * self.xi[:, np.newaxis], axis=0)
        for k in range(self.K):
            if self.nk[k] != 0:
                # Eqn (86)
                Sigma_muhat_k = 1.0/(1.0/self.usqr + self.nk[k]/self.tausqr[k])
                # Eqn (85)
                xibar_k = 1.0/self.nk[k] * Gsum[k]
                # Eqn (84)
                muhat_k = Sigma_muhat_k * (self.mu0/self.usqr + self.nk[k]/self.tausqr[k]*xibar_k)
                # Eqn (83)
                self.mu[k] = self.rng.normal(loc=muhat_k, scale=np.sqrt(Sigma_muhat_k))
            else:
                self.mu[k] = self.rng.normal(loc=self.mu0, scale=np.sqrt(self.usqr))

    def update_tausqr(self):  # Step 10
        # Eqn (88)
        nu_k = self.nk + 1
        # Eqn (89)
        tk_sqr = 1.0/nu_k * (self.wsqr + np.sum(self.G*(self.xi[:, np.newaxis]-self.mu)**2, axis=0))
        # Eqn (87)
        self.tausqr = tk_sqr * nu_k / self.rng.chisquare(nu_k, size=self.K)

    def update_mu0(self):  # Step 11
        # Eqn (94)
        mubar = np.mean(self.mu)
        # Eqn (93)
        self.mu0 = self.rng.normal(loc=mubar, scale=np.sqrt(self.usqr/self.K))

    def update_usqr(self):  # Step 12
        # Eqn (96)
        nu_u = self.K + 1
        # Eqn (97)
        usqrhat = 1.0/nu_u * (self.wsqr + np.sum((self.mu - self.mu0)**2))
        usqr = np.inf
        while not usqr <= self.usqrmax:
            usqr = usqrhat * nu_u / self.rng.chisquare(nu_u)
        self.usqr = usqr

    def update_wsqr(self):  # Step 13
        # Eqn (102)
        a = 0.5 * (self.K + 3)
        # Eqn (103)
        b = 0.5 * (1.0/self.usqr + np.sum(1.0/self.tausqr))
        # Eqn (101)
        self.wsqr = self.rng.gamma(a, 1.0/b)

    def initialize_chain(self, chain_length):
        self.chain_dtype = [('alpha', float),
                            ('beta', float),
                            ('sigsqr', float),
                            ('xi', (float, self.N)),
                            ('eta', (float, self.N)),
                            ('pi', (float, self.K)),
                            ('mu', (float, self.K)),
                            ('tausqr', (float, self.K)),
                            ('mu0', float),
                            ('usqr', float),
                            ('wsqr', float),
                            ('ximean', float),
                            ('xisig', float),
                            ('corr', float)]
        self.chain = np.empty((chain_length,), dtype=self.chain_dtype)
        self.ichain = 0

    def extend(self, length):
        extension = np.empty((length), dtype=self.chain_dtype)
        self.chain = np.hstack((self.chain, extension))

    def update_chain(self):
        self.chain['alpha'][self.ichain] = self.alpha
        self.chain['beta'][self.ichain] = self.beta
        self.chain['sigsqr'][self.ichain] = self.sigsqr
        self.chain['pi'][self.ichain] = self.pi
        self.chain['mu'][self.ichain] = self.mu
        self.chain['tausqr'][self.ichain] = self.tausqr
        self.chain['mu0'][self.ichain] = self.mu0
        self.chain['xi'][self.ichain] = self.xi
        self.chain['eta'][self.ichain] = self.eta
        self.chain['usqr'][self.ichain] = self.usqr
        self.chain['wsqr'][self.ichain] = self.wsqr
        ximean = np.sum(self.pi * self.mu)
        self.chain['ximean'][self.ichain] = ximean
        xisig = np.sqrt(np.sum(self.pi * (self.tausqr + self.mu**2)) - ximean**2)
        self.chain['xisig'][self.ichain] = xisig
        self.chain['corr'][self.ichain] = self.beta * xisig / np.sqrt(self.beta**2 * xisig**2
                                                                      + self.sigsqr)
        self.ichain += 1

    def step(self, niter):
        for i in range(niter):
            self.update_cens_y()
            old_settings = np.seterr(divide='ignore', invalid='ignore')
            self.update_xi()
            self.update_eta()
            np.seterr(**old_settings)
            self.update_G()
            self.update_alpha_beta()
            self.update_sigsqr()
            self.update_pi()
            self.update_mu()
            self.update_tausqr()
            self.update_mu0()
            self.update_usqr()
            self.update_wsqr()
            self.update_chain()


class LinMix(object):
    """ A class to perform linear regression of `y` on `x` when there are measurement errors in
    both variables.  The regression assumes:

    eta = alpha + beta * xi + epsilon

    x = xi + xerr

    y = eta + yerr

    Here, `alpha` and `beta` are the regression coefficients, `epsilon` is the intrinsic random
    scatter about the regression, `xerr` is the measurement error in `x`, and `yerr` is the
    measurement error in `y`.  `epsilon` is assumed to be normally-distributed with mean zero and
    variance `sigsqr`.  `xerr` and `yerr` are assumed to be normally-distributed with means equal
    to zero, variances `xsig`^2 and `ysig`^2, respectively, and covariance `xycov`. The
    distribution of `xi` is modelled as a mixture of normals, with group proportions `pi`, means
    `mu`, and variances `tausqr`.

    Args:
        x(array_like): The observed independent variable. - this is now of dimensions [N, nGMM_err]
        y(array_like): The observed dependent variable. - this is now of dimensions [N, nGMM_err]
        xsig(array_like): 1-sigma measurement errors in x. - this is now of dimensions [N, nGMM_err]
        ysig(array_like): 1-sigma measurement errors in y. - this is now of dimensions [N, nGMM_err]
        xycov(array_like): Covariance between the measurement errors in x and y. - this is now of dimensions [N, nGMM_err]
        delta(array_like): Array indicating whether a data point is censored (i.e., not detected),
            or not.  If delta[i] == 1, then the ith source is detected.  If delta[i] == 0, then
            the ith source is not detected and y[i] will be interpreted as an upper limit.  Note
            that if there are censored data points, then the maximum-likelihood estimate
            (alpha, beta, sigsqr) is not valid.  By default, all data points are assumed to be
            detected.
        K(int): The number of Gaussians to use in the mixture model for the distribution of xi.
        nchains(int): The number of Monte Carlo Markov Chains to instantiate.
        parallelize(bool): Use a separate thread for each chain.  Only makes sense for nchains > 1.
        seed(int): Random seed.  If `None`, then get seed from np.random.randint().
        nGMM_err(int): The number of Gaussians to use in the mixture models of the measurement uncertainties.
        pi_err(array_like): The relative amplitudes of the mixture model Gaussians

    Attributes:
        nchains(int): The number of instantiated MCMCs.
        chain(numpy recarray): The concatenated MCMCs themselves.  Actually, only the concatenation
            of the last half of each chain is stored here after convergence is reached.  The
            recarray has the following columns:
                - alpha(float): The regression intercept.
                - beta(float): The regression slope.
                - sigsqr(float): The regression intrinsic scatter.
                - pi(array_like): The mixture model component fractions.
                - mu(array_like): The mixture model component means.
                - tausqr(array_like): The mixture model component variances.
                - mu0(float): The hyperparameter describing the prior variance of the distribution
                    of mixture means.
                - usqr(float): The hyperparameter describing the prior variance of the distribution
                    of mixture variances.
                - wsqr(float): The hyperparameter describing the typical scale for the prior on
                    `usqr` and `tausqr`.
                - ximean(float): The mean of the distribution for the independent latent variable
                    `xi`.
                - xisig(float): The standard deviation of the distribution for the independent
                    latent variable `xi`.
                - corr(float): The linear correlation coefficient between the latent dependent and
                    independent variables `xi` and `eta`.
    """
    def __init__(self, xArr, yArr, xsigArr=None, ysigArr=None, xycovArr=None, delta=None, K=3,
                 nchains=4, parallelize=True, seed=None, nGMM_err=1, pi_err=None):
        self.nchains = nchains
        self.parallelize = parallelize

        if seed is None:
            seed = np.random.randint(2**32-1)

        if pi_err is None:
            pi_err = np.ones(len(x))
        if self.parallelize:
            # Will place 1 chain in 1 thread.
            from multiprocessing import Process, Pipe
            # Create a pipe for each thread.
            self.pipes = []
            slave_pipes = []
            for i in range(self.nchains):
                master_pipe, slave_pipe = Pipe()
                self.pipes.append(master_pipe)
                slave_pipes.append(slave_pipe)

            # Create chain pool.
            self.pool = []
            for sp in slave_pipes:
                self.pool.append(Process(target=task_manager, args=(sp,)))
                self.pool[-1].start()

            init_kwargs0 = {'xArr':xArr,
                            'yArr':yArr,
                            'xsigArr':xsigArr,
                            'ysigArr':ysigArr,
                            'xycovArr':xycovArr,
                            'delta':delta,
                            'K':K,
                            'nchains':self.nchains,
                            'nGMM_err':nGMM_err,
                            'pi_err':pi_err}
            for i, p in enumerate(self.pipes):
                init_kwargs = init_kwargs0.copy()
                init_kwargs['rng'] = np.random.RandomState(seed+i)
                p.send({'task':'init',
                        'init_args':init_kwargs})
        else:
            self._chains = []
            for i in range(self.nchains):
                self._chains.append(Chain(xArr, yArr, xsigArr, ysigArr, xycovArr, delta, K, nGMM_err, pi_err, self.nchains))
                self._chains[-1].initial_guess()

    def _get_psi(self):
        if self.parallelize:
            for p in self.pipes:
                p.send({'task':'fetch',
                        'key':'chain'})
            chains = [p.recv() for p in self.pipes]
            self.pipes[0].send({'task':'fetch',
                                'key':'ichain'})
            ndraw = int(self.pipes[0].recv()/2)
        else:
            chains = [c.chain for c in self._chains]
            ndraw = int(self._chains[0].ichain/2)
        psi = np.empty((ndraw, self.nchains, 6), dtype=float)
        psi[:, :, 0] = np.vstack([c['alpha'][0:ndraw] for c in chains]).T
        beta = np.vstack([c['beta'][0:ndraw] for c in chains]).T
        psi[:, :, 1] = beta
        sigsqr = np.vstack([c['sigsqr'][0:ndraw] for c in chains]).T
        psi[:, :, 2] = np.log(sigsqr)
        ximean = np.vstack([np.sum(c['pi'][0:ndraw] * c['mu'][0:ndraw], axis=1)
                            for c in chains]).T
        psi[:, :, 3] = ximean
        xivar = np.vstack([np.sum(c['pi'][0:ndraw] * (c['tausqr'][0:ndraw] + c['mu'][0:ndraw]**2),
                                  axis=1)
                           for c in chains]).T - ximean**2
        psi[:, :, 4] = xivar
        psi[:, :, 5] = np.arctanh(beta * np.sqrt(xivar / (beta**2 * xivar + sigsqr)))
        return psi

    def _get_Rhat(self):
        psi = self._get_psi()
        ndraw = psi.shape[0]
        psibarj = np.sum(psi, axis=0)/ndraw
        psibar = np.mean(psibarj, axis=0)
        sjsqr = np.sum((psi-psibarj)**2 / (ndraw-1.0), axis=(0, 1))
        Bvar = ndraw / (self.nchains-1.0) * np.sum((psibarj-psibar)**2, axis=0)
        Wvar = sjsqr / self.nchains
        varplus = (1.0 - 1.0 / ndraw) * Wvar + Bvar / ndraw
        Rhat = np.sqrt(varplus / Wvar)
        return Rhat

    def _initialize_chains(self, miniter):
        if self.parallelize:
            for p in self.pipes:
                p.send({'task':'init_chain',
                        'miniter':miniter})
        else:
            for c in self._chains:
                c.initialize_chain(miniter)

    def _step(self, niter):
        if self.parallelize:
            for p in self.pipes:
                p.send({'task':'step',
                        'niter':niter})
        else:
            for c in self._chains:
                c.step(niter)

    def _extend(self, niter):
        if self.parallelize:
            for p in self.pipes:
                p.send({'task':'extend',
                        'niter':niter})
        else:
            for c in self._chains:
                c.extend(niter)

    def _build_chain(self, ikeep):
        if self.parallelize:
            for p in self.pipes:
                p.send({'task':'fetch',
                        'key':'chain'})
            self.chain = np.hstack([p.recv()[ikeep:] for p in self.pipes])
        else:
            self.chain = np.hstack([c.chain[ikeep:] for c in self._chains])

    def run_mcmc(self, miniter=5000, maxiter=100000, silent=False):
        """ Run the Markov Chain Monte Carlo for the LinMix object.

        Bayesian inference is employed, and a Markov chain containing random draws from the
        posterior is developed.  Convergence of the MCMC to the posterior is monitored using the
        potential scale reduction factor (RHAT, Gelman et al. 2004). In general, when RHAT < 1.1
        then approximate convergence is reached.  After convergence is reached, the second halves
        of all chains are concatenated and stored in the `.chain` attribute as a numpy recarray.

        Args:
            miniter(int): The minimum number of iterations to use.
            maxiter(int): The maximum number of iterations to use.
            silent(bool): If true, then suppress updates during sampling.
        """
        print("run_mcmc")
        checkiter = 100
        self._initialize_chains(miniter)
        for i in range(0, miniter, checkiter):
            self._step(checkiter)
            Rhat = self._get_Rhat()

            if not silent:
                print()
                print("Iteration: ", i+checkiter)
                print ("Rhat values for alpha, beta, log(sigma^2)"
                       ", mean(xi), log(var(xi)), atanh(corr(xi, eta)):")
                print(Rhat)

        i += checkiter
        while not np.all(Rhat < 1.1) and (i < maxiter):
            self._extend(checkiter)
            self._step(checkiter)

            Rhat = self._get_Rhat()
            if not silent:
                print()
                print("Iteration: ", i+checkiter)
                print ("Rhat values for alpha, beta, log(sigma^2)"
                       ", mean(xi), log(var(xi)), atanh(corr(xi, eta)):")
                print(Rhat)
                i += checkiter

        # Throw away first half of each chain
        self._build_chain(int(i/2))
        # Clean up threads
        if self.parallelize:
            for p in self.pipes:
                p.send({'task':'kill'})
