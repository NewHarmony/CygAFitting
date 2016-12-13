#!usr/bin/env python

import numpy as np
from scipy.stats import norm
from scipy.stats import halfcauchy
from scipy.special import gammaln as scipy_gammaln
from sherpa.astro.ui import *
import sys

"""
    Script containing the Posterior class for the jet/lobe spectra of Cygnus A
    This class can be used for optimization or MCMC sampling
    
    Written by:
    Martijn de Vries
    martijndevries777@hotmail.com
    
"""




logmin = 1000000000.0

class PoissonPosterior(object):
    
    def __init__(self, l_d, l_m, l_rmf, j_d, j_m, j_rmf, kT_prior, Z_prior, ratios, verbose=True):
        # lobe_data, lobe_models etc are lists of the data/models for each lobe/jet region
        self.lobe_data = l_d
        self.lobe_models = l_m
        self.lobe_rmf = l_rmf
        self.jet_data = j_d
        self.jet_models = j_m
        self.jet_rmf = j_rmf
        self.kT_prior = kT_prior
        self.Z_prior = Z_prior
        self.e_min = 0.5
        self.e_max = 7.0 # energy range to fit
        self.verbose = verbose
        self.regnr = len(l_d) # amount of lobe/jet regions
        self.ratios = np.array(ratios)


        # Check if lengths of the data list, model list and the prior arrays are the same
        if not all(x == self.regnr for x in (len(l_d), len(l_m), len(j_d), len(j_m), self.kT_prior.shape[0], self.Z_prior.shape[0])):
            raise Exception('Error: length of data lists, model lists, and prior arrays must be the same')
        return
    
    def jet_loglikelihood(self, pars_list, neg=False):
        
        # Iterate over individual lobe region log-likelihoods and sum
        jet_res = 0 #
        for i, item in enumerate(self.jet_data):
            
            rmf = self.jet_data[i].get_rmf()
            erange = np.array(rmf.e_min) # need to convert to numpy array to use a double mask
            bounds = (erange > self.e_min) & (erange < self.e_max)
            
            model = self.jet_models[i]
            data = self.jet_data[i]
            pars = pars_list[i,:]
            
            model._set_thawed_pars(pars)
            mean_model = data.eval_model(model)
            
            #stupid hack to make it not go -infinity
            mean_model += np.exp(-30.)
            res = np.nansum(-mean_model[bounds] + data.counts[bounds]*np.log(mean_model[bounds]) -  scipy_gammaln(data.counts[bounds] + 1.))
            
            if not np.isfinite(res):
                res = -logmin
            jet_res += res
    
        if neg:
            return -jet_res
        else:
            return jet_res

    def jet_logprior(self, pars_array, hyper_pars):
        
        # hyperparameters for cauchy-distributed normalization priors
        T_Csig = hyper_pars[0]
        L_Csig = hyper_pars[1]
        J_Csig = hyper_pars[2]

        # Iterate over individual jet region log priors and sum
        logprior = 0
    
        for i, item in enumerate(self.jet_data):
            
            pars = pars_array[i,:] # 2D pars array [i,j], i=reg nr, j=par nr
            
            # Gaussian priors for kT and Z, based on fit result of surrounding region
            kT = pars[0]
            mu_kT = self.kT_prior[i,0]
            sigma_kT = self.kT_prior[i,1]
            p_kT = norm.pdf(kT, loc=mu_kT, scale=sigma_kT)
            if kT < 0.05: p_kT = 0

            Z = pars[1]
            mu_Z = self.Z_prior[i,0]
            sigma_Z = self.Z_prior[i,1]
            p_Z = norm.pdf(Z, loc=mu_Z, scale=sigma_Z)
            if Z < 0.05: p_Z = 0

            # Half-Cauchy  prior
            # see e.g. Polson and Scott (2012)
            T_lognorm = pars[2]
            p_Tnorm = halfcauchy.pdf(T_lognorm*self.ratios[i], scale=T_Csig)

            PL1_lognorm = pars[4]
            p_PL1norm = halfcauchy.pdf(PL1_lognorm*self.ratios[i], scale=L_Csig)
            
            PL2_lognorm = pars[6]
            p_PL2norm = halfcauchy.pdf(PL2_lognorm, scale=J_Csig)
        
            logprior_reg = np.log(p_kT * p_Z * p_Tnorm * p_PL1norm * p_PL2norm)
    
            if not np.isfinite(logprior_reg):
                logprior += -logmin
            else:
                logprior += logprior_reg

        # PI and hyper priors
        PI1 = pars_array[0,3]
        p_PI1 = ((PI1 > 1.2) & (PI1 < 2.5))

        PI2 = pars_array[0,5]
        p_PI2= ((PI2 > 1.2) & (PI2 < 2.5))
        
        p_TCsig = ((T_Csig > 1e-7) & (T_Csig < 1e-3))
        
        p_LCsig = ((L_Csig > 1e-9) & (L_Csig < 1e-4))
        
        p_JCsig = ((J_Csig > 1e-9) & (J_Csig < 1e-4))
        
        logprior_linked = np.log(p_PI1 * p_PI2 * p_TCsig * p_LCsig * p_JCsig)

        if not np.isfinite(logprior_linked):
            logprior += -logmin
        else:
            logprior += logprior_linked

        return logprior

    def lobe_loglikelihood(self, pars_list, neg=False):
        
        # Iterate over individual lobe region log-likelihoods and sum
        lobe_res = 0 #
        for i, item in enumerate(self.lobe_data):
            
            rmf = self.lobe_data[i].get_rmf()
            erange = np.array(rmf.e_min) # need to convert to numpy array to use a double mask
            bounds = (erange > self.e_min) & (erange < self.e_max)
            
            model = self.lobe_models[i]
            data = self.lobe_data[i]
            pars = pars_list[i,:]

            model._set_thawed_pars(pars)
            mean_model = data.eval_model(model)
            
            #stupid hack to make it not go -infinity
            mean_model += np.exp(-20.)
            res = np.nansum(-mean_model[bounds] + data.counts[bounds]*np.log(mean_model[bounds]) -  scipy_gammaln(data.counts[bounds] + 1.))
            
            if not np.isfinite(res):
                res = -logmin
            lobe_res += res

        if neg:
            return -lobe_res
        else:
            return lobe_res

    def lobe_logprior(self, pars_array, hyper_pars):
    
        T_Csig = hyper_pars[0]
        L_Csig = hyper_pars[1]
        
        # Iterate over individual lobe region log-priors and sum
        logprior = 0
        
        for i, item in enumerate(self.lobe_data):
            
            pars = pars_array[i,:] # 2D pars array [i,j], i=reg nr, j=par nr

            # Gaussian priors for kT and Z, based on fit result of surrounding region
            kT = pars[0]
            mu_kT = self.kT_prior[i,0]
            sigma_kT = self.kT_prior[i,1]
            p_kT = norm.pdf(kT, loc=mu_kT, scale=sigma_kT)
            if (kT > 10.0) or (kT < 0): p_kT = 0

            Z = pars[1]
            mu_Z = self.Z_prior[i,0]
            sigma_Z = self.Z_prior[i,1]
            p_Z = norm.pdf(Z, loc=mu_Z, scale=sigma_Z)
            if Z < 0.05 or Z > 1.0: p_Z = 0

            T_lognorm = pars[2]
            p_Tnorm = halfcauchy.pdf(T_lognorm, scale=T_Csig)
            
            PL_lognorm = pars[4]
            p_PLnorm = halfcauchy.pdf(PL_lognorm, scale=L_Csig)
            
            #print 'lobe ', p_kT, p_Z, p_Tnorm, p_PLnorm
            logprior_reg = np.log(p_kT * p_Z * p_Tnorm * p_PLnorm)
            
            if not np.isfinite(logprior_reg):
                logprior += -logmin
            else:
                logprior += logprior_reg

        # PI
        PI = pars_array[0,3]
        p_PI = ((PI > 1.2) & (PI < 2.5))
            
        logprior_linked = np.log(p_PI)
        
        if not np.isfinite(logprior_linked):
            logprior += -logmin
        else:
            logprior += logprior_linked
        
        return logprior

    def logposterior(self, pars_array, neg=False):

        # Recover pars arrays for lobe and jet
        cut = self.regnr*5
        lobe_pars_array = pars_array[:cut].reshape(self.regnr,5)
        jet_pars_array = pars_array[cut:-3].reshape(self.regnr,7)
        hyper_pars = pars_array[-3:]
        
        # lobe component PI
        lobe_pars_array[:,3] = lobe_pars_array[0,3]
        jet_pars_array [:,3] = lobe_pars_array[0,3]
        
        # jet component PI
        jet_pars_array[:,5] = jet_pars_array[0,5]
        
        # Test to link T/Z
        for i in range(self.regnr):
            jet_pars_array[i,0] = lobe_pars_array[i,0]
            jet_pars_array[i,1] = lobe_pars_array[i,1]
        
        # Note: I'm passing parameters that don't do anything to scipy.optimize. I hope that doesn't matter
    
        ll = self.jet_loglikelihood(jet_pars_array)+self.lobe_loglikelihood(lobe_pars_array)
        logprior = self.jet_logprior(jet_pars_array, hyper_pars) + self.lobe_logprior(lobe_pars_array, hyper_pars)
        
        lpost = ll + logprior


        if self.verbose == True:
            print('Lobe + jet LL ' + str(ll))
            print('Lobe + jet prior ' + str(logprior))

        if neg is True:
            if self.verbose == True: print('lpost ' + str(-lpost))
            return -lpost
        else:
            if self.verbose == True: print('lpost' + str(lpost))
            return lpost
    
    def __call__(self, pars_array, neg=False):
        return self.logposterior(pars_array, neg)

