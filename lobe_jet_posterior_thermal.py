#!usr/bin/env python

import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import halfcauchy

import scipy.optimize
from scipy.special import gammaln as scipy_gammaln
from sherpa.astro.ui import *
import sys




logmin = 1000000000.0
class PoissonPosterior(object):
    
    def __init__(self, l_d, l_m, l_rmf, j_d, j_m, j_rmf, kT_prior, Z_prior, verbose=True):
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
            mean_model += np.exp(-20.)
            res = np.nansum(-mean_model[bounds] + data.counts[bounds]*np.log(mean_model[bounds]) -  scipy_gammaln(data.counts[bounds] + 1.))
            
            if not np.isfinite(res):
                res = -logmin
            jet_res += res
        
        
        if neg:
            return -jet_res
        else:
            return jet_res

    def jet_logprior(self, pars_array, hyper_pars):
    
        # Iterate over individual jet region log priors and sum
        logprior = 0
        T_Csig = hyper_pars
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

            T_lognorm = pars[2]
            p_Tnorm = halfcauchy.pdf(T_lognorm, loc=0, scale=T_Csig)
            

            logprior_reg = np.log(p_kT * p_Z * p_Tnorm)
    
            if not np.isfinite(logprior_reg):
                logprior += -logmin
            else:
                logprior += logprior_reg
    
        p_TCsig = ((T_Csig > 1e-7) & (T_Csig < 1e-3))
        logprior_TCsig = np.log(p_TCsig)
        
 
        if not np.isfinite(logprior_TCsig):
            logprior += -logmin
        else:
            logprior += p_TCsig

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

    def lobe_logprior(self, pars_array,  hyper_pars):
    
        # Iterate over individual lobe region log-priors and sum
        logprior = 0
        T_Csig = hyper_pars

        for i, item in enumerate(self.lobe_data):
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
            
            T_lognorm = pars[2]
            p_Tnorm = halfcauchy.pdf(T_lognorm, loc=0, scale=T_Csig)

            
            logprior_reg = np.log(p_kT * p_Z * p_Tnorm)
            
            if not np.isfinite(logprior_reg):
                logprior += -logmin
            else:
                logprior += logprior_reg


        return logprior

    def logposterior(self, pars_array, neg=False):

        # Recover pars arrays for lobe and jet
        cut = self.regnr*3
        lobe_pars_array = pars_array[:cut].reshape(self.regnr,3)
        jet_pars_array = pars_array[cut:-1].reshape(self.regnr,3)
        
        hyper_pars = pars_array[-1]
        
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

