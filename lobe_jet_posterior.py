#!usr/bin/env python

import logging
import numpy as np
from scipy.stats import norm
import scipy.optimize
from scipy.special import gammaln as scipy_gammaln
from sherpa.astro.ui import *
import sys




logmin = 100000000000.0
class PoissonPosterior(object):
    
    def __init__(self, l_d, l_m, l_rmf, j_d, j_m, j_rmf, kT_prior, Z_prior):
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

        self.regnr = len(l_d) # amount of lobe/jet regions


        # Check if lengths of the data list, model list and the prior arrays are the same
        if not all(x == self.regnr for x in (len(l_d), len(l_m), len(j_d), len(j_m), kT_prior.shape[0], Z_prior.shape[0])):
            sys.exit('Error: length of data lists, model lists, and prior arrays must be the same')
        return
    
    def jet_loglikelihood(self, pars_list, neg=False):
        
        # Iterate over individual lobe region log-likelihoods and sum
        jet_res = 0 #
        for i, item in enumerate(self.jet_data):
            
            rmf = jet_data[i].get_rmf()
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
                res = logmin
            jet_res += res
        
        return jet_res

    def jet_logprior(self, pars_array):
    
        # Iterate over individual jet region log priors and sum
        logprior = 0
        for i, item in enumerate(self.jet_data):
            
            pars = pars_array[i,:] # 2D pars array [i,j], i=reg nr, j=par nr
            
            # Gaussian priors for kT and Z, based on fit result of surrounding region
            kT = pars[0]
            mu_kT = kT_prior[i,0]
            sigma_kT = kT_prior[i,1]
            p_kT = norm.pdf(kT, loc=mu_kT, scale=sigma_kT)
            
            Z = pars[1]
            mu_Z = Z_prior[i,0]
            sigma_Z = Z_prior[i,1]
            p_Z = norm.pdf(Z, loc=mu_Z, scale=sigma_Z)
            
            T_lognorm = np.log10(pars[2])
            p_Tnorm = ((T_lognorm > -7) & (T_lognorm < -3))
            
            PL1_lognorm = np.log10(pars[4])
            p_PL1norm = ((PL1_lognorm > -7) & (PL1_lognorm <-3))
            
            PL2_lognorm = np.log10(pars[6])
            p_PL2norm = ((PL2_lognorm > -7) & (PL2_lognorm <-3))
            
            logprior += np.log(p_kT * p_Z * p_Tnorm * p_PL1norm * p_PL2norm)
        
        # PI1 prior (lobe component)
        PI1 = pars_array[0,3]
        p_PI1 = (( PI1 > 1.0) & (PI1 < 2.5))
        
        # PI2 prior (jet component)
        PI2 = pars_array[0,5]
        p_PI2 = (( PI2 > 1.0) & (PI2 < 2.5))
        
        logprior += np.log(p_PI1 * p_PI2)

        if not np.isfinite(logprior):
            return logmin
        else:
            return logprior

    def lobe_loglikelihood(self, pars_list, neg=False):
        
        # Iterate over individual lobe region log-likelihoods and sum
        lobe_res = 0 #
        for i, item in enumerate(self.lobe_data):
            
            rmf = lobe_data[i].get_rmf()
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
                res = logmin
            lobe_res += res

        return lobe_res

    def lobe_logprior(self, pars_array):
    
        # Iterate over individual lobe region log-priors and sum
        logprior = 0
        for i, item in enumerate(self.lobe_data):
            
            pars = pars_array[i,:] # 2D pars array [i,j], i=reg nr, j=par nr
            print pars
            
            # Gaussian priors for kT and Z, based on fit result of surrounding region
            kT = pars[0]
            mu_kT = kT_prior[i,0]
            sigma_kT = kT_prior[i,1]
            p_kT = norm.pdf(kT, loc=mu_kT, scale=sigma_kT)

            Z = pars[1]
            mu_Z = Z_prior[i,0]
            sigma_Z = Z_prior[i,1]
            p_Z = norm.pdf(Z, loc=mu_Z, scale=sigma_Z)
            
            T_lognorm = np.log10(pars[2])
            p_Tnorm = ((T_lognorm > -15) & (T_lognorm < -3))
            
            PL_lognorm = np.log10(pars[4])
            p_PLnorm = ((PL_lognorm > -15) & (PL_lognorm <-3))
            
            print p_kT, p_Z, p_Tnorm, p_PLnorm
            
            logprior += np.log(p_kT * p_Z * p_Tnorm * p_PLnorm)
            print 'logprior', logprior
        
        # PI prior (shared between lobe regions)
        PI = pars_array[0,3]
        p_PI = (( PI > 1.0) & (PI < 2.5))
        
        logprior += np.log(p_PI)
        
        if not np.isfinite(logprior):
            return logmin
        else:
            return logprior

    def logposterior(self, pars_array, neg=False):

        # Recover pars arrays for lobe and jet
        cut = self.regnr*5
        lobe_pars_array = pars_array[:cut].reshape(self.regnr,5)
        jet_pars_array = pars_array[cut:].reshape(self.regnr,7)
        
        # lobe component PI
        lobe_pars_array[:,3] = lobe_pars_array[0,3]
        jet_pars_array [:,3] = lobe_pars_array[0,3]
        
        # jet component PI
        jet_pars_array[:,5] = jet_pars_array[0,5]
        
        # I could also link the temperature of jet and lobe region here if I want
        # Note: I'm passing parameters that don't do anything to scipy.optimize. I hope that doesn't matter
      
        lpost = self.jet_loglikelihood(jet_pars_array) + self.jet_logprior(jet_pars_array) + self.lobe_loglikelihood(lobe_pars_array) + self.lobe_logprior(lobe_pars_array)
        
        if neg is True:
            print 'lpost ', -lpost
            return -lpost
        else:
            print 'lpost', lpost
            return lpost
    
    def __call__(self, pars_array, neg=False):
        return self.logposterior(pars_array, neg)

logger = logging.getLogger("sherpa") #Logger to suppress sherpa output when loading spectra

# switch off sherpa file-loading output
logger.setLevel(logging.WARN)


# Load in spectra
for i in range(36):
    load_pha(i, 'combined_reg' + str(i) + '_src.pi')
    subtract(i)
    notice(0.5, 7.0)

# switch normal level back on
logger.setLevel(logging.INFO)


# Set appropriate models
for i in range(36):
    
    if (i % 4 == 0) or (i % 4 == 1):  # Northern and southern 'thermal regions' 1-9
        set_source(i, 'xsphabs.abs1 * (xsapec.plsm' + str(i) + ')')
        thaw('plsm' + str(i) + '.Abundanc')
        set_par('plsm' + str(i) + '.redshift', 0.0562)
        group_counts(i, 18) # I fit these in Sherpa, they are not part of the bayesian model
        ignore_id(i, 0.0, 0.52)
        ignore_id(i, 6.98, None)
    
    if i % 4 == 2:      # Lobe regions 1 - 9 (2,6,10,14,18,22,26,30,34)
        set_source(i, 'xsphabs.abs1 * (xsapec.plsm' + str(i) + ' + xspowerlaw.pow' + str(i) + ')' )
        if i == 1: set_par('pow' + str(i) + '.PhoIndex', 1.6, min=1.0, max=2.2)
        set_par('plsm' + str(i) + '.redshift', 0.0562)
        thaw('plsm' + str(i) + '.Abundanc')
    
    if i % 4 == 3:     # Jet regions 1 - 9  (3,7,11,15,19,23,27,31,35)
        set_source(i, 'xsphabs.abs1 * (xsapec.plsm' + str(i) + ' + xspowerlaw.pow' + str(i-1) + '+ xspowerlaw.pow' + str(i) +  ')' )
        if i == 1: set_par('pow' + str(i) + '.PhoIndex', 1.6, min=1.0, max=2.2)
        set_par('plsm' + str(i) + '.redshift', 0.0562)
        thaw('plsm' + str(i) + '.Abundanc')



ignore_id(5, 0.0, 0.53)
ignore_id(5, 6.9, None)
freeze(abs1.nH)
abs1.nH = 0.31 # Standard value for CygA  from HI surveys

set_stat('chi2xspecvar')
set_method('levmar')

# Fit thermal region spectra for temperature/abundance priors
fit_array = [0,1,4,5,8,9]
for i in fit_array:
    fit(i)

covar(0,1,4,5,8,9)
covar_result = get_covar_results()

regnr = 3 # just do 3 regions for now

kT_prior = np.zeros((regnr, 2))
Z_prior = np.zeros((regnr,2))

# build kT_prior and Z_prior matrices (i,0 = mu; i,1 = sigma)
# Simplified version for now: kT_avg = (kT_1 + kT_2)/2
# fkT_avg = fkT_1 + abs(kT_avg - kT_1)
# Maybe its better to just do a joint fit to the top and bottom thermal region
for i in range(regnr):
    j = i*6
    
    kT_avg = (covar_result.parvals[0+j] + covar_result.parvals[3+j])/2.
    kT_prior[i, 0] = kT_avg
    fkT_avg = covar_result.parmaxes[0+j] + (np.abs(kT_avg - covar_result.parvals[0+j]))
    kT_prior[i, 1] = fkT_avg/2
    
    Z_avg = (covar_result.parvals[1+j] + covar_result.parvals[4+j])/2.
    Z_prior[i, 0] = Z_avg
    fZ_avg = covar_result.parmaxes[1+j] + (np.abs(Z_avg - covar_result.parvals[1+j]))
    Z_prior[i, 1] =  fZ_avg/2


# Make the data, model and rmf lists of the jet and lobe regions
lobe_data = []
lobe_models = []
lobe_rmf = []
jet_data = []
jet_models = []
jet_rmf = []

for i in range(regnr):
    
    lobe_nr = 4*i + 2
    lobe_data.append(get_data(lobe_nr))
    lobe_models.append(get_model(lobe_nr))
    lobe_rmf.append(get_rmf(lobe_nr))

    jet_nr = 4*i + 3
    jet_data.append(get_data(jet_nr))
    jet_models.append(get_model(jet_nr))
    jet_rmf.append(get_rmf(jet_nr))


lpost = PoissonPosterior(lobe_data, lobe_models, lobe_rmf, jet_data, jet_models, jet_rmf, kT_prior, Z_prior)

lobe_guess = np.zeros((regnr,5))

for i in range(regnr): # reasonable initial guesses for lobe params
    lobe_guess[i,:] = [6.0, 0.5, 1e-4, 1.6, 1e-5]

jet_guess = np.zeros((regnr, 7))
for i in range(regnr): # reasonable initial guesses for jet params
    jet_guess[i,:] = [6.0, 0.5, 1e-4, 1.6, 1e-6, 1.7, 1e-5]

# flatten everything into one big 1D array for scipy.optimize. The 2D arrays are recovered inside logposterior function
init_guess = np.hstack((lobe_guess.flatten(), jet_guess.flatten()))


# Scipy optimize test
fitmethod = scipy.optimize.minimize
neg = False
print init_guess
results = fitmethod(lpost,x0=init_guess,method='Nelder-mead', args=(neg,))
print results

