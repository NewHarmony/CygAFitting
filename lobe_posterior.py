#!usr/bin/env python

import logging
import numpy as np
import scipy.stats
import scipy.optimize
from scipy.special import gammaln as scipy_gammaln
from sherpa.astro.ui import *
import sys


logger = logging.getLogger("sherpa") #Logger to suppress sherpa output when loading spectra

# Only lobe regions so far (will expand later)
logmin = 100000000000.0
class PoissonPosterior(object):
    
    def __init__(self, d, m, kT_prior, Z_prior):
        # lobe_data and lobe_models are lists
        self.lobe_data = d
        self.lobe_models = m
        self.kT_prior = kT_prior
        self.Z_prior = Z_prior
        self.regnr = len(d) # amount of lobe regions


        # Check if lengths of the data list, model list and the prior arrays are the same
        if not all(x == self.regnr for x in (len(d), len(m), kT_prior.shape[0], kT_prior.shape[0])):
            sys.exit('Error: length of data list, model list, and prior arrays must be the same')
        return
    
    def loglikelihood(self, pars_list, neg=False):
        
        pars_list[:,3] = pars_list[0,3] # PI shared between all lobe regions

        # Iterate over individual lobe region log-likelihoods and sum
        lobe_res = 0 #
        for i, item in enumerate(self.lobe_data):
            
            model = self.lobe_models[i]
            data = self.lobe_data[i]
            pars = pars_list[i,:]
            
            model._set_thawed_pars(pars)
            mean_model = data.eval_model(model)
            
            #stupid hack to make it not go -infinity
            mean_model += np.exp(-20.)
            res = np.nansum(-mean_model + data.counts*np.log(mean_model) -  scipy_gammaln(data.counts + 1.))
            if not np.isfinite(res):
                res = logmin
            lobe_res += res

        return lobe_res

    def logprior(self, pars_array):
    
        # Iterate over individual lobe region log-priors and sum
        logprior = 0
        for i, item in enumerate(self.lobe_data):
            
            pars = pars_array[i,:] # 2D pars array [i,j], i=reg nr, j=par nr
            print pars
            
            # Gaussian priors for kT and Z, based on fit result of surrounding region
            kT = pars[0]
            mu_kT = kT_prior[i,0]
            sigma_kT = kT_prior[i,1]
            p_kT = gaussian(kT, mu_kT, sigma_kT)
            if i==2: print kT, mu_kT, sigma_kT, p_kT

            Z = pars[1]
            mu_Z = Z_prior[i,0]
            sigma_Z = Z_prior[i,1]
            p_Z = gaussian(kT, mu_Z, sigma_Z)
            
            T_lognorm = np.log10(pars[2])
            #if np.isnan(T_lognorm)==True: T_lognorm = -4
            p_Tnorm = ((T_lognorm > -7) & (T_lognorm < -3))
            
            PL_lognorm = np.log10(pars[4])
            #if  np.isnan(PL_lognorm) == True: PL_lognorm == -5
            p_PLnorm = ((PL_lognorm > -7) & (PL_lognorm <-3))
            
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
        # reshape pars_array into a 2D array because scipy.optimize only accepts 1D array

        pars_array = pars_array.reshape(self.regnr,5) # 5=nr of lobe region thawed params
    
        lpost = self.loglikelihood(pars_array) + self.logprior(pars_array)
        
        if neg is True:
            print 'lpost ', -lpost
            return -lpost
        else:
            print 'lpost', lpost
            return lpost
    
    def __call__(self, pars_array, neg=False):
        return self.logposterior(pars_array, neg)


def gaussian(x, mu, sigma):
    return 1./(np.sqrt(2*np.pi*sigma**2)) * np.exp(- (x - mu)**2/(2*sigma)**2)


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
abs1.nH = 0.31 # Standard for the galactic absorption

set_stat('chi2xspecvar')
set_method('levmar')

# Fit thermal region spectra for temperature/abundance priors
fit_array = [0,1,4,5,8,9]
for i in fit_array:
    fit(i)

#covar(5)

covar(0,1,4,5,8,9)
covar_result = get_covar_results()

# lobe get data and models
lobe_data = []
lobe_models = []

regnr = 3

kT_prior = np.zeros((regnr, 2))
Z_prior = np.zeros((regnr,2))

# build kT_prior and Z_prior matrices (i,0 = mu; i,1 = sigma)
# Simplified version for now: kT_avg = (kT_1 + kT_2)/2
# fkT_avg = fkT_1 + (kT_avg - kT_1)
# Maybe its better to just do a joint fit to the top and bottom thermal region
for i in range(regnr):
    j = i*6
    
    kT_avg = (covar_result.parvals[0+j] + covar_result.parvals[3+j])/2.
    kT_prior[i, 0] = kT_avg
    fkT_avg = covar_result.parmaxes[0+j] + (kT_avg - covar_result.parvals[0+j])
    kT_prior[i, 1] = fkT_avg
    
    Z_avg = (covar_result.parvals[1+j] + covar_result.parvals[4+j])/2.
    Z_prior[i, 0] = Z_avg
    fZ_avg = covar_result.parmaxes[1+j] + (Z_avg - covar_result.parvals[1+j])
    Z_prior[i, 1] =  fZ_avg
    

print kT_prior,
print '_______'
print Z_prior
# Make the data list and model list of the lobe regions
for i in range(regnr):
    lobe_nr = 4*i + 2
    lobe_data.append(get_data(lobe_nr))
    lobe_models.append(get_model(lobe_nr))

lpost = PoissonPosterior(lobe_data, lobe_models, kT_prior, Z_prior)
initial_guess = np.zeros((regnr,5))

for i in range(regnr): # reasonable initial guesses
    initial_guess[i,:] = [6.0, 0.5, 1e-4, 1.6, 1e-5]

# Scipy optimize test
fitmethod = scipy.optimize.minimize
neg = False
results = fitmethod(lpost,x0=initial_guess,method='Nelder-mead', args=(neg,))
print results
#results = results[0:15].reshape(3,5)
#print results[:,0]
#print results[:,1]
#print results[:,2]
#print results[:,3]
#print results[:,4]

#print('Optimum parameter values: ' + str(popt))
#print('Likelihood at optimimum parameter values: ' + str(fopt))
