#!usr/bin/env python

import emcee
import logging
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
import scipy.optimize
from scipy.special import gammaln as scipy_gammaln
from sherpa.astro.ui import *
import sys


logger = logging.getLogger("sherpa") #Logger to suppress sherpa output when loading spectra


## function that makes the scatter plot
def scatter_plot(samples, ndims):
    fig, axes = plt.subplots(ndims,ndims,figsize=(15,15))
    
    ### for one-parameter model, make scatter plot
    if ndims == 1:
        axes.hist(samples, bins=20)
    
    ### for more than one parameter, make matrix plot
    else:
        for i in xrange(ndims): ## x dimension
            for j in xrange(ndims): ## y dimension
                if i == j:
                    axes[i,j].hist(samples[:,i], bins=20)
                else:
                    axes[i,j].scatter(samples[:,i], samples[:,j], color="black")

    return

# Only lobe regions so far (will expand later)
logmin = 1000000000.0
class PoissonPosterior(object):
    
    def __init__(self, d, m, rmf,  kT_prior, Z_prior):
        # lobe_data and lobe_models are lists
        self.lobe_data = d
        self.lobe_models = m
        self.lobe_rmf = rmf
        self.kT_prior = kT_prior
        self.Z_prior = Z_prior
        self.e_min = 0.5
        self.e_max = 7.0 # energy range to fit
        self.regnr = len(d) # amount of lobe regions


        # Check if lengths of the data list, model list and the prior arrays are the same
        if not all(x == self.regnr for x in (len(d), len(m), len(rmf), kT_prior.shape[0], Z_prior.shape[0])):
            raise Exception('Error: length of data list, model list, and prior arrays must be the same')
        return
    
    def loglikelihood(self, pars_list, neg=False):


        pars_list[:,3] = pars_list[0,3] # PI shared between all lobe regions

        # Iterate over individual lobe region log-likelihoods and sum
        lobe_res = 0
        for i, item in enumerate(self.lobe_data):
            
            rmf = lobe_data[i].get_rmf()
            erange = np.array(rmf.e_min) # need to convert to numpy array to use a double mask
            bounds = (erange > self.e_min) & (erange < self.e_max)
            
            model = self.lobe_models[i]
            data = self.lobe_data[i]
            pars = pars_list[i,:]
            
            # assume that normalizations are in log, so we'll exponentiate them:
            pars[2] = np.exp(pars[2])
            pars[4] = np.exp(pars[4])
            
            model._set_thawed_pars(pars)
            mean_model = data.eval_model(model)
            
            #stupid hack to make it not go -infinity
            mean_model += np.exp(-20.)
            res = np.nansum(-mean_model[bounds] + data.counts[bounds]*np.log(mean_model[bounds]) -  scipy_gammaln(data.counts[bounds] + 1.))
            
            # Set the normalizations back to log (for the priors)
            pars[2] = np.log(pars[2])
            pars[4] = np.log(pars[4])
            
            if not np.isfinite(res):
                res = -logmin
            lobe_res += res

        print('log likelihood' + str(lobe_res))
        if neg:
            return -lobe_res
        else:
            return lobe_res

    def logprior(self, pars_array):
    
        # Iterate over individual lobe region log-priors and sum
        logprior = 0
        for i, item in enumerate(self.lobe_data):
            
            pars = pars_array[i,:] # 2D pars array [i,j], i=reg nr, j=par nr
            #print('Pars: ' + str(pars))
            
            # Gaussian priors for kT and Z, based on fit result of surrounding region
            kT = pars[0]
            mu_kT = kT_prior[i,0]
            sigma_kT = kT_prior[i,1]
            p_kT = norm.pdf(kT, loc=mu_kT, scale=sigma_kT)
            if (kT > mu_kT+1.5) or (kT < mu_kT-1.5): p_kT = 0

            Z = pars[1]
            mu_Z = Z_prior[i,0]
            sigma_Z = Z_prior[i,1]
            p_Z = norm.pdf(Z, loc=mu_Z, scale=sigma_Z)
            if Z < 0 or Z >1 : p_Z = 0

            T_lognorm = pars[2]
            p_Tnorm = ((T_lognorm > -15) & (T_lognorm < -3))
            
            PL_lognorm = pars[4]
            p_PLnorm = ((PL_lognorm > -15) & (PL_lognorm <-4))
            
            #print 'Priors:', p_kT, p_Z, p_Tnorm, p_PLnorm
            
            logprior_reg = np.log(p_kT * p_Z * p_Tnorm * p_PLnorm)
            if not np.isfinite(logprior_reg):
                logprior += -logmin
            else:
                logprior += logprior_reg
    
        # PI prior (shared between lobe regions)
        PI = pars_array[0,3]
        p_PI = (( PI > 1.0) & (PI < 2.5))
        
        logprior_PI = np.log(p_PI)

        if not np.isfinite(logprior_PI):
            logprior += -logmin
        else:
            logprior += logprior_PI

        print('Log prior' + str(logprior))
        return logprior

    def logposterior(self, pars_array, neg=False):
        # reshape pars_array into a 2D array because scipy.optimize only accepts 1D array
        
        pars_array = pars_array.reshape(self.regnr,5) # 5=nr of lobe region thawed params
        lpost = self.loglikelihood(pars_array) + self.logprior(pars_array)
        
    
        if neg is True:
            print('lpost ' + str(-lpost))
            return -lpost
        else:
            print('lpost' + str(lpost))
            return lpost
    
    def __call__(self, pars_array, neg=False):
        return self.logposterior(pars_array, neg)


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
        group_counts(i, 20) # I fit these in Sherpa, they are not part of the bayesian model
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

fit_array = [0,1,4,5,8,9,12,13]
for i in fit_array:
    fit(i)

covar(0,1,4,5,8,9) #12,13)
covar_result = get_covar_results()

# lobe get data and models
lobe_data = []
lobe_models = []
lobe_rmf = []

regnr = 3 # hardcoded for now, will change later..


kT_prior = np.zeros((regnr, 2))
Z_prior = np.zeros((regnr,2))

# build kT_prior and Z_prior matrices (i,0 = mu; i,1 = sigma)
# Simplified version for now: kT_avg = (kT_1 + kT_2)/2
# fkT_avg = fkT_1 + abs((kT_avg - kT_1))
# Maybe its better to just do a joint fit to the top and bottom thermal region
for i in range(regnr):
    j = i*6
    
    kT_avg = (covar_result.parvals[0+j] + covar_result.parvals[3+j])/2.
    kT_prior[i, 0] = kT_avg
    fkT_avg = covar_result.parmaxes[0+j] + (np.abs(kT_avg - covar_result.parvals[0+j]))
    kT_prior[i, 1] = fkT_avg/5 # cheap trick to fix temp/Z a bit more

    Z_avg = (covar_result.parvals[1+j] + covar_result.parvals[4+j])/2.
    Z_prior[i, 0] = Z_avg
    fZ_avg = covar_result.parmaxes[1+j] + (np.abs(Z_avg - covar_result.parvals[1+j]))
    Z_prior[i, 1] =  fZ_avg/10



# Make the data list and model list of the lobe regions
for i in range(regnr):
    lobe_nr = 4*i + 2
    lobe_data.append(get_data(lobe_nr))
    lobe_models.append(get_model(lobe_nr))
    lobe_rmf.append(get_rmf(lobe_nr))


lpost = PoissonPosterior(lobe_data, lobe_models, lobe_rmf, kT_prior, Z_prior)


# Best-fit parameters of 3 lobe regions fit
start_pars = np.array([4.93267106, 0.27263125, -8.77750573, 1.22546794, -11.45876729, 6.3437716,   0.50333696, -8.13363911, 1.22546794, -11.64686045, 6.26648673, 0.66538454, -7.86290784, 1.22546794, -13.61277446])


#MCMC

start_cov = np.diag(np.abs(start_pars/100))

nwalkers = 100
niter = 20
ndim = len(start_pars)
burnin = 10

p0 = np.array([np.random.multivariate_normal(start_pars, start_cov) for
               i in range(nwalkers)])

# initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lpost, args=[False], threads=4)

pos, prob, state = sampler.run_mcmc(p0, burnin)
_, _, _ = sampler.run_mcmc(pos, niter, rstate0=state)

flatchain = sampler.flatchain


# Is this a good way to show stuff? The only parameter shared between regions is PI, so I figure that it's most useful to show things per region
plt.figure(1)
plt.title('lobe region 1')
scatter_plot(flatchain[:,[0,1,2,3,4]], 5)
                       
plt.figure(2)
plt.title('lobe region 2')
scatter_plot(flatchain[:,[5,6,7,4,9]], 5)

plt.figure(3)
plt.title('lobe region 3')
scatter_plot(flatchain[:,[10,11,12,4,14]], 5)


                       

plt.show()
