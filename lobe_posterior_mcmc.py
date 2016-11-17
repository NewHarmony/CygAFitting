#!usr/bin/env python

import corner
import emcee
import logging
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
import scipy.optimize
from scipy.special import gammaln as scipy_gammaln
from sherpa.astro.ui import *
import sys


from lobe_posterior import * # so it uses the exact same class as the fitting routine


logger = logging.getLogger("sherpa") #Logger to suppress sherpa output when loading spectra


# Only lobe regions so far (will expand later)
logmin = 1000000000.0

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
# Simple version for now: kT_avg = (kT_1 + kT_2)/2
# fkT_avg = sqrt(fkT_1**2 + fkT_2**2)
for i in range(regnr):
    j = i*6
    
    kT_avg = (covar_result.parvals[0+j] + covar_result.parvals[3+j])/2.
    kT_prior[i, 0] = kT_avg
    fkT_avg = np.sqrt(covar_result.parmaxes[0+j]**2 + covar_result.parmaxes[3+j]**2)
    kT_prior[i, 1] = fkT_avg/2
    
    Z_avg = (covar_result.parvals[1+j] + covar_result.parvals[4+j])/2.
    Z_prior[i, 0] = Z_avg
    fZ_avg = np.sqrt(covar_result.parmaxes[1+j]**2 + covar_result.parmaxes[4+j]**2)
    Z_prior[i, 1] =  fZ_avg/2




# Make the data list and model list of the lobe regions
for i in range(regnr):
    lobe_nr = 4*i + 2
    lobe_data.append(get_data(lobe_nr))
    lobe_models.append(get_model(lobe_nr))
    lobe_rmf.append(get_rmf(lobe_nr))


lpost = PoissonPosterior(lobe_data, lobe_models, lobe_rmf, kT_prior, Z_prior)


# Best-fit parameters of 3 lobe regions fit
start_pars = np.array([5.9983775, 0.48645429, -8.67181377, 1.39615739, -12.0427719, 5.37033591, 0.52261251, -8.28872141, 1.39615739, -10.82143326, 5.0715555, 0.53349356, -8.30533869, 1.39615739, -10.24779775])


#MCMC

start_cov = np.diag(np.abs(start_pars/100))

nwalkers = 100
niter = 250
ndim = len(start_pars)
burnin = 50

p0 = np.array([np.random.multivariate_normal(start_pars, start_cov) for
               i in range(nwalkers)])

# initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lpost, args=[False], threads=4)

pos, prob, state = sampler.run_mcmc(p0, burnin)
_, _, _ = sampler.run_mcmc(pos, niter, rstate0=state)

flatchain = sampler.flatchain


# Is this a good way to show stuff? The only parameter shared between regions is PI, so I figure that it's most useful to show things per region


fig1 = corner.corner(flatchain[:,[0,1,2,3,4]], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, labels=['kT', 'Z', '$T_{norm}$', 'PI', '$PL_{norm}$'])

fig2 = corner.corner(flatchain[:,[5,6,7,3,9]], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, labels=['kT', 'Z', '$T_{norm}$', 'PI', '$PL_{norm}$'])

fig3 = corner.corner(flatchain[:,[10,11,12,3,14]], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, labels=['kT', 'Z', '$T_{norm}$', 'PI', '$PL_{norm}$'])

plt.show()
