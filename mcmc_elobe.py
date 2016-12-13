#!usr/bin/env python

import corner
import emcee
import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import scipy.optimize
from scipy.special import gammaln as scipy_gammaln
from sherpa.astro.ui import *
import sys

from lobe_jet_posterior import *
from plot_mcmc_results import *


if __name__ == '__main__':

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
    fit_array  = [0,1,4,5,8,9,12,13,16,17]

    for i,reg in enumerate(fit_array):
        fit(reg)
        covar(reg)
        reg_results = get_covar_results()
        reg_parvals = reg_results.parvals
        reg_parmaxes = reg_results.parmaxes
        
        if i == 0:
            covar_parvals = reg_parvals
            covar_parmaxes = reg_parmaxes
        else:
            covar_parvals = np.hstack([covar_parvals, reg_parvals])
            covar_parmaxes = np.hstack([covar_parmaxes, reg_parmaxes])

    regnr = 4

    kT_prior = np.zeros((regnr, 2))
    Z_prior = np.zeros((regnr,2))

    # build kT_prior and Z_prior matrices (i,0 = mu; i,1 = sigma)
    # mu_kT = (kT_1 + kT_2)/2
    # sigma_kT = sqrt(fkT_1**2 + fkT_2**2)
    for i in range(regnr):
        j = i*6
    
        kT_avg = (covar_parvals[0+j] + covar_parvals[3+j])/2.
        kT_prior[i, 0] = kT_avg
        fkT_avg = np.sqrt(covar_parmaxes[0+j]**2 + covar_parmaxes[3+j]**2)
        kT_prior[i, 1] = fkT_avg/2
    
        Z_avg = (covar_parvals[1+j] + covar_parvals[4+j])/2.
        Z_prior[i, 0] = Z_avg
        fZ_avg = np.sqrt(covar_parmaxes[1+j]**2 + covar_parmaxes[4+j]**2)
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

    # Area ratios of the jet/lobe regions, used to compare normalizations
    # The real normalization ratio should be the volume ratio, but area is a first order estimate)
    ratios = [ 0.677524429967, 0.371934604905, 0.388017118402, 0.411978221416]

    lpost = PoissonPosterior(lobe_data, lobe_models, lobe_rmf, jet_data, jet_models, jet_rmf, kT_prior, Z_prior, ratios=ratios,verbose=False)


    # Eastern lobe starting params:
    start_pars =  np.array([  6.44712600e+00,   6.33098494e-01,   1.35344355e-04,
                   1.55534432e+00,   1.39723280e-05,   6.86885245e+00,
                   5.01312423e-01,   3.37112583e-04,   1.55534432e+00,
                   8.30561260e-10,   6.76307809e+00,   7.05894218e-01,
                   3.53192589e-04,   1.55534432e+00,   8.50539269e-06,
                   6.06488610e+00,   6.69974251e-01,   4.10451993e-04,
                   1.55534432e+00,   4.09514678e-08,   6.39500063e+00,
                   6.76039334e-01,   1.02339989e-04,   1.55534432e+00,
                   3.79379707e-06,   1.58065081e+00,   2.13418884e-05,
                   6.89948463e+00,   5.10102877e-01,   1.30553600e-04,
                   1.55534432e+00,   8.83273923e-06,   1.58065081e+00,
                   7.16861306e-06,   6.65570709e+00,   7.10729831e-01,
                   1.59128448e-04,   1.55534432e+00,   8.66605922e-08,
                   1.58065081e+00,   9.48604682e-06,   6.08674022e+00,
                   6.97544124e-01,   1.63609204e-04,   1.55534432e+00,
                   1.39135288e-08,   1.58065081e+00,   9.60352502e-06,
                   1.39860948e-04,   2.56979416e-07,   1.08071912e-05])

    #MCMC
    cov_diag = np.zeros(start_pars.shape)
    cov_diag[:-3] = np.clip(np.abs(start_pars[:-3]/5), 1e-8, 500)
    start_cov = np.diag(cov_diag)


    nwalkers = 200
    niter = 50
    ndim = len(start_pars)
    burnin = 5
  
    p0 = np.array([np.random.multivariate_normal(start_pars, start_cov) for
               i in range(nwalkers)])

    # initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lpost, args=[False], threads=40)

    pos, prob, state = sampler.run_mcmc(p0, burnin)
    sampler.reset()
    _, _, _ = sampler.run_mcmc(pos, niter, rstate0=state)


    flatchain = sampler.flatchain

    acceptance = np.mean(sampler.acceptance_fraction)
    acor = np.mean(sampler.acor)
    print('The ensemble acceptance rate is: ' + str(acceptance))
    print('The auto correlation time is: ' + str(acor))
    L = acceptance*len(flatchain)

    hdr = 'Acceptance Rate: ' + str(acceptance) + '\n Autocorrelation time: ' + str(acor)
    # save sampler
    np.savetxt('MCMC_flatchain_elobe_n' + str(nwalkers) + '_b' + str(burnin) + '_i' + str(niter) + '.txt', flatchain, header=hdr)

    # Plot results from plot_mcmc_results script
    plot_results(flatchain, regnr, lobe='east')

