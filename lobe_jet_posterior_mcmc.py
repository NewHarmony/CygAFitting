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
        fkT_avg = np.sqrt(covar_result.parmaxes[0+j]**2 + covar_result.parmaxes[3+j]**2)
        kT_prior[i, 1] = fkT_avg/2
    
        Z_avg = (covar_result.parvals[1+j] + covar_result.parvals[4+j])/2.
        Z_prior[i, 0] = Z_avg
        fZ_avg = np.sqrt(covar_result.parmaxes[1+j]**2 + covar_result.parmaxes[4+j]**2)
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

    start_pars = np.array([6.91860961  , 0.78720923 , -8.99997726 ,  1.75435284, -10.86896919 ,\
                       7.95266117 ,  0.50502138 , -8.24966977  , 1.75435284 ,-10.76552872 ,\
                       6.6920166 ,   0.33556179  ,-7.90850756  , 1.75435284, -11.16897997,\
                       6.91860961 ,  0.78720923 , -8.97907755 ,  1.75435284 , -13.63377279, 1.78663358, -10.83551357 ,\
                       7.95266117,   0.50502138 , -8.99780965,   1.75435284 , -10.91281741  , 1.78663358 , -13.11711422 , \
                       6.6920166,    0.33556179 , -8.84984251  , 1.75435284 , -13.43136407 ,  1.78663358,  -10.93006715])

    #MCMC

    start_cov = np.diag(np.abs(start_pars/3))

    nwalkers = 200
    niter = 1000
    ndim = len(start_pars)
    burnin = 500

    p0 = np.array([np.random.multivariate_normal(start_pars, start_cov) for
               i in range(nwalkers)])

    # initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lpost, args=[False], threads=4)

    pos, prob, state = sampler.run_mcmc(p0, burnin)
    _, _, _ = sampler.run_mcmc(pos, niter, rstate0=state)


    flatchain = sampler.flatchain

    acceptance = np.mean(sampler.acceptance_fraction)
    print('The ensemble acceptance rate is: ' + str(acceptance))
    L = acceptance*len(flatchain)

    # save sampler
    np.savetxt('lobe_jet_mcmc_flatchain.txt', flatchain)

    for i in range(3):
        plt.figure()
        ax1 = plt.subplot('511')
        ax1.plot(flatchain[:,5*i+0])
    
        ax2 = plt.subplot('512')
        ax2.plot(flatchain[:,5*i+1])

        ax3 = plt.subplot('513')
        ax3.plot(flatchain[:,5*i+2])
    
        if i ==0:
            ax4 = plt.subplot('514')
        ax4.plot(flatchain[:,5*i+3])
    
        ax5 = plt.subplot('515')
        ax5.plot(flatchain[:,5*i+4])


    # Is this a good way to show stuff? The only parameter shared between regions is PI, so I figure that it's most useful to show things per region


    fig1 = corner.corner(flatchain[:,[0,1,2,3,4]], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, labels=['kT', 'Z', '$T_{norm}$', 'PI', '$PL_{norm}$'])

    fig2 = corner.corner(flatchain[:,[5,6,7,3,9]], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, labels=['kT', 'Z', '$T_{norm}$', 'PI', '$PL_{norm}$'])

    fig3 = corner.corner(flatchain[:,[10,11,12,3,14]], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, labels=['kT', 'Z', '$T_{norm}$', 'PI', '$PL_{norm}$'])
    
    fig4 = corner.corner(flatchain[:,[15,16,17,18,19,20,21]], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, labels=['kT', 'Z', '$T_{norm}$', 'PI1', '$PL1_{norm}$', 'PI2', '$PL2_{norm}$'])
    
    fig5 = corner.corner(flatchain[:,[22,23,24,25,26,27,28]], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, labels=['kT', 'Z', '$T_{norm}$', 'PI1', '$PL1_{norm}$', 'PI2', '$PL2_{norm}$'])

    fig6 = corner.corner(flatchain[:,[29,30,31,32,33,34,35]], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, labels=['kT', 'Z', '$T_{norm}$', 'PI1', '$PL1_{norm}$', 'PI2', '$PL2_{norm}$'])
    
    plt.show()


