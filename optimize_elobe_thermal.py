#!usr/bin/env python

import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import scipy.optimize
from scipy.special import gammaln as scipy_gammaln
from sherpa.astro.ui import *
import sys

from lobe_jet_posterior_thermal import *

"""
    Script to optimize the jet+lobe spectra of the eastern lobe of Cygnus A
    Dependent on lobe_jet_posterior.py which contains the posterior class with the
    likelihood and prior functions
    
    Written by:
    Martijn de Vries
    martijndevries777@hotmail.com
    
"""

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
            set_source(i, 'xsphabs.abs1 * (xsapec.plsm' + str(i) + ')' )
            set_par('plsm' + str(i) + '.redshift', 0.0562)
            thaw('plsm' + str(i) + '.Abundanc')
    
        if i % 4 == 3:     # Jet regions 1 - 9  (3,7,11,15,19,23,27,31,35)
            set_source(i, 'xsphabs.abs1 * (xsapec.plsm' + str(i)  +')' )
            set_par('plsm' + str(i) + '.redshift', 0.0562)
            thaw('plsm' + str(i) + '.Abundanc')



    ignore_id(5, 0.0, 0.53)
    ignore_id(5, 6.9, None)
    freeze(abs1.nH)
    abs1.nH = 0.31 # Standard value for CygA  from HI surveys

    set_stat('chi2xspecvar')
    set_method('levmar')

    regnr = 4 # not changing this for now

    # Fit thermal region spectra for temperature/abundance priors
    fit_array  = [0,1,4,5,8,9,12,13,16,17] # fits of thermal regions around 4 lobe/jet regions

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


    #area ratios of the jet/lobe regions, used to compare normalizations (first order estimate)
    ratios = [ 0.677524429967, 0.371934604905, 0.388017118402, 0.411978221416]

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

    lobe_guess = np.zeros((regnr,3))
    for i in range(regnr): # reasonable initial guesses for lobe params
        # Initial guesses based off just the lobe fits
        if i == 0: lobe_guess[i,:] = [6.58, 0.57, 1e-4]
        elif i == 1: lobe_guess[i,:] = [6.70, 0.56, 1e-4]
        elif i == 2: lobe_guess[i,:] = [6.64, 0.70, 1e-4]
        elif i == 3: lobe_guess[i,:] = [5.50, 0.70, 1e-4]
        else: lobe_guess[i,:] = [6.0, 0.5, 1e-4, 1.6, 1e-5]

    jet_guess = np.zeros((regnr, 3))
    for i in range(regnr): # reasonable initial guesses for jet params
        if i == 0: jet_guess[i,:] =  [6.39, 0.63 , 1e-4]
        elif i == 1: jet_guess[i,:] = [6.82, 0.51, 1e-4]
        elif i == 2: jet_guess[i,:] = [6.64, 0.70, 1e-4]
        elif i == 3: jet_guess[i,:] = [6.01, 0.71, 1e-4]
        else: jet_guess[i,:] = [6.64, 0.70, 1e-4, 1.60, 1e-5, 1.55, 2e-5]


 

    # flatten everything into one big 1D array for scipy.optimize. The 2D arrays are recovered inside logposterior function
    init_guess = np.hstack((lobe_guess.flatten(), jet_guess.flatten()))

    init_guess =  np.array([  6.75169128e+00,   5.24436896e-01,   1.94234942e-04,
                     6.85679003e+00,   5.16508341e-01,   3.36226951e-04,
                     6.84446262e+00,   6.79967110e-01,   3.87565574e-04,
                     6.05968370e+00,   7.05262667e-01,   4.06884898e-04,
                     6.77615959e+00,   4.88895506e-01,   2.05183734e-04,
                     7.11495430e+00,   4.52354438e-01,   1.95664953e-04,
                     6.75439828e+00,   6.72390340e-01,   1.96717022e-04,
                     6.24757298e+00,   6.64952568e-01,   2.00948105e-04, 1e-4])

    fitmethod = scipy.optimize.minimize

    min_method = 'Nelder-Mead'

    neg = True
    results = fitmethod(lpost,x0=init_guess,method=min_method, args=(neg,), tol=1e-3)


    if min_method == 'BFGS':
        hess_inv = results.hess_inv
        np.savetxt('hess_inv_elobe_therm.txt', hess_inv)


    print 'full results:', results

    lobe_results = results.x[0:regnr*3]
    jet_results = results.x[regnr*3:-1]


    pars = lobe_results.reshape(regnr,3)
    jet_pars = jet_results.reshape(regnr, 3)



    print('Lobe Results :')
    print('Temperatures:', pars[:,0])
    print('Abundances:' , pars[:,1])
    print('Therm Norm:', pars[:,2])

    print ('Jet Results :')
    print('Temperatures:', jet_pars[:,0])
    print('Abundances :', jet_pars[:,1])
    print('Therm Norm:', jet_pars[:,2])

    print('Hyper pars:', results.x[-1])

    # Plot of lobe fits
    fig, axarr = plt.subplots(nrows=2, ncols=len(lobe_data), sharex=True, figsize=(23,10))
    col = 0
    rows = 2
    j = 0
    
    for i in range(len(lobe_data)):
        
        if j*2 >= rows:
            col += 1
            j = 0
        
        ax1 = axarr[0+2*j, col]
        ax2 = axarr[1+2*j,col]
        j += 1
        
        # Plot data
        ax1.errorbar(lobe_rmf[i].e_min, lobe_data[i].counts, yerr=np.sqrt(lobe_data[i].counts), label='Data', fmt='o', ecolor='g', markersize=2)
        
        # Plot model with pars from scipy.optimize
        lobe_models[i]._set_thawed_pars(results.x[i*3:(i*3)+3])
        model_counts = lobe_data[i].eval_model(lobe_models[i])
        ax1.plot(lobe_rmf[i].e_min, model_counts, label='Model', c='r')
        
        
        res =lobe_data[i].counts/model_counts
        res[res > 1e038] = 0 # filter inf and nan values
        indx = np.where((np.isnan(res)) == False)
        
        
        ax2.scatter(lobe_rmf[i].e_min[indx], res[indx]-1)
        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        ax2.set_ylim(-1,1)
        ax2.axhline(y=0)
        ax2.set_xlabel('Energy [keV]')
        ax1.set_ylabel('Counts')
        ax2.set_ylabel('Residual')
        ax1.set_yscale('log')
        ax1.set_xlim(0.5, 8)
        ax1.set_ylim(0.1, 1000)
        ax1.set_title('Lobe region ' + str(i))

    # Plot of jet fits
    fig, axarr = plt.subplots(nrows=2, ncols=len(lobe_data), sharex=True, figsize=(23,10))
    col = 0
    rows = 2
    j = 0
    
    for i, item in enumerate(jet_data):
        
        if j*2 >= rows:
            col += 1
            j = 0
        
        ax1 = axarr[0+2*j, col]
        ax2 = axarr[1+2*j,col]
        j += 1
        
        # Plot data
        ax1.errorbar(jet_rmf[i].e_min, jet_data[i].counts, yerr=np.sqrt(jet_data[i].counts), label='Data', fmt='o', ecolor='g', markersize=2)
        
        # Plot model with pars from scipy.optimize
        ind = regnr*3
        jet_models[i]._set_thawed_pars(results.x[ind + i*3:ind +(i*3)+3])
        model_counts = jet_data[i].eval_model(jet_models[i])
        ax1.plot(jet_rmf[i].e_min, model_counts, label='Model', c='r')
        
        res = (jet_data[i].counts - model_counts)/ jet_data[i].counts
        res[res > 1e038] = 0 # filter inf and nan values
        indx = np.where((np.isnan(res)) == False)
        
        ax2.scatter(jet_rmf[i].e_min[indx], res[indx])
        
        fig.subplots_adjust(hspace=0)
        
        ax2.axhline(y=0)
        ax2.set_ylim(-1,1)
        ax2.set_ylabel('Residual')
        
        ax2.set_xlabel('Energy [keV]')
        ax1.set_ylabel('Counts')
        ax1.set_yscale('log')
        ax1.set_xlim(0.5, 8)
        ax1.set_ylim(0.1, 1000)
        ax1.set_title('Jet region ' + str(i))
    
    plt.legend()
    plt.show()




