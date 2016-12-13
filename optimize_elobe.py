#!usr/bin/env python

import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import scipy.optimize
from scipy.special import gammaln as scipy_gammaln
from sherpa.astro.ui import *
import sys

from lobe_jet_posterior import *

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

    lpost = PoissonPosterior(lobe_data, lobe_models, lobe_rmf, jet_data, jet_models, jet_rmf, kT_prior, Z_prior, ratios=ratios)

    lobe_guess = np.zeros((regnr,5))
    for i in range(regnr): # reasonable initial guesses for lobe params
        # Initial guesses based off just the lobe fits
        if i == 0: lobe_guess[i,:] = [6.58, 0.57, 1e-4, 1.60, 1e-5]
        elif i == 1: lobe_guess[i,:] = [6.70, 0.56, 1e-4, 1.60, 1e-5]
        elif i == 2: lobe_guess[i,:] = [6.64, 0.70, 1e-4, 1.60, 1e-5]
        elif i == 3: lobe_guess[i,:] = [5.50, 0.70, 1e-4, 1.60, 1e-5]
        else: lobe_guess[i,:] = [6.0, 0.5, 1e-4, 1.6, 1e-5]

    jet_guess = np.zeros((regnr, 7))
    for i in range(regnr): # reasonable initial guesses for jet params
        if i == 0: jet_guess[i,:] =  [6.39, 0.63 , 1e-4, 1.60, 1e-5, 1.55, 2e-5]
        elif i == 1: jet_guess[i,:] = [6.82, 0.51, 1e-4, 1.60, 1e-5, 1.55, 2e-5]
        elif i == 2: jet_guess[i,:] = [6.64, 0.70, 1e-4, 1.60, 1e-5, 1.55, 2e-5]
        elif i == 3: jet_guess[i,:] = [6.01, 0.71, 1e-4, 1.60, 1e-5, 1.55, 2e-5]
        else: jet_guess[i,:] = [6.64, 0.70, 1e-4, 1.60, 1e-5, 1.55, 2e-5]


    # flatten everything into one big 1D array for scipy.optimize. The 2D arrays are recovered inside logposterior function
    init_guess = np.hstack((lobe_guess.flatten(), jet_guess.flatten()))

    init_guess =np.array([  6.39788314e+00,   7.17338184e-01,   9.87785296e-05,
                   1.59786368e+00,   2.36584632e-05,   6.81125715e+00,
                   5.17219415e-01,   3.18854969e-04,   1.59786368e+00,
                   4.35668743e-06,   6.76562645e+00,   6.76609201e-01,
                   3.72159118e-04,   1.59786368e+00,   4.50033007e-06,
                   6.02524402e+00,   6.90189056e-01,   3.88203696e-04,
                   1.59786368e+00,   5.33175216e-06,   6.39788314e+00,
                   6.69674512e-01,   9.91726831e-05,   1.59786368e+00,
                   9.58958261e-06,   1.58611923e+00,   1.64782593e-05,
                   6.81125715e+00,   5.29061215e-01,   1.10054444e-04,
                   1.59786368e+00,   1.11929736e-05,   1.58611923e+00,
                   1.00954963e-05,   6.76562645e+00,   7.54195482e-01,
                   1.34337146e-04,   1.59786368e+00,   5.73045201e-06,
                   1.58611923e+00,   1.00968644e-05,   6.02524402e+00,
                   7.38500431e-01,   1.35728604e-04,   1.59786368e+00,
                   6.57654564e-06,   1.58611923e+00,   1.00866441e-05,
                  1.91594128e-04,   2.37824976e-06,   1.29864446e-05])

    #Optimization routine
    fitmethod = scipy.optimize.minimize

    min_method = 'Nelder-Mead'

    neg = True
    results = fitmethod(lpost,x0=init_guess,method=min_method, args=(neg,), tol=1e-3)


    if min_method == 'BFGS':
        hess_inv = results.hess_inv
        np.savetxt('hess_inv_elobe.txt', hess_inv)


    print 'full results:', results

    lobe_results = results.x[0:regnr*5]
    jet_results = results.x[regnr*5:-3]

    hyper_params = results.x[-3:]


    pars = lobe_results.reshape(regnr,5)
    jet_pars = jet_results.reshape(regnr, 7)


    #re-link PL:
    pars[:,3] = pars[0,3]
    jet_pars[:,3] = pars[:,3]
    jet_pars[:,5] = jet_pars[0,5]

    #re-link kT:
    for i in range(regnr):
        jet_pars[i,0] = pars[i,0]
        jet_pars[i,1] = pars[i,1]


    print('Lobe Results :')
    print('Temperatures:', pars[:,0])
    print('Abundances:' , pars[:,1])
    print('Therm Norm:', pars[:,2])
    print('PI', pars[0,3])
    print('PL Norm:', pars[:,4])

    print ('Jet Results :')
    print('Temperatures:', jet_pars[:,0])
    print('Abundances :', jet_pars[:,1])

    print('Therm Norm:', jet_pars[:,2])
    print('PI1', jet_pars[0,3])
    print('PL Norm1:', jet_pars[:,4])
    print('PI2', jet_pars[0,5])
    print('PL Norm2:', jet_pars[:,6])


    print('Hyper Params:', hyper_params)

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
        lobe_models[i]._set_thawed_pars(results.x[i*5:(i*5)+5])
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
        ind = regnr*5
        jet_models[i]._set_thawed_pars(results.x[ind + i*7:ind +(i*7)+7])
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




