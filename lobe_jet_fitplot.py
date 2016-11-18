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
    """
    fit_array = [0,1,4,5,8,9]
    for i in fit_array:
        fit(i)

    covar(0,1,4,5,8,9)
    covar_result = get_covar_results()
    """
    regnr = 3 # just do 3 regions for now

    kT_prior = np.zeros((regnr, 2))
    Z_prior = np.zeros((regnr,2))

    # build kT_prior and Z_prior matrices (i,0 = mu; i,1 = sigma)
    # Simplified version for now: kT_avg = (kT_1 + kT_2)/2
    # fkT_avg = fkT_1 + abs(kT_avg - kT_1)
    # Maybe its better to just do a joint fit to the top and bottom thermal region
    """
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
    """

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

    # temperary fit params to play around with so I can fix the plots
    results = np.array([6.91860961  , 0.78720923 , -8.99997726 ,  1.75435284, -10.86896919 ,\
          7.95266117 ,  0.50502138 , -8.24966977  , 1.75435284 ,-10.76552872 ,\
          6.6920166 ,   0.33556179  ,-7.90850756  , 1.75435284, -11.16897997,\
          6.91860961 ,  0.78720923 , -8.97907755 ,  1.75435284 , -13.63377279, 1.78663358, -10.83551357 ,\
          7.95266117,   0.50502138 , -8.99780965,   1.75435284 , -10.91281741  , 1.78663358 , -13.11711422 , \
          6.6920166,    0.33556179 , -8.84984251  , 1.75435284 , -13.43136407 ,  1.78663358,  -10.93006715])

    lobe_results = results[0:regnr*5]
    jet_results = results[regnr*5:]


    pars = lobe_results.reshape(regnr,5)
    jet_pars = jet_results.reshape(regnr, 7)

    # Set normalization params back to their real value
    pars[:,2] = np.exp(pars[:,2])
    pars[:,4] = np.exp(pars[:,4])

    jet_pars[:,2] = np.exp(jet_pars[:,2])
    jet_pars[:,4] = np.exp(jet_pars[:,4])
    jet_pars[:,6] = np.exp(jet_pars[:,6])

    print('Lobe Results :')
    print('Temperatures:', pars[:,0])
    print('Abundances:' , pars[:,1])
    print('Therm Norm:', pars[:,2])
    print('PI', pars[0,3])
    print('PL Norm:', pars[:,4])

    print ('Jet Results :')
    print('Temperatures:', jet_pars[:,0])
    print('Abundances :', jet_pars[:,0])

    print('Therm Norm:', jet_pars[:,2])
    print('PI1', jet_pars[0,3])
    print('PL Norm1:', jet_pars[:,4])
    print('PI2', jet_pars[:,5])
    print('PL Norm2:', jet_pars[:,6])



    fig, axarr = plt.subplots(nrows=2, ncols=len(lobe_data), sharex=True, figsize=(23,10))

    col = 0
    rows = 2
    j = 0

    # Plot of lobe fits
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
        lobe_models[i]._set_thawed_pars(results[i*5:(i*5)+5])
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
        ax2.set_ylabel('Frac. residual data/model')
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
        jet_models[i]._set_thawed_pars(results[ind + i*7:ind +(i*7)+7])
        model_counts = jet_data[i].eval_model(jet_models[i])
        ax1.plot(jet_rmf[i].e_min, model_counts, label='Model', c='r')
        
        res =jet_data[i].counts/model_counts
        res[res > 1e038] = 0 # filter inf and nan values
        indx = np.where((np.isnan(res)) == False)
        
        ax2.scatter(jet_rmf[i].e_min[indx], res[indx]-1)

        fig.subplots_adjust(hspace=0)

        ax2.axhline(y=0)
        ax2.set_ylim(-1,1)
        ax2.set_ylabel('Frac. Residual data/model')

        ax2.set_xlabel('Energy [keV]')
        ax1.set_ylabel('Counts')
        ax1.set_yscale('log')
        ax1.set_xlim(0.5, 8)
        ax1.set_ylim(0.1, 1000)
        ax1.set_title('Jet region ' + str(i))

    plt.legend()
    plt.show()



