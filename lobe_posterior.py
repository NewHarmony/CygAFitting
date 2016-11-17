#!usr/bin/env python

import logging
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
import scipy.optimize
from scipy.special import gammaln as scipy_gammaln
from sherpa.astro.ui import *
import sys



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
            p_Tnorm = ((T_lognorm > -15) & (T_lognorm < -5))
            
            PL_lognorm = pars[4]
            p_PLnorm = ((PL_lognorm > -15) & (PL_lognorm <-8))
            #if PL_lognorm > 0.2*T_lognorm: p_PLnorm = 0


            #print 'Priors:', p_kT, p_Z, p_Tnorm, p_PLnorm
            
            logprior_reg = np.log(p_kT * p_Z * p_Tnorm * p_PLnorm)
            if not np.isfinite(logprior_reg):
                logprior += -logmin
            else:
                logprior += logprior_reg
    
        # PI prior (shared between lobe regions)
        PI = pars_array[0,3]
        p_PI = norm.pdf(kT, loc=1.6, scale=0.10)

        if (( PI < 1.2) or (PI > 2.5)): p_PI = 0
    
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


if __name__ == '__main__': # So that I can call the PoissonPosterior class from other scripts
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
            #group_counts(i, 23) # I fit these in Sherpa, they are not part of the bayesian model

            ignore_id(i, 0.0, 0.52)
            ignore_id(i, 6.98, None)
    
        if i % 4 == 2:      # Lobe regions 1 - 9 (2,6,10,14,18,22,26,30,34)
            set_source(i, 'xsphabs.abs1 * (xsapec.plsm' + str(i) + ' + xspowerlaw.pow' + str(i) + ')' )
            if i == 1: set_par('pow' + str(i) + '.PhoIndex', 1.6, min=1.0, max=2.2)
            set_par('plsm' + str(i) + '.redshift', 0.0562)
            thaw('plsm' + str(i) + '.Abundanc')
            #group_counts(i, 20)
            #ignore_id(i, 0.0, 0.52)
            #Ignore_id(i, 6.98, None)

    
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

    fit_array = [0,1,4,5,8,9,12,13,16,17]
    for i in fit_array:
        fit(i)

    covar(0,1,4,5,8,9,12,13)#,16,17)#,12,13)
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
    initial_guess = np.zeros((regnr,5))

    for i in range(regnr): # reasonable initial guesses
        initial_guess[i,:] = [6.0, 0.5, -7, 1.6, -11]

    # Optimization routine
    fitmethod = scipy.optimize.minimize
    neg = True
    results = fitmethod(lpost,x0=initial_guess,method='Nelder-mead',options={'maxiter':10000}, args=(neg,))

    #results = fitmethod(lpost,x0=initial_guess, maxiter=200, args=(neg,))



    print(results.x)
    print('Results :')

    pars = results.x.reshape(regnr,5)

    # Set normalization params back to their real value
    pars[:,2] = np.exp(pars[:,2])
    pars[:,4] = np.exp(pars[:,4])


    print('Temperatures:', pars[:,0])
    print('Abundances:' , pars[:,1])
    print('Therm Norm:', pars[:,2])
    print('PI', pars[0,3])
    print('PL Norm:', pars[:,4])


    # Plot of fits
    for i, item in enumerate(lobe_data):
        ax = plt.subplot('22' + str(i))
    
        # Plot data
        ax.errorbar(lobe_rmf[i].e_min, lobe_data[i].counts, yerr=np.sqrt(lobe_data[i].counts), label='Data', fmt='o', ecolor='g', markersize=2)

        # Plot model with pars from scipy.optimize
        lobe_models[i]._set_thawed_pars(results.x[i*5:(i*5)+5])
        model_counts = lobe_data[i].eval_model(lobe_models[i])
        ax.plot(lobe_rmf[i].e_min, model_counts, label='Model', color='r')
    
        ax.set_xlabel('Energy [keV]')
        ax.set_ylabel('Counts')
        ax.set_yscale('log')
        ax.set_xlim(0.5, 8)
        ax.set_ylim(0.1, 1000)
        ax.set_title('Lobe region ' + str(i))

    plt.legend()
    plt.show()

