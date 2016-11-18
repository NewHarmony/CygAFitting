#!usr/bin/env python

import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import scipy.optimize
from scipy.special import gammaln as scipy_gammaln
from sherpa.astro.ui import *
import sys




logmin = 1000000000.0
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
        if not all(x == self.regnr for x in (len(l_d), len(l_m), len(j_d), len(j_m), self.kT_prior.shape[0], self.Z_prior.shape[0])):
            raise Exception('Error: length of data lists, model lists, and prior arrays must be the same')
        return
    
    def jet_loglikelihood(self, pars_list, neg=False):
        
        # Iterate over individual lobe region log-likelihoods and sum
        jet_res = 0 #
        for i, item in enumerate(self.jet_data):
            
            rmf = self.jet_data[i].get_rmf()
            erange = np.array(rmf.e_min) # need to convert to numpy array to use a double mask
            bounds = (erange > self.e_min) & (erange < self.e_max)
            
            model = self.jet_models[i]
            data = self.jet_data[i]
            pars = pars_list[i,:]
            
            # assume that normalizations are in log, so we'll exponentiate them:
            pars[2] = np.exp(pars[2])
            pars[4] = np.exp(pars[4])
            pars[6] = np.exp(pars[6])
            
            model._set_thawed_pars(pars)
            mean_model = data.eval_model(model)
            
            #stupid hack to make it not go -infinity
            mean_model += np.exp(-20.)
            res = np.nansum(-mean_model[bounds] + data.counts[bounds]*np.log(mean_model[bounds]) -  scipy_gammaln(data.counts[bounds] + 1.))
            
            if not np.isfinite(res):
                res = -logmin
            jet_res += res
        
            # Set the normalizations back to log (for the priors)
            pars[2] = np.log(pars[2])
            pars[4] = np.log(pars[4])
            pars[6] = np.log(pars[6])

        if neg:
            return -jet_res
        else:
            return jet_res

    def jet_logprior(self, pars_array):
    
        # Iterate over individual jet region log priors and sum
        logprior = 0
        for i, item in enumerate(self.jet_data):
            
            pars = pars_array[i,:] # 2D pars array [i,j], i=reg nr, j=par nr
            
            # Gaussian priors for kT and Z, based on fit result of surrounding region
            kT = pars[0]
            mu_kT = self.kT_prior[i,0]
            sigma_kT = self.kT_prior[i,1]
            p_kT = norm.pdf(kT, loc=mu_kT, scale=sigma_kT)
            if (kT > mu_kT+1.5) or (kT < mu_kT-1.5): p_kT = 0

            Z = pars[1]
            mu_Z = self.Z_prior[i,0]
            sigma_Z = self.Z_prior[i,1]
            p_Z = norm.pdf(Z, loc=mu_Z, scale=sigma_Z)
            if Z < 0 or Z > 1.0: p_Z = 0

            T_lognorm = pars[2]
            p_Tnorm = ((T_lognorm > -9) & (T_lognorm < -3))

            PL1_lognorm = pars[4]
            p_PL1norm = ((PL1_lognorm > -15) & (PL1_lognorm <-3))
            if np.exp(PL1_lognorm) > 0.2*np.exp(T_lognorm): p_PL1norm = 0
            
            PL2_lognorm = pars[6]
            p_PL2norm = ((PL2_lognorm > -15) & (PL2_lognorm <-3))
            #if PL2_lognorm > T_lognorm: p_PL2norm = 0
            
            #if np.exp(PL2_lognorm) > 0.5*np.exp(T_lognorm): p_PL2norm = 0
            #if PL2_lognorm < PL1_lognorm: p_PL2norm = 0
            
            #if PL1_lognorm > PL2_lognorm: p_PL1norm = 0
            

            logprior_reg = np.log(p_kT * p_Z * p_Tnorm * p_PL1norm * p_PL2norm)
    
            if not np.isfinite(logprior_reg):
                logprior += -logmin
            else:
                logprior += logprior_reg

        # PI1 prior (lobe component)
        PI1 = pars_array[0,3]
        p_PI1 = norm.pdf(PI1, loc=1.6, scale=0.1)
        if ((PI1 < 1.2) or (PI1 > 2.5)): p_PI1 = 0
        
        # PI2 prior (jet component)
        PI2 = pars_array[0,3]
        p_PI2 = norm.pdf(PI2, loc=1.5, scale=0.1)
        if ((PI2 < 1.2) or (PI2 > 2.5)): p_PI2 = 0
        
        logprior_PI = np.log(p_PI1 * p_PI2)

        if not np.isfinite(logprior_PI):
            logprior += -logmin
        else:
            logprior += logprior_PI

        return logprior

    def lobe_loglikelihood(self, pars_list, neg=False):
        
        # Iterate over individual lobe region log-likelihoods and sum
        lobe_res = 0 #
        for i, item in enumerate(self.lobe_data):
            
            rmf = self.lobe_data[i].get_rmf()
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

        if neg:
            return -lobe_res
        else:
            return lobe_res

    def lobe_logprior(self, pars_array):
    
        # Iterate over individual lobe region log-priors and sum
        logprior = 0
        for i, item in enumerate(self.lobe_data):
            
            pars = pars_array[i,:] # 2D pars array [i,j], i=reg nr, j=par nr
            #print pars
            
            # Gaussian priors for kT and Z, based on fit result of surrounding region
            kT = pars[0]
            mu_kT = self.kT_prior[i,0]
            sigma_kT = self.kT_prior[i,1]
            p_kT = norm.pdf(kT, loc=mu_kT, scale=sigma_kT)
            if (kT > 10.0) or (kT < 0): p_kT = 0

            Z = pars[1]
            mu_Z = self.Z_prior[i,0]
            sigma_Z = self.Z_prior[i,1]
            p_Z = norm.pdf(Z, loc=mu_Z, scale=sigma_Z)
            if Z < 0 or Z > 1.0: p_Z = 0

            T_lognorm = pars[2]
            p_Tnorm = ((T_lognorm > -9) & (T_lognorm < -3))
            
            PL_lognorm = pars[4]
            p_PLnorm = ((PL_lognorm > -15) & (PL_lognorm <-3))
            #if np.exp(PL_lognorm) > 0.2*np.exp(T_lognorm): p_PLnorm = 0

            #print p_kT, p_Z, p_Tnorm, p_PLnorm
            
            logprior_reg = np.log(p_kT * p_Z * p_Tnorm * p_PLnorm)
            
            if not np.isfinite(logprior_reg):
                logprior += -logmin
            else:
                logprior += logprior_reg

        # PI prior (shared between lobe regions)
        PI = pars_array[0,3]
        p_PI = norm.pdf(PI, loc=1.6, scale=0.1)
        if ((PI < 1.2) or (PI > 2.5)): p_PI = 0
        
        logprior_PI = np.log(p_PI)
        
        if not np.isfinite(logprior_PI):
            logprior += -logmin
        else:
            logprior += logprior_PI
        
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
        
        # Test to link T/Z
        for i in range(self.regnr):
            lobe_pars_array[i,0] = jet_pars_array[i,0]
            lobe_pars_array[i,1] = jet_pars_array[i,1]
        
        # Note: I'm passing parameters that don't do anything to scipy.optimize. I hope that doesn't matter
      
        ll = self.jet_loglikelihood(jet_pars_array)+self.lobe_loglikelihood(lobe_pars_array)
        logprior = self.jet_logprior(jet_pars_array) + self.lobe_logprior(lobe_pars_array)
        
        print('Lobe + jet LL ' + str(ll))
        print('Lobe + jet prior ' + str(logprior))
        
        lpost = ll + logprior

        if neg is True:
            print('lpost ' + str(-lpost))
            return -lpost
        else:
            print('lpost' + str(lpost))
            return lpost
    
    def __call__(self, pars_array, neg=False):
        return self.logposterior(pars_array, neg)

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

    lobe_guess = np.zeros((regnr,5))

    for i in range(regnr): # reasonable initial guesses for lobe params
        # Initial guesses based off just the lobe fits
        if i == 0: lobe_guess[i,:] = [6.58, 0.57, -8.71, 1.60, -11.87]
        if i == 1: lobe_guess[i,:] = [6.56, 0.56, -8.26, 1.60, -10.88]
        if i == 2: lobe_guess[i,:] = [6.64, 0.70, -8.06, 1.60, -10.90]
        else: lobe_guess[i,:] = [6.0, 0.5, -6, 1.6, -7]

    jet_guess = np.zeros((regnr, 7))
    for i in range(regnr): # reasonable initial guesses for jet params
        if i == 0: jet_guess[i,:] = [6.58, 0.57, -8.71, 1.55, -11.87, 1.52, -9.5]
        if i == 1: jet_guess[i,:] = [6.56, 0.56, -8.26, 1.55, -10.88, 1.52, -9.5]
        if i == 2: jet_guess[i,:] = [6.64, 0.70, -8.06, 1.55, -10.90, 1.52, -9.5]
        else: jet_guess[i,:] = [6.64, 0.70, -8.06, 1.55, -10.90, 1.7, -10]

    # flatten everything into one big 1D array for scipy.optimize. The 2D arrays are recovered inside logposterior function
    init_guess = np.hstack((lobe_guess.flatten(), jet_guess.flatten()))


    #Optimization routine
    fitmethod = scipy.optimize.minimize
    neg = True
    results = fitmethod(lpost,x0=init_guess,method='Nelder-Mead', args=(neg,))
    print results
    print 'full results:', results.x

    lobe_results = results.x[0:regnr*5]
    jet_results = results.x[regnr*5:]


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
    print('Abundances :', jet_pars[:,1])

    print('Therm Norm:', jet_pars[:,2])
    print('PI1', jet_pars[0,3])
    print('PL Norm1:', jet_pars[:,4])
    print('PI2', jet_pars[0,5])
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
        
        res =jet_data[i].counts/model_counts
        res[res > 1e038] = 0 # filter inf and nan values
        indx = np.where((np.isnan(res)) == False)
        
        ax2.scatter(jet_rmf[i].e_min[indx], res[indx]-1)
        
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




