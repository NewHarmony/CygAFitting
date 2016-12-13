import corner
import emcee
from matplotlib import pyplot as plt

"""
    Script to plot mcmc results of the jet+lobe spectra of the eastern lobe of Cygnus A
    Needs a flatchain array which contains all the MCMC sampes
    
    Written by:
    Martijn de Vries
    martijndevries777@hotmail.com
    
"""

def plot_results(flatchain, regnr, lobe='east'):

    # Convergence plots for lobe regions
    for i in range(regnr):
        plt.figure()
        
        ax1 = plt.subplot('511')
        ax1.plot(flatchain[:,5*i+0])
        
        ax2 = plt.subplot('512')
        ax2.plot(flatchain[:,5*i+1])
        
        ax3 = plt.subplot('513')
        ax3.plot(flatchain[:,5*i+2])
        
        ax4 = plt.subplot('514')
        ax4.plot(flatchain[:,5*i+3])
        
        ax5 = plt.subplot('515')
        ax5.plot(flatchain[:,5*i+4])

        plt.savefig('L' + str(i) + '_conv_' + lobe[0] + '.pdf', dpi=300)

    # jet regions
    for i in range(regnr,2*regnr):
    
        ind = 5*regnr + 7*(i-regnr)
        plt.figure()
        
        ax1 = plt.subplot('711')
        ax1.plot(flatchain[:,ind])
        
        ax2 = plt.subplot('712')
        ax2.plot(flatchain[:,ind+1])
        
        ax3 = plt.subplot('713')
        ax3.plot(flatchain[:,ind+2])
        
        ax4 = plt.subplot('714')
        ax4.plot(flatchain[:,ind+3])
        
        ax5 = plt.subplot('715')
        ax5.plot(flatchain[:,ind+4])
        
        ax6 = plt.subplot('716')
        ax6.plot(flatchain[:,ind+5])
        
        ax7 = plt.subplot('717')
        ax7.plot(flatchain[:,ind+6])

        plt.savefig('J' + str(i-regnr) + '_conv_' + lobe[0] + '.pdf', dpi=300)

    # Hyperparams
    plt.figure()
    ax1 = plt.subplot('311')
    ax1.plot(flatchain[:,-3])
    
    ax2 = plt.subplot('312')
    ax2.plot(flatchain[:,-2])
    
    ax3 = plt.subplot('313')
    ax3.plot(flatchain[:,-1])
    
    plt.savefig('Hyper_conv_.pdf')
    
    # Corner plots
    parm_range = [0.999,0.999,0.999,0.999,0.999, 0.999, 0.999, 0.999]
    parm_range2 = [0.999,0.999,0.999,0.999,0.999, 0.999, 0.999 , 0.999, 0.999, 0.999]
    
    labels1 = ['kT', 'Z', '$T_{norm}$', 'PI', '$PL_{norm}$', '$\sigma_{Tnorm}$', '$\sigma_{PLnorm}$']
    labels2 = ['kT', 'Z', '$T_{norm}$', 'PI1', '$PL_{norm}1$', 'PI2', '$PL_{norm}2$' , '$\sigma_{Tnorm}$', '$\sigma_{PLnorm1}$', '$\sigma_{PLnorm2}$']
    
    fig1 = corner.corner(flatchain[:,[0,1,2,3,4,48,49]], bins=35, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, range=parm_range, labels=labels1)
    
    fig2 = corner.corner(flatchain[:,[5,6,7,3,9,48,49]], bins=35, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, range=parm_range, labels=labels1)
    
    fig3 = corner.corner(flatchain[:,[10,11,12,3,14,48,49]], bins=35, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, range=parm_range, labels=labels1)
    
    fig4 = corner.corner(flatchain[:,[15,16,17,3,19,48,49]], bins=35,  quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, range=parm_range, labels=labels1)
    
    fig5 = corner.corner(flatchain[:,[0,1,22,3,24,25,26,48,49,50]], bins=35, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, range=parm_range2, labels=labels2)
    
    fig6 = corner.corner(flatchain[:,[5,6,29,3,31,25,33,48,49,50]], bins=35, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, range=parm_range2, labels=labels2)
    
    fig7 = corner.corner(flatchain[:,[10,11,36,3,38,25,40,48,49,50]],bins=35,  quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, range=parm_range2, labels=labels2)
    
    fig8 = corner.corner(flatchain[:,[15,16,43,3,45,25,47,48,49,50]],bins=35,  quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12}, range=parm_range2, labels=labels2)
    
    
    fig1.savefig('L0_corner_' + lobe[0] + '.pdf', dpi=300)
    fig2.savefig('L1_corner_' + lobe[0] + '.pdf', dpi=300)
    fig3.savefig('L2_corner_' + lobe[0] + '.pdf', dpi=300)
    fig4.savefig('L3_corner_' + lobe[0] + '.pdf', dpi=300)
    
    fig5.savefig('J0_corner_' + lobe[0] + '.pdf', dpi=300)
    fig6.savefig('J1_corner_' + lobe[0] + '.pdf', dpi=300)
    fig7.savefig('J2_corner_' + lobe[0] + '.pdf', dpi=300)
    fig8.savefig('J3_corner_' + lobe[0] + '.pdf', dpi=300)


if __name__ == '__main__':

    flatchain = np.loadtxt('lj_mcmc_BFGS_n1000_elobe_3.txt')
    regnr = 4
    plot_results(flatchain, regnr, lobe='east')
