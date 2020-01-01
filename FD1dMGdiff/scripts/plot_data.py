#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
sys.path.append('..')

from cycler import cycler
default_cycler = (cycler(color=['r', 'g', 'b', 'y']) *
                  #cycler(marker=['.'])
                  #*
                  cycler(linestyle=['-', '--', ':', '-.']) 
                  )
                  
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
fsz = 24. # font size

# ========== What would you like to plot? ==========
# For heterogeneous case, consider upload several fluxes
plot_flx      = False
Plot_flx_dev  = False
Plot_dD       = False
Plot_err      = True

I0 = 400 # No. of mesh points
dx0 = 21.5/I0
I = np.array([I0-240, I0-160, I0-80, I0, 480, 560, 720, 800])
L = np.multiply(dx0,I)
itr = 10 # No. of iterations
#ccng = 3 # Core configuration (Rahnema1997)
# ------ Load flx_RM? ------
plot_RM = True
if plot_RM:
    ''' Upload one file '''
    #path_RM_flx_file = "../../FD1dMGdiff/output/kflx_Rahnema1997_C%d_LBC0RBC0_I%d_itr%d.npy" %(ccng,I,itr)
    #path_RM_flx_file = "../../FD1dMGdiff/output/kflx_LBC0RBC0_I%d_%d.npy" % (I,itr)
    
    path_RM_flx_file = "../../FD1dMGdiff/output/kflx_Wd_LBC0RBC0_I%d_L%d_it%d.npy" % (I[0],L[0],itr)
    
        #data = np.load(path_RM_flx_file)
    k_RM, flx_save, xi, st, dD = np.load(path_RM_flx_file,allow_pickle=True)
    flx = {'flx0':flx_save}
    ''' Uploading several files of different widths '''
    for i in range (1,np.size(I)):
        path_RM_flx_file = "../../FD1dMGdiff/output/kflx_Wd_LBC0RBC0_I%d_L%d_it%d.npy" % (I[i],L[i],itr)
        #data = np.load(path_RM_flx_file)
        k_RM, flx_save, xi, st, dD = np.load(path_RM_flx_file,allow_pickle=True)        
        flx['flx{}'.format(i)] = flx_save
    
        #flx_RM = flx_save[:,:,1:]
        #flx_D = flx_save[:,:,0]
        #flx[:,:,:,i] = flx_RM
# ------ Load flx_SN? ------
plot_SN = False
if plot_SN:
    N = 16  # No. of angles (S16)
    #path_Sn_flx_file = "CORE3LBC0RBC0_I%d_N%d.npy" % (I,N)
    path_Sn_flx_file = "../../SNMG1DSlab/output/CORE%dLBC0RBC0_I%d_N%d.npy"  % (ccng,I,N)
    ref_data = np.load(path_Sn_flx_file,allow_pickle=True)
    k_SN, flxm_SN = ref_data[0], ref_data[1]
    # ------ Normalization of flxm - by reaction rate ------
    RR_SN = sum(np.multiply(flxm_SN[:,0,:],st).flatten())
    RR_RM = sum(np.multiply(flx_RM[:,:,1],st).flatten())
    #RR_D = sum(np.multiply(flx_D[:,:,0],st).flatten())
    flx_SN = (RR_RM/RR_SN)*flxm_SN[:,0,:]

# ------ General ------
xim = (xi[1:] + xi[:-1]) / 2. # mid-cell points
G = np.shape(st)


# ====================================================== #
# ======================== PLOT ======================== #
# ====================================================== #


# ====================================================== #
# =====================  plot flux ===================== #
# ====================================================== #
if plot_flx:        
    fig, ax = plt.subplots(figsize=(11,7))
    # change the fontsize of major/minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=fsz)
    ax.tick_params(axis='both', which='minor', labelsize=fsz)
    for g in range (0,G):
        ax.set_prop_cycle(default_cycler)
        if plot_RM:
            ax.plot(xim, flx_RM[g,:,-1],label='$RM$' if g == 0 else "") # RM
            ax.plot(xim, flx_D[g,:],label='$D_{0}$' if g == 0 else "")  # Diffusion

        if plot_SN:
            ax.plot(xim, flx_SN[g,:],label='$S16$' if g == 0 else "")
        
    
    plt.xlabel(r'$x$ $[cm]$',fontsize=fsz)
    plt.ylabel(r'$\phi$ $[AU]$',fontsize=fsz) 
    plt.xlim(xi[0], xi[-1])
    plt.ylim(0, )
    
    ax.annotate("Fast",fontsize=fsz,
                 xy=(xi[I//4], flx_RM[0,I//4,-1]), xycoords='data',
                 xytext=(35, 2.5), textcoords='data',
                 arrowprops=dict(arrowstyle="->",
                                 connectionstyle="arc3,rad=-0.3"))

    ax.annotate("Thermal",fontsize=fsz,
                xy=(xi[I//2], flx_RM[1,I//2,-1]), xycoords='data',
                xytext=(47, 1.2), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=-0.3"))
    
    #plt.xticks(xi[0::25]) # x-axis valus represent fuel assembly
    plt.grid(True,'both','both')
    ax.legend(loc='upper right',fontsize=fsz,ncol=1)
    
    fig.savefig('Rahnema1997_C3FLX2G_LBC0RBC0_I%d_itr%d.pdf' %(I,itr),dpi=300,bbox_inches='tight')
    #plt.show()
    os.system("pdfcrop Rahnema1997_C3FLX2G_LBC0RBC0_I%d_itr%d.pdf Rahnema1997_C3FLX2G_LBC0RBC0_I%d_itr%d.pdf"%(I,itr,I,itr))

# ====================================================== #
# ================  plot flux deviation ================ #
# ====================================================== #
if Plot_flx_dev:
    #J = [1,2,5,10,25,50,200] # No. of itr to plot
    J = [1,5,10] # No. of itr to plot
    # Optical length
    tau_m = np.zeros((G,I))
    #tau_i = np.zeros((G,I))
    for g in range (0,G):
        tau_m[g,:] = xim * st[g,:]

    dev_flx = np.zeros((G,I,np.size(J)))
    
    # Calculate flux deviation
    for j in range (0,np.size(J)):
        dev_flx[:,:,j] = (flx_RM[:,:,J[j]] - flx_SN) / flx_SN *100

    plt.figure(figsize=(14, 5)) 
    fig, ax = plt.subplots(G,figsize=(14, 9))
    
    for g in range (0,G):
    
        ax[g].tick_params(axis='both', which='major', labelsize=fsz)
        ax[g].tick_params(axis='both', which='minor', labelsize=fsz)
        ax[g].set_prop_cycle(default_cycler)
        for j in range (0,np.size(J)):
            ax[g].plot(tau_m[g,0:I//2], dev_flx[g,0:I//2,j],label='it-%d'%J[j])
        
        ax[g].set_xlim(tau_m[g,0], tau_m[g,-1])
        ax[g].set_xscale('log')
        ax[g].grid(True,'both','both')
        ax[g].legend(loc='upper right',fontsize=fsz//1.5,ncol=np.size(J))
    
    ax[0].set_ylabel(r'$\Delta \phi_{1} $ [\%]',fontsize = fsz)        
    ax[1].set_ylabel(r'$\Delta \phi_{2} $ [\%]',fontsize = fsz)        
    
    plt.xlabel(r'$\tau$ [optical length]',fontsize=fsz)
    
    fig.savefig('Rahnema1997_C3CVG_LBC0RBC0_I%d_itr%d.pdf' %(I,itr),dpi=150,bbox_inches='tight')
    os.system("pdfcrop Rahnema1997_C3CVG_LBC0RBC0_I%d_itr%d.pdf Rahnema1997_C3CVG_LBC0RBC0_I%d_itr%d.pdf"%(I,itr,I,itr))
#    plt.show()
#    
# ====================================================== #
# ======================  plot dD ====================== #
# ====================================================== #
if Plot_dD: 

    plt.figure(figsize=(14, 5)) 
    fig, ax = plt.subplots(G,figsize=(14, 7))
    
    for g in range (0,G):
    
        ax[g].tick_params(axis='both', which='major', labelsize=fsz)
        ax[g].tick_params(axis='both', which='minor', labelsize=fsz)
        ax[g].set_prop_cycle(default_cycler)
        ax[g].plot(xi, dD[g,])
        ax[g].set_xlim(xi[0], xi[-1])
        ax[g].grid(True,'both','both')
        ax[g].legend(loc='upper right',fontsize=fsz//1.5,ncol=1)
    
    ax[0].set_ylabel(r'$\delta D_{1} $ [AU]',fontsize = fsz)        
    ax[1].set_ylabel(r'$\delta D_{2} $ [AU]',fontsize = fsz)        
    plt.xlabel(r'$x$ [cm]',fontsize=fsz)
    fig.savefig('Rahnema1997_C3dD_LBC0RBC0_I%d_itr%d.pdf' %(I,itr),dpi=150,bbox_inches='tight')
    os.system("pdfcrop Rahnema1997_C3dD_LBC0RBC0_I%d_itr%d.pdf Rahnema1997_C3dD_converg_I%d_itr%d.pdf"%(I,itr,I,itr))
#    plt.show()

# ============================================================== #
# ================== Flux differences (Error) ================== #
# ============================================================== #
if Plot_err: 
    flx_err_diff = ((flx_D - flx_SN)/flx_SN)*100
    flx_err_RM = ((flx_RM - flx_SN)/flx_SN)*100
    
    tau_m = np.zeros((G,I))
    #tau_i = np.zeros((G,I))
    for g in range (0,G):
        tau_m[g,:] = xim * st[g,:]
    
    plt.figure(figsize=(14, 5)) 
    fig, ax = plt.subplots(G,figsize=(14, 7))
    
    for g in range (0,G):
        for i in range (0,np.size(I)):
            ax[g].tick_params(axis='both', which='major', labelsize=fsz)
            ax[g].tick_params(axis='both', which='minor', labelsize=fsz)
            ax[g].set_prop_cycle(default_cycler)
            ax[g].plot(tau_m[g,-6:-1], flx['flx{}'.format(i)][g,-6:-1,-1])
            ax[g].plot(tau_m[g,-5:-1], flx_err_RM[g,-5:-1,-1])
            ax[g].grid(True,'both','both')
            ax[g].legend(loc='upper right',fontsize=fsz//1.5,ncol=1)
      
    ax[0].set_ylabel(r'$\delta D_{1} $ [AU]',fontsize = fsz)        
    ax[1].set_ylabel(r'$\delta D_{2} $ [AU]',fontsize = fsz)        
    plt.xlabel(r'$\tau$ [optical length]',fontsize=fsz)
    
    fig.savefig('MC2011_Homog_WD_LBC0RBC0_I%d_L%d_itr%d.pdf' %(I,L,itr),dpi=150,bbox_inches='tight')
    os.system("pdfcrop MC2011_Homog_WD_LBC0RBC0_I%d_L%d_itr%d.pdf MC2011_Homog_WD_LBC0RBC0_I%d_L%d_itr%d.pdf"%(I,L,itr,I,L,itr))
#    plt.show()
    