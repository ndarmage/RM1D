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
Plot_flx_cvg  = False
Plot_dD       = False
Plot_err      = False
Plot_k        = True

I = 400 # No. of mesh points
dx = 21.5/I
itr = 5 # No. of iterations
#ccng = 3 # Core configuration (Rahnema1997)
# ------ Load flx_RM? ------
plot_RM = True
if plot_RM:
    ''' Upload one file '''
    path_RM_flx_file1 = "../../../FD1dMGdiff/output/kflx_MC2011_LBC0RBC0_CMFD_I400_ritr5.npy"
    path_RM_flx_file2 = "../../../FD1dMGdiff/output/kflx_MC2011_LBC0RBC0_pCMFD_I400_ritr5.npy"
    
        #data = np.load(path_RM_flx_file)
    k_RM1, flx_RM1, xi1, st, dD1 = np.load(path_RM_flx_file1,allow_pickle=True)
    k_RM2, flx_RM2, xi2, st, dD2 = np.load(path_RM_flx_file2,allow_pickle=True)
    xi = xi2

# ------ Load flx_SN? ------
plot_SN = True
if plot_SN:
    N = 16  # No. of angles (S16)
    
    # Heterogeneous case - Rahnema1997
    #path_Sn_flx_file = "CORE3LBC0RBC0_I%d_N%d.npy" % (I,N)
    #path_Sn_flx_file = "../../SNMG1DSlab/output/CORE%dLBC0RBC0_I%d_N%d.npy"  % (ccng,I,N)
    
    # Homogeneous case - slab width (based on  Tomatis&Dall'Osso MC2011)
    path_SN_flx_file = "../../../SNMG1DSlab/output/kflx_MC2011_LBC0RBC0_I400_L21_N16.npy"
    ref_data = np.load(path_SN_flx_file,allow_pickle=True)
    k_SN, flxm = ref_data[0], ref_data[1][:,0,:]

    # ------ Normalization of flxm - by reaction rate ------
    RR_SN = sum(np.multiply(flxm,st).flatten())
    RR_RM2 = sum(np.multiply(flx_RM2[:,:,-1],st).flatten())
    flx_SN = (RR_RM2/RR_SN)*flxm
       
# ------ General ------
xim = (xi[1:] + xi[:-1]) / 2.# mid-cell points
G = st.shape[0]


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
            ax.plot(xim, (flx_RM1[g,:,-1] - flx_RM2[g,:,-1])/flx_RM1[g,:,-1]*100,label='$RM$' if g == 0 else "") # RM
            #ax.plot(xim, flx_SN[g,0,:],label='$SN$' if g == 0 else "") # SN
            #ax.plot(xim, flx_D[g,:],label='$D_{0}$' if g == 0 else "")  # Diffusion

#        if plot_SN:
#            ax.plot(xim, flx_SN[g,:],label='$S16$' if g == 0 else "")
#        
    
    plt.xlabel(r'$x$ $[cm]$',fontsize=fsz)
    plt.ylabel(r'$\phi$ $[AU]$',fontsize=fsz) 
    plt.xlim(xi[0], xi[-1])
    #plt.ylim(0, )
    
#    ax.annotate("Fast",fontsize=fsz,
#                 xy=(xi[I//4], flx_RM[0,I//4,-1]), xycoords='data',
#                 xytext=(35, 2.5), textcoords='data',
#                 arrowprops=dict(arrowstyle="->",
#                                 connectionstyle="arc3,rad=-0.3"))
#
#    ax.annotate("Thermal",fontsize=fsz,
#                xy=(xi[I//2], flx_RM[1,I//2,-1]), xycoords='data',
#                xytext=(47, 1.2), textcoords='data',
#                arrowprops=dict(arrowstyle="->",
#                                connectionstyle="arc3,rad=-0.3"))
#    
    #plt.xticks(xi[0::25]) # x-axis valus represent fuel assembly
    plt.grid(True,'both','both')
    ax.legend(loc='upper right',fontsize=fsz,ncol=1)
    
    filename = 'Tomatis2011_pCMFD_flx_%d_RMitr%d.pdf' %(I,itr)
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    #plt.show()
    os.system("pdfcrop %s %s" % (filename,filename))

# ====================================================== #
# ================  plot flux convergence ================ #
# ====================================================== #
if Plot_flx_cvg:
    J = [1,2,3,4,5] # No. of itr to plot

    # Optical length
    tau_m = np.zeros((G,I))
    #tau_i = np.zeros((G,I))
    for g in range (0,G):
        tau_m[g,:] = xim * st[g,:]

    dev_flx = np.zeros((G,I,np.size(J)))
    
    # Calculate flux deviation
    for j in range (0,np.size(J)):
        dev_flx[:,:,j] = (flx_RM2[:,:,J[j]] - flx_SN[:,:]) / flx_SN[:,:] *100

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
    
    filename = 'Tomatis2011_CMFD_cvg_I%d_RMitr%d.pdf' %(I,itr)
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    #plt.show()
    os.system("pdfcrop %s %s" % (filename,filename))
        
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
        ax[g].plot(xi, dD2[0][g,:],label='dDp')
        ax[g].plot(xi, dD2[1][g,:],label='dDm')
        ax[g].set_xlim(xi[0], xi[-1])
        ax[g].grid(True,'both','both')
        ax[g].legend(loc='upper right',fontsize=fsz//1.5,ncol=1)
    
    ax[0].set_ylabel(r'$\delta D_{1}^+ $ [AU]',fontsize = fsz)        
    ax[1].set_ylabel(r'$\delta D_{2}_- $ [AU]',fontsize = fsz)        
    plt.xlabel(r'$x$ [cm]',fontsize=fsz)
    filename = 'Tomatis2011_LBC0RBC0_pCMFD_dD_I%d_RMitr%d.pdf' %(I,itr)
    fig.savefig(filename,dpi=150,bbox_inches='tight')
    os.system("pdfcrop %s %s"%(filename,filename))
#    plt.show()

# ============================================================== #
# ================== Flux differences (Error) ================== #
# ============================================================== #
if Plot_err: 

    flx_err_D = {'ferrD0':((flx_RM['flx0'.format(i)][:,:,0] - 
                        flx_SN['flx0'.format(i)])/flx_SN['flx0'.format(i)])*100}
    flx_err_RM = {'ferrRM0':((flx_RM['flx0'.format(i)][:,:,-1] - 
                        flx_SN['flx0'.format(i)])/flx_SN['flx0'.format(i)])*100}

    for i in range(1,np.size(I)):
        flx_err_D['ferrD{}'.format(i)] = ((flx_RM['flx{}'.format(i)][:,:,0] - 
                        flx_SN['flx{}'.format(i)])/flx_SN['flx{}'.format(i)])*100
        
        flx_err_RM['ferrRM{}'.format(i)] = ((flx_RM['flx{}'.format(i)][:,:,-1] - 
                        flx_SN['flx{}'.format(i)])/flx_SN['flx{}'.format(i)])*100
    
    taug = np.zeros((G,I[0]))
    
    for g in range (0,G):
        taug[g,:] = xim['xim0'][0:I[0]] * st[g,0:I[0]]                   
    tau_m = {'tau0':taug}    
    
    #tau_i = np.zeros((G,I))
    for i in range (1,np.size(I)):
        taug = np.zeros((G,I[i]))
        for g in range (0,G):
            taug[g,:] = xim['xim{}'.format(i)][0:I[i]] * st[g,0:I[i]]
        tau_m['tau{}'.format(i)] = taug
    
    plt.figure(figsize=(14, 5)) 
    fig, ax = plt.subplots(G,figsize=(16, 10))
    
    for g in range (0,G):
        ax[g].set_prop_cycle(default_cycler)
        for i in range (0,np.size(I)):
            ax[g].tick_params(axis='both', which='major', labelsize=fsz)
            ax[g].tick_params(axis='both', which='minor', labelsize=fsz)
            ax[g].plot(tau_m['tau{}'.format(i)][g,0:20], flx_err_D['ferrD{}'.format(i)][g,0:20],label='D%.1f'%L[i] if g == 0 else "")
            ax[g].plot(tau_m['tau{}'.format(i)][g,0:20], flx_err_RM['ferrRM{}'.format(i)][g,0:20],label='RM%.1f'%L[i] if g == 0 else "")
            #ax[g].plot(xim['xim{}'.format(i)][0:20], flx_err_D['ferrD{}'.format(i)][g,0:20],label='D%.1f'%L[i] if g == 0 else "")
            #ax[g].plot(xim['xim{}'.format(i)][0:20], flx_err_RM['ferrRM{}'.format(i)][g,0:20],label='RM%.1f'%L[i] if g == 0 else "")
            ax[g].grid(True,'both','both')
            #ax[g].set_xscale('log')
            ax[g].legend(loc='upper right',fontsize=fsz//2,ncol=np.size(I))
      
    ax[0].set_ylabel(r'$\Delta \phi_{1} $ [AU]',fontsize = fsz)        
    ax[1].set_ylabel(r'$\Delta \phi_{2} $ [AU]',fontsize = fsz)        
    plt.xlabel(r'$\tau$ [optical length]',fontsize=fsz)
    
    fig.savefig('MC2011_Homog_Width_LBC0RBC0_RMitr%d.pdf' %(itr),dpi=300,bbox_inches='tight')
    #os.system("pdfcrop MC2011_Homog_WD_LBC0RBC0_I%d_L%d_itr%d.pdf MC2011_Homog_WD_LBC0RBC0_I%d_L%d_itr%d.pdf"%(I,L,itr,I,L,itr))
#    plt.show()
    
    
    
    
    
# ============================================================== #
# ================== Flux differences (Error) ================== #
# ============================================================== #
if Plot_k:     
    
    fig, ax = plt.subplots(figsize=(10,2.5))
    
    ax.tick_params(axis='both', which='major', labelsize=15)
    It = np.array([0,1,2,3,4,5])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.set_xlabel('Iteration No.',fontsize=18)
    ax.set_ylabel('$k_{eff}$',fontsize=18)
    ax.tick_params(axis='y')
    ax.set_axisbelow(True) 
    plt.ylim(min(k_RM1)-0.0005, max(k_RM1)+0.0005)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.yticks(np.linspace(k_RM1[0],k_RM1[-1],4))
    ax.scatter(It, k_RM1, color='r', marker='x', label='CMFD')
    ax.scatter(It, k_RM2, color='b',marker='^', label='pCMFD')
    ax.legend(loc='lower right',fontsize=fsz//2)
    
    filename = 'Tomatis2011_k_over_itr_10RMit.pdf'
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    os.system("pdfcrop %s %s" %(filename,filename))
    
