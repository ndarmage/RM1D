#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick
#from matplotlib.ticker import ScalarFormatter
#class ScalarFormatterForceFormat(ScalarFormatter):
#    def _set_format(self,vmin,vmax):  # Override function that finds format to use.
#        self.format = "%1.2f"  # Give format here

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
Plot_k        = False
plot_flx_zoom = True
Plot_rho      = False
Plot_LB       = False
I = 400 # No. of mesh points
#dx = 21.5/I
itr = 250 # No. of iterations
#ccng = 3 # Core configuration (Rahnema1997)
# ------ Load flx_RM? ------
plot_RM = True
if plot_RM:
    ''' Upload one file '''
    path_RM_flx_file1 = "kflx_MC2011_LBC0RBC0_CMFD_I400_it250.npy"
    path_RM_flx_file2 = "kflx_MC2011_LBC0RBC0_pCMFD_I400_it250.npy"
    #path_RM_flx_file3 = "../../../FD1dMGdiff/output/kflx_MC2011_LBC0RBC0_I400_it10.npy"
        #data = np.load(path_RM_flx_file)
    k_RM1, flx_RM1, xi1, st, dD1 = np.load(path_RM_flx_file1,allow_pickle=True)
    k_RM2, flx_RM2, xi2, st, dD2 = np.load(path_RM_flx_file2,allow_pickle=True)
    #cCMFD = np.load(path_RM_flx_file3,allow_pickle=True)
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
    RR_RM1 = sum(np.multiply(flx_RM1[:,:,-1],st).flatten())
    flx_SN = (RR_RM1/RR_SN)*flxm

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
    fig, ax = plt.subplots(figsize=(14,7))
    # change the fontsize of major/minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=fsz)
    ax.tick_params(axis='both', which='minor', labelsize=fsz)
    for g in range (0,G):
        ax.set_prop_cycle(default_cycler)
        if plot_RM:
            ax.plot(xim[I//2:-1], flx_RM1[g,I//2:-1,-1],label='$RM-CMFD$' if g == 0 else "") # CMFD
            ax.plot(xim[I//2:-1], flx_RM2[g,I//2:-1,-1],label='$RM-pCMFD$' if g == 0 else "") # pCMFD
            
            #ax.plot(xim, flx_D[g,:],label='$D_{0}$' if g == 0 else "")  # Diffusion

        if plot_SN:
#            ax.plot(xim, flx_SN[g,:],label='$S16$' if g == 0 else "")
            ax.plot(xim[I//2:-1], flx_SN[g,I//2:-1],label='$SN$' if g == 0 else "") # SN
    
    plt.xlabel(r'$x$ $[cm]$',fontsize=fsz)
    plt.ylabel(r'$\phi$ $[AU]$',fontsize=fsz) 
    #plt.xlim(xi[0], xi[-1])
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
    
    filename = 'Tomatis2011_flx_%d_RMitr%d.pdf' %(I,itr)
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    #plt.show()
    os.system("pdfcrop %s %s" % (filename,filename))

# ====================================================== #
# ================  plot flux convergence ================ #
# ====================================================== #
if Plot_flx_cvg:
    J = [0,1,2,5,10,25,50,150,250] # No. of itr to plot

    # Optical length
    tau_m = np.zeros((G,I))
    #tau_i = np.zeros((G,I))
    for g in range (0,G):
        tau_m[g,:] = xim * st[g,:]

    #dev_flx_CMFD = np.zeros((G,I,np.size(J)))
    dev_flx_pCMFD = np.zeros((G,I,np.size(J)))
    
    # Calculate flux deviation
    for j in range (0,np.size(J)):
        #dev_flx_CMFD[:,:,j] = (flx_RM1[:,:,J[j]] - flx_SN[:,:]) / flx_SN[:,:] *100
        dev_flx_pCMFD[:,:,j] = (flx_RM2[:,:,J[j]] - flx_SN[:,:]) / flx_SN[:,:] *100

    
    plt.figure(figsize=(14, 5)) 
    fig, ax = plt.subplots(G,figsize=(14, 9))
    
    for g in range (0,G):
    
        ax[g].tick_params(axis='both', which='major', labelsize=fsz)
        ax[g].tick_params(axis='both', which='minor', labelsize=fsz)
        ax[g].set_prop_cycle(default_cycler)
        for j in range (0,np.size(J)):
            #ax[g].plot(tau_m[g,0:I//2], dev_flx_CMFD[g,0:I//2,j],label='it-%d'%J[j])
            ax[g].plot(tau_m[g,0:I//2], dev_flx_pCMFD[g,0:I//2,j],label='it-%d'%J[j])
        
        ax[g].set_xlim(tau_m[g,0], tau_m[g,-1])
        ax[g].set_xscale('log')
        ax[g].grid(True,'both','both')
        ax[g].legend(loc='upper right',fontsize=fsz//1.5,ncol=np.size(J)//2)
    
    ax[0].set_ylabel(r'$\Delta \phi_{1} $ [\%]',fontsize = fsz)        
    ax[1].set_ylabel(r'$\Delta \phi_{2} $ [\%]',fontsize = fsz)        
    
    plt.xlabel(r'$\tau$ [optical length]',fontsize=fsz)
    
    filename = 'Tomatis2011_cvg_pCMFD_I%d_RMitr%d.pdf' %(I,itr)
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
        ax[g].plot(xi, dD1[g,:],label='CMFD')
        ax[g].plot(xi, dD2[0][g,:],label='dDp')
        ax[g].plot(xi, dD2[1][g,:],label='dDm')
        ax[g].set_xlim(xi[0], xi[-1])
        ax[g].grid(True,'both','both')
        ax[g].legend(loc='upper center',fontsize=fsz//1.5,ncol=3)
        ax[g].set_ylim(dD2[0][g,1]-0.1, dD2[1][g,-2]+0.2)
        plt.xticks(xi[0::50]) # x-axis valus represent fuel assembly
    plt.setp(ax[0].get_xticklabels(), visible=False)
    ax[0].set_ylabel(r'$\delta D_{1}^+ $ [AU]',fontsize = fsz)        
    ax[1].set_ylabel(r'$\delta D_{2}^- $ [AU]',fontsize = fsz)        
    plt.xlabel(r'$x$ [cm]',fontsize=fsz)
    
    filename = 'Tomatis2011_dD_I%d_RMitr%d.pdf' %(I,itr)
    fig.savefig(filename,dpi=150,bbox_inches='tight')
    os.system("pdfcrop %s %s"%(filename,filename))
#    plt.show()

# ============================================================== #
# ================== Flux differences (Error) ================== #
# ============================================================== #
if Plot_err: 

    flx_err_D = (flx_RM1[:,:,0] - flx_SN)/flx_SN*100
    flx_err_CMFD =(flx_RM1[:,:,-1] -flx_SN)/flx_SN*100
    flx_err_pCMFD =(flx_RM2[:,:,-1] -flx_SN)/flx_SN*100

    # Optical length
    tau_m = np.zeros((G,I))
    #tau_i = np.zeros((G,I))
    for g in range (0,G):
        tau_m[g,:] = xim * st[g,:]
        
    plt.figure(figsize=(14, 5)) 
    fig, ax = plt.subplots(G,figsize=(14, 6))
    
    for g in range (0,G):
        ax[g].set_prop_cycle(default_cycler)
        for i in range (0,np.size(I)):
            ax[g].tick_params(axis='both', which='major', labelsize=fsz)
            ax[g].tick_params(axis='both', which='minor', labelsize=fsz)
            ax[g].plot(tau_m[g,0:30], flx_err_D[g,0:30],label='$D_0$' if g == 0 else "")
            ax[g].plot(tau_m[g,0:30], flx_err_pCMFD[g,0:30],'r^',label='pCMFD' if g == 0 else "")
            ax[g].plot(tau_m[g,0:30], flx_err_CMFD[g,0:30],label='CMFD' if g == 0 else "")
            
            ax[g].grid(True,'both','both')
            #ax[g].set_xscale('log')
            ax[g].legend(loc='upper right',fontsize=fsz//1.5,ncol=3)
            ax[g].set_xlim(tau_m[g,0], tau_m[g,30])
    
    ax[0].set_ylabel(r'$\Delta \phi_{1} $ [AU]',fontsize = fsz)        
    ax[1].set_ylabel(r'$\Delta \phi_{2} $ [AU]',fontsize = fsz)        
    plt.xlabel(r'$\tau$ [optical length]',fontsize=fsz)
    
    filename = 'Tomatis2011_flx_err_I%d_RMitr%d.pdf' %(I,itr)
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    os.system("pdfcrop %s %s" %(filename,filename))    
    
# ============================================================== #
# =========================== k_eff ============================ #
# ============================================================== #
if Plot_k:     
    
    fig, ax = plt.subplots(figsize=(14,2.5))
    
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    It = np.array([0,1,2,5,10,25,50,75,100,150,200,250])
    
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
    
    for j in range(0,np.size(It)):
        ax.scatter(It[j], k_RM1[j], color='r', marker='x', label='CMFD' if j == 0 else "" )
        ax.scatter(It[j], k_RM2[j], color='b',marker='^', label='pCMFD' if j == 0 else "")
    
    ax.legend(loc='lower right',fontsize=fsz//2)
    
    filename = 'Tomatis2011_k_over_I%d_RMitr%d.pdf' %(I,itr)
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    os.system("pdfcrop %s %s" %(filename,filename))
    
# ================================================== #
# ================== Flux zoon-in ================== #
# ================================================== #
if plot_flx_zoom:
    
        # Optical length
    tau_m = np.zeros((G,I))
    for g in range (0,G):
        tau_m[g,:] = xim * st[g,:]
    
#    fig = plt.subplots(figsize=(14,11))
#    ax1 = plt.subplot(2,1,1)
#    ax1.tick_params(axis='both', which='major', labelsize=24)
#    ax1.tick_params(axis='both', which='minor', labelsize=24)
#    plt.plot(tau_m[0,199::5], flx_RM1[0,199::5,-1],':r^',label='RM-CMFD')
#    plt.plot(tau_m[0,199:],   flx_RM1[0,199:,0],'g--',label='$D_0$')
#    plt.plot(tau_m[0,199::5], flx_RM2[0,199::5,-1],'b-*',label='RM-pCMFD')
#    plt.plot(tau_m[0,199:],   flx_SN[0,199:],'k-',label='S16')
#    #plt.yticks(np.linspace(-0.1,0.1,5)) # x-axis valus represent fuel assembly
#    plt.setp(ax1.get_xticklabels(), visible=False)
#    plt.ylabel(r'$\phi_1$ $[AU]$',fontsize=24)
#    plt.grid(True,'both','both')
#    ax1.legend(loc='lower left',fontsize=20,ncol=4)
#    #plt.yticks(np.linspace(6, 50,5)) # apply the y-limits
#    plt.ylim(min(flx_SN[0,:]-0.5), max(flx_SN[0,:])+1.5)
#    # -------------------- 1st zoomin --------------------
#    #axins1 = zoomed_inset_axes(ax1, 3.3, loc="upper right")
#    axins1.plot(tau_m[0,379::5], flx_RM1[0,379::5,-1],':r^')
#    axins1.plot(tau_m[0,379:],   flx_RM1[0,379:,0],'g--')
#    axins1.plot(tau_m[0,379::5], flx_RM2[0,379::5,-1],'b-*')
#    axins1.plot(tau_m[0,379:],   flx_SN[0,379:],'k-')
#    axins1.set_xlim(tau_m[0,383], tau_m[0,-1]+0.05) # apply the x-limits
#    axins1.set_ylim(flx_SN[0,380]-0.45, flx_SN[0,-1]+0.5) # apply the y-limits
#    plt.grid(True,'both','both')
#    plt.yticks(visible=True,fontsize=15)
#    #plt.xticks()
#    #plt.xticks(np.linspace(xi1[380],xi1[-1],5),visible=True,fontsize=15)
#    axins1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
#
#    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#    mark_inset(ax1, axins1, loc1=3, loc2=4, fc="none", ec="0.6")
#    # -------------------- 2nd plot --------------------
#    ax2 = plt.subplot(2,1,2, sharex=ax1)
#    ax2.tick_params(axis='both', which='major', labelsize=24)
#    ax2.tick_params(axis='both', which='minor', labelsize=24)
#    plt.plot(tau_m[1,199::5], flx_RM1[1,199::5,-1],':r^',label='RM-CMFD')
#    plt.plot(tau_m[1,199:],   flx_RM1[1,199:,0],'g--',label='$D_0$')
#    plt.plot(tau_m[1,199::5], flx_RM2[1,199::5,-1],'b-*',label='RM-pCMFD')
#    plt.plot(tau_m[1,199:],   flx_SN[1,199:],'k-',label='S16')
#    plt.setp(ax2.get_xticklabels(), visible=True)
#    plt.setp(ax2.get_yticklabels(), visible=True)
#    #plt.xlim(xi1[200], xi1[-1])
#    plt.ylim(min(flx_SN[1,:]-0.1), max(flx_SN[1,:])+0.2) 
#    plt.ylabel(r'$\phi_2$ $[AU]$',fontsize=24)
#    plt.grid(True,'both','both')
#    ax2.legend(loc='lower left',fontsize=20,ncol=4)
#    plt.xlabel(r'$x$ $[cm]$',fontsize=28)
#    #plt.xticks(np.linspace(xi1[200],xi1[-1],10))
#    # -------------------- 2nd zoomin --------------------
#    axins2 = zoomed_inset_axes(ax2, 4.5, loc="upper right")
#    axins2.plot(tau_m[1,384::5], flx_RM1[1,384::5,-1],':r^')
#    axins2.plot(tau_m[1,384:],   flx_RM1[1,384:,0],'g--')
#    axins2.plot(tau_m[1,384::5], flx_RM2[1,384::5,-1],'b-*')
#    axins2.plot(tau_m[1,384:],   flx_SN[1,384:],'k-')
#    
#    #axins2.set_xlim(xi1[388], xi1[-1]) # apply the x-limits
#    plt.grid(True,'both','both')
#    plt.yticks(visible=True,fontsize=15)
#    #plt.xticks(np.linspace(xi1[385],xi1[-1],5),visible=True,fontsize=15)
#    axins2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
#    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#    mark_inset(ax2, axins2, loc1=3, loc2=4, fc="none", ec="0.6")
#    
#    
#    plt.subplots_adjust(hspace=.1)
#    
    
    
    
    plt.figure(figsize=(14, 5)) 
    fig, ax = plt.subplots(G,figsize=(14, 9))
    #axins = 
    #axins = fig.add_subplot(ax[G-1], 3.3)
    for g in range (0,G):
    
        ax[g].tick_params(axis='both', which='major', labelsize=fsz)
        ax[g].tick_params(axis='both', which='minor', labelsize=fsz)
        #ax[g].set_prop_cycle(default_cycler)
        
        ax[g].plot(tau_m[g,0:I//2:5], flx_RM1[g,0:I//2:5,-1],':r^',label='RM-CMFD')
        ax[g].plot(tau_m[g,0:I//2:5], flx_RM2[g,0:I//2:5,-1],'g-*',label='RM-pCMFD')
        ax[g].plot(tau_m[g,0:I//2],   flx_RM1[g,0:I//2,0],'b--',label='$D_0$')
        ax[g].plot(tau_m[g,0:I//2],   flx_SN[g,0:I//2],'k-.',label='S16')
        
        #ax[g].set_xticks(tau_m[g,I//2],tau_m[g,-1])
        ax[g].set_xlim(tau_m[g,0], tau_m[g,I//2])
        ax[g].grid(True,'both','both')
        ax[g].legend(loc='upper left',fontsize=fsz//1.5,ncol=4)
        ax[g].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        #plt.yticks(visible=True)
     
    ax[0].set_ylabel(r'$\phi_{1} $ [AU]',fontsize = fsz)        
    ax[1].set_ylabel(r'$\phi_{2} $ [AU]',fontsize = fsz)        
    plt.xlabel(r'$\tau$ [optical length]',fontsize=fsz)  
    
    
    axins0 = zoomed_inset_axes(ax[0], 2.8, loc=7)
    axins0.plot(tau_m[0,0:19:5], flx_RM1[0,0:19:5,-1],':r^')
    axins0.plot(tau_m[0,0:19:5], flx_RM2[0,0:19:5,-1],'g-*')
    axins0.plot(tau_m[0,0:19],   flx_RM1[0,0:19,0],'b--')
    axins0.plot(tau_m[0,0:19],   flx_SN[0,0:19],'k-.')
    axins0.set_xlim(tau_m[0,0], tau_m[0,19]) # apply the x-limits
    axins0.set_ylim(flx_SN[0,0], flx_SN[0,19]) # apply the y-limits
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax[0], axins0, loc1=2, loc2=3, fc="none", ec="0.4")
        
    
    plt.grid(True,'both','both')
    axins1 = zoomed_inset_axes(ax[1], 2.8, loc=7)
    axins1.plot(tau_m[1,0:19:5], flx_RM1[1,0:19:5,-1],':r^')
    axins1.plot(tau_m[1,0:19:5], flx_RM2[1,0:19:5,-1],'g-*')
    axins1.plot(tau_m[1,0:19],   flx_RM1[1,0:19,0],'b--')
    axins1.plot(tau_m[1,0:19],   flx_SN[1,0:19],'k:')
    axins1.set_xlim(tau_m[1,0], tau_m[1,19]) # apply the x-limits
    axins1.set_ylim(flx_SN[1,0], flx_SN[1,19]) # apply the y-limits
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax[1], axins1, loc1=2, loc2=3, fc="none", ec="0.4")
    
    plt.grid(True,'both','both')

   
    filename = 'Tomatis2011_flx_I%d_RMitr%d.pdf' %(I,itr)
    plt.savefig(filename,dpi=300,bbox_inches='tight')
    os.system("pdfcrop %s %s" %(filename,filename))


# ============================================================ #
# ===================== Delta rho [pcm] ====================== #
# ============================================================ #
if Plot_rho:     
    drho1 = np.zeros(itr+1)
    drho2 = np.zeros(itr+1)
    
    drho1 = (1./k_RM1 - 1/k_SN )*1e5
    drho2 = (1./k_RM2 - 1/k_SN )*1e5
    RMitr = np.linspace(1,250,250)
    fig, ax = plt.subplots(figsize=(14,2.5))
    
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.set_xlabel('Iteration No.',fontsize=18)
    ax.set_ylabel('$\Delta\\rho$ [pcm]',fontsize=18)
    ax.tick_params(axis='y')
    ax.set_axisbelow(True) 
    #plt.ylim(min(k_RM1)-0.0005, max(k_RM1)+0.0005)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    #plt.yticks(np.linspace(k_RM1[0],k_RM1[-1],4))
    
    ax.set_prop_cycle(default_cycler)
#    ax.scatter(RMitr[0:20], drho1[0:20], color='b', marker='x', label='CMFD')
#    ax.scatter(RMitr[20::5], drho1[20::5], color='b', marker='x')
#    ax.scatter(RMitr[0:20], drho2[0:20], color='r',marker='^', label='pCMFD')
#    ax.scatter(RMitr[20::5], drho2[20::5], color='r',marker='^')
    ax.plot(RMitr, drho1[1:], label='CMFD')
    ax.plot(RMitr, drho2[1:], label='pCMFD')
    ax.set_xscale('log')
    
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.legend(loc='upper right',fontsize=fsz//2)
    plt.xlim(RMitr[0], RMitr[-1]+10) 
    
    filename = 'Tomatis2011_rho_I%d_RMitr%d.pdf' %(I,itr)
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    os.system("pdfcrop %s %s" %(filename,filename))


# ============================================================ #
# =================== Left Boundary err % ==================== #
# ============================================================ #
if Plot_LB:     
    
    
    RMitr = np.linspace(1,250,250)
    
    plt.figure(figsize=(14, 5)) 
    fig, ax = plt.subplots(G,figsize=(14, 6))

    
    for g in range (0,G):
        flx_err_CMFD =(flx_RM1[g,0,1:] -flx_SN[g,0])/flx_SN[g,0]*100
        flx_err_pCMFD =(flx_RM2[g,0,1:] -flx_SN[g,0])/flx_SN[g,0]*100
        ax[g].set_prop_cycle(default_cycler)
        ax[g].tick_params(axis='both', which='major', labelsize=fsz)
        ax[g].tick_params(axis='both', which='minor', labelsize=fsz)
        ax[g].set_xlim(RMitr[0], RMitr[-1]) 
        ax[g].plot(RMitr, flx_err_CMFD,label='CMFD' if g == 0 else "")
        ax[g].plot(RMitr, flx_err_pCMFD,label='pCMFD' if g == 0 else "")
        ax[g].grid(True,'both','both')
        ax[g].legend(loc='upper right',fontsize=fsz//2,ncol=np.size(I))
    plt.setp(ax[0].get_xticklabels(), visible=False)
    ax[0].set_ylabel(r'$\Delta \phi_{1} $ [AU]',fontsize = fsz)        
    ax[1].set_ylabel(r'$\Delta \phi_{2} $ [AU]',fontsize = fsz)        
    plt.xlabel('Iteration No.',fontsize=fsz)
    
    
    filename = 'Tomatis2011_flx_err_LB_I%d_RMitr%d.pdf' %(I,itr)
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    os.system("pdfcrop %s %s" %(filename,filename))    
    