#!/usr/bin/env python3
'''
Ploting bare slab heterogeneous case by Rahnema-1997 for the ANE2020 paper
'''
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
Plot_MAX_err  = False # Under development
Plot_k        = False
plot_flx_zoom = False
Plot_dkeff    = False
Plot_LB       = True
I = 728 # No. of mesh points
itr = 250 # No. of iterations

# ------ Load flx_RM? ------
plot_RM = True
if plot_RM:
    # Uploading CMFD results
    path_RM_C1 = "kflx_Rahnema1997_CMFD_C1_LBC0RBC0_I728_itr250.npy"
    path_RM_C2 = "kflx_Rahnema1997_CMFD_C2_LBC0RBC0_I728_itr250.npy"
    path_RM_C3 = "kflx_Rahnema1997_CMFD_C3_LBC0RBC0_I728_itr250.npy"
    k_RM_C1, flx_RM_C1, xi_C1, st_C1, dD_C1 = np.load(path_RM_C1,allow_pickle=True)
    k_RM_C2, flx_RM_C2, xi_C2, st_C2, dD_C2 = np.load(path_RM_C2,allow_pickle=True)
    k_RM_C3, flx_RM_C3, xi_C3, st_C3, dD_C3 = np.load(path_RM_C3,allow_pickle=True)
    # Uploading pCMFD results
    path_pRM_C1 = "kflx_Rahnema1997_pCMFD_C1_LBC0RBC0_I728_itr250.npy"
    path_pRM_C2 = "kflx_Rahnema1997_pCMFD_C2_LBC0RBC0_I728_itr250.npy"
    path_pRM_C3 = "kflx_Rahnema1997_pCMFD_C3_LBC0RBC0_I728_itr250.npy"
    k_pRM_C1, flx_pRM_C1, xi_C1, st_C1, pdD_C1 = np.load(path_pRM_C1,allow_pickle=True)
    k_pRM_C2, flx_pRM_C2, xi_C2, st_C2, pdD_C2 = np.load(path_pRM_C2,allow_pickle=True)
    k_pRM_C3, flx_pRM_C3, xi_C3, st_C3, pdD_C3 = np.load(path_pRM_C3,allow_pickle=True)
    
    xi = xi_C1

# ------ Load flx_SN? ------
plot_SN = True
if plot_SN:
    N = 16  # No. of angles (S16)
    # Uploading SN results
    path_SN_C1 = "kflx_Rahnema1997_SN_CORE1LBC0RBC0_I728_N16.npy"
    path_SN_C2 = "kflx_Rahnema1997_SN_CORE2LBC0RBC0_I728_N16.npy"
    path_SN_C3 = "kflx_Rahnema1997_SN_CORE3LBC0RBC0_I728_N16.npy"
    ref_data1 = np.load(path_SN_C1,allow_pickle=True)
    ref_data2 = np.load(path_SN_C2,allow_pickle=True)
    ref_data3 = np.load(path_SN_C3,allow_pickle=True)
    k_SN_C1, flxm_C1 = ref_data1[0], ref_data1[1][:,0,:]
    k_SN_C2, flxm_C2 = ref_data2[0], ref_data2[1][:,0,:]
    k_SN_C3, flxm_C3 = ref_data3[0], ref_data3[1][:,0,:]
    
    # ------ Normalization of flxm - by reaction rate ------
    RR_SN_C1 = sum(np.multiply(flxm_C1,st_C1).flatten())
    RR_RM_C1 = sum(np.multiply(flx_RM_C1[:,:,-1],st_C1).flatten())
    flx_SN_C1 = (RR_RM_C1/RR_SN_C1)*flxm_C1
    
    RR_SN_C2 = sum(np.multiply(flxm_C2,st_C2).flatten())
    RR_RM_C2 = sum(np.multiply(flx_RM_C2[:,:,-1],st_C2).flatten())
    flx_SN_C2 = (RR_RM_C2/RR_SN_C2)*flxm_C2
    
    RR_SN_C3 = sum(np.multiply(flxm_C3,st_C3).flatten())
    RR_RM_C3 = sum(np.multiply(flx_RM_C3[:,:,-1],st_C3).flatten())
    flx_SN_C3 = (RR_RM_C3/RR_SN_C3)*flxm_C3
    

# ------ General ------
xim = (xi_C1[1:] + xi_C1[:-1]) / 2.# mid-cell points
G = st_C1.shape[0]

# ====================================================== #
# ======================== PLOT ======================== #
# ====================================================== #


# ====================================================== #
# =====================  plot flux ===================== #
# ====================================================== #

if plot_flx:
    core_typ = 'C2'
    flx_CMFD = flx_RM_C2
    flx_pCMFD = flx_pRM_C2
    flx_D = flx_RM_C2[:,:,0]
    flx_SN = flx_SN_C2
    xi = xi_C2
    
    fig, ax = plt.subplots(figsize=(13,7))
    ax.tick_params(axis='both', which='major', labelsize=fsz)
    ax.tick_params(axis='both', which='minor', labelsize=fsz)
    # Plot fast flux
    #ax.set_prop_cycle(default_cycler)
    ax.plot(xim, flx_CMFD[0,:,-1],'b-',label='CMFD')
    ax.plot(xim, flx_pCMFD[0,:,-1],'g:',label='pCMFD')
    ax.plot(xim, flx_D[0,:],'k--',label='$D_0$')
    ax.plot(xim, flx_SN[0,:],'r-.',label='S16')
    # Plot thermal flux
    ax.plot(xim, flx_CMFD[1,:,-1],'b-')
    ax.plot(xim, flx_pCMFD[1,:,-1],'g:')
    ax.plot(xim, flx_D[1,:],'k--')
    ax.plot(xim, flx_SN[1,:],'r-.')
    
    plt.xlabel(r'$x$ $[cm]$',fontsize=26)
    plt.ylabel(r'$\phi$ $[AU]$',fontsize=24) 
    plt.xlim(0, xi[-1])
    plt.ylim(0, )
    
    ax.annotate("Fast",fontsize=20, xy=(xi[150], flx_CMFD[0,150,-1]),  xycoords='data',
                xytext=(31,1.3), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=-0.2"))
    
    ax.annotate("Thermal",fontsize=20, xy=(xi[320], flx_CMFD[1,320,-1]),  xycoords='data',
                xytext=(50,1.2), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=-0.2"))
    
    ax.text(3.5, 2.3, core_typ , color='black', fontsize=20,
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xticks(xi[::104]) # x-axis valus represent fuel assembly
    ax.grid(True,'both','both')
    ax.legend(loc='upper right',fontsize=fsz//1.5,ncol=1)
    filename = 'Rahnema1997_flx_%s_%d_RMitr%d.pdf' %(core_typ,I,itr)
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    plt.show()
    os.system("pdfcrop %s %s" % (filename,filename))
    

# ====================================================== #
# ================  plot flux convergence ================ #
# ====================================================== #
if Plot_flx_cvg:
    
    RM = 'pCMFD'
    core_typ = 'C1'
    
    flx_CMFD = flx_RM_C1
    flx_pCMFD = flx_pRM_C1
    flx_D = flx_RM_C1[:,:,0]
    flx_SN = flx_SN_C1
    xi = xi_C1
    st = st_C1

    J = [0,1,2,5,10,25,50,150,250] # No. of itr to plot

    # Optical length
    tau_m = np.zeros((G,I))
    #tau_i = np.zeros((G,I))
    for g in range (0,G):
        tau_m[g,:] = xim * st[g,:]

    dev_flx_CMFD = np.zeros((G,I,np.size(J)))
    dev_flx_pCMFD = np.zeros((G,I,np.size(J)))
    
    # Calculate flux deviation
    for j in range (0,np.size(J)):
        if RM == 'CMFD':
            dev_flx_CMFD[:,:,j] = (flx_CMFD[:,:,J[j]] - flx_SN[:,:]) / flx_SN[:,:] *100
        elif RM == 'pCMFD':
            dev_flx_pCMFD[:,:,j] = (flx_pCMFD[:,:,J[j]] - flx_SN[:,:]) / flx_SN[:,:] *100

    
    plt.figure(figsize=(14, 5)) 
    fig, ax = plt.subplots(G,figsize=(14, 9))
    
    for g in range (0,G):
    
        ax[g].tick_params(axis='both', which='major', labelsize=fsz)
        ax[g].tick_params(axis='both', which='minor', labelsize=fsz)
        ax[g].set_prop_cycle(default_cycler)
        for j in range (0,np.size(J)):
            if RM == 'CMFD':
                ax[g].plot(tau_m[g,0:I//2], dev_flx_CMFD[g,0:I//2,j],label='it-%d'%J[j])
            elif RM == 'pCMFD':
                ax[g].plot(tau_m[g,0:I//2], dev_flx_pCMFD[g,0:I//2,j],label='it-%d'%J[j])
        
        ax[g].set_xlim(tau_m[g,0], tau_m[g,-1])
        ax[g].set_xscale('log')
        ax[g].grid(True,'both','both')
        ax[g].legend(loc='upper right',fontsize=fsz//1.5,ncol=5)
    
    ax[0].set_ylabel(r'$\Delta \phi_{1} $ [\%]',fontsize = fsz)        
    ax[1].set_ylabel(r'$\Delta \phi_{2} $ [\%]',fontsize = fsz)        
    
    plt.xlabel(r'$\tau$ [optical length]',fontsize=fsz)
    
    ax[0].text(12, 12, core_typ , color='black', fontsize=20,
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    filename = 'Rahnema1997_cvg_%s_%s_I%d_RMitr%d.pdf' %(RM,core_typ,I,itr)
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    #plt.show()
    os.system("pdfcrop %s %s" % (filename,filename))
        
# ====================================================== #
# ======================  plot dD ====================== #
# ====================================================== #
if Plot_dD: 
    
    core_typ = 'C3'
    dD = dD_C3
    pdD = pdD_C3
    xi = xi_C3
    
    plt.figure(figsize=(14, 5)) 
    fig, ax = plt.subplots(G,figsize=(14, 7))
    
    for g in range (0,G):
    
        ax[g].tick_params(axis='both', which='major', labelsize=fsz)
        ax[g].tick_params(axis='both', which='minor', labelsize=fsz)
        ax[g].set_prop_cycle(default_cycler)
        ax[g].plot(xi, dD[g,:],label='CMFD')
        ax[g].plot(xi, pdD[0][g,:],label='$pCMFD^+$')
        ax[g].plot(xi, pdD[1][g,:],label='$pCMFD^-$')
        ax[g].set_xlim(xi[0], xi[-1])
        ax[g].grid(True,'both','both')
        ax[g].legend(loc='upper center',fontsize=fsz//1.5,ncol=3)
        ax[g].set_ylim(pdD[0][g,1]-0.1, pdD[1][g,-2]+0.3)
        plt.xticks(xi[::104]) # x-axis valus represent fuel assembly
        ax[g].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    ax[0].set_ylabel(r'$\delta D_{1} $ [AU]',fontsize = fsz)        
    ax[1].set_ylabel(r'$\delta D_{2} $ [AU]',fontsize = fsz)        
    plt.xlabel(r'$x$ [cm]',fontsize=fsz)
    
    ax[0].text(4, 0.35, core_typ , color='black', fontsize=20,
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    
    filename = 'Rahnema1997_dD_%s_I%d_RMitr%d.pdf' %(core_typ,I,itr)
    fig.savefig(filename,dpi=150,bbox_inches='tight')
    os.system("pdfcrop %s %s"%(filename,filename))
#    plt.show()

# ============================================================== #
# ========================= Flux Error ========================= #
# ============================================================== #
if Plot_err:
    core_typ = 'C3'
    flx_CMFD = flx_RM_C3
    flx_pCMFD = flx_pRM_C3
    flx_D = flx_RM_C3[:,:,0]
    flx_SN = flx_SN_C3
    xi = xi_C3
    st = st_C3
    
    flx_err_D = (flx_D - flx_SN)/flx_SN*100
    flx_err_CMFD =(flx_CMFD[:,:,-1] -flx_SN)/flx_SN*100
    flx_err_pCMFD =(flx_pCMFD[:,:,-1] -flx_SN)/flx_SN*100

    # Optical length
    tau_m = np.zeros((G,I))
    
    for g in range (0,G):
        tau_m[g,:] = xim * st[g,:]
        
    plt.figure(figsize=(14, 5)) 
    fig, ax = plt.subplots(G,figsize=(14, 6))
    
    for g in range (0,G):
        ax[g].set_prop_cycle(default_cycler)
        for i in range (0,np.size(I)):
            ax[g].tick_params(axis='both', which='major', labelsize=fsz)
            ax[g].tick_params(axis='both', which='minor', labelsize=fsz)
            
            # ax[g].plot(tau_m[g,0:40], flx_err_CMFD[g,0:40],label='CMFD' if g == 0 else "")
            # ax[g].plot(tau_m[g,0:40], flx_err_pCMFD[g,0:40],'r:d',label='pCMFD' if g == 0 else "")
            # ax[g].plot(tau_m[g,0:40], flx_err_D[g,0:40],label='$D_0$' if g == 0 else "")
            # ax[g].set_xlim(tau_m[g,0], tau_m[g,39])
        
            ax[g].plot(xim[0:I//2:5], flx_err_pCMFD[g,0:I//2:5],'r:x',label='pCMFD' if g == 0 else "")
            ax[g].plot(xim[0:I//2], flx_err_D[g,0:I//2],label='$D_0$' if g == 0 else "")
            ax[g].plot(xim[0:I//2], flx_err_CMFD[g,0:I//2],label='CMFD' if g == 0 else "")
            ax[g].set_xlim(xi[0], xi[I//2+1])
            plt.xticks(xi[:I//2:104]) # x-axis valus represent fuel assembly
            ax[g].grid(True,'both','both')
            #ax[g].set_xscale('log')
            ax[g].legend(loc='upper right',fontsize=fsz//1.5,ncol=3)
            
    plt.setp(ax[0].get_xticklabels(), visible=False)
    ax[0].set_ylabel(r'$\Delta \phi_{1} $ [AU]',fontsize = fsz)        
    ax[1].set_ylabel(r'$\Delta \phi_{2} $ [AU]',fontsize = fsz)        
    plt.xlabel(r'$x$ [cm]',fontsize=fsz)
    
    filename = 'Rahname1997_flx_err_%s_I%d_RMitr%d.pdf' %(core_typ,I,itr)
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    os.system("pdfcrop %s %s" %(filename,filename))    

# ============================================================== #
# ========================= MAX Flux Error ========================= #
# ============================================================== #
if Plot_MAX_err: 

    core_typ = 'C3'
    flx_CMFD = flx_RM_C3
    flx_pCMFD = flx_pRM_C3
    flx_D = flx_RM_C3[:,:,0]
    flx_SN = flx_SN_C3
    xi = xi_C3
    st = st_C3
    
    max_flx_err_CMFD = (flx_CMFD[:,:,2:] - flx_CMFD[:,:,1:-1])/ flx_CMFD[:,:,1:-1]
    max_flx_err_pCMFD = (flx_pCMFD[:,:,2:] - flx_pCMFD[:,:,1:-1])/ flx_pCMFD[:,:,1:-1]

    RMitr = np.linspace(1,249,249)
    
    plt.figure(figsize=(14, 5)) 
    fig, ax = plt.subplots(G,figsize=(14, 6))
    
    for g in range (0,G):
        ax[g].set_prop_cycle(default_cycler)
        for i in range (0,np.size(I)):
            ax[g].tick_params(axis='both', which='major', labelsize=fsz)
            ax[g].tick_params(axis='both', which='minor', labelsize=fsz)
            ax[g].plot(RMitr, max_flx_err_CMFD[g,:],label='CMFD' if g == 0 else "")
            ax[g].plot(RMitr, max_flx_err_pCMFD[g,:],'r-d',label='pCMFD' if g == 0 else "")
            ax[g].grid(True,'both','both')
            ax[g].set_xscale('log')
            ax[g].set_yscale('log')
            ax[g].legend(loc='upper right',fontsize=fsz//1.5,ncol=3)
            #ax[g].set_xlim(tau_m[g,0], tau_m[g,39])
    
    ax[0].set_ylabel(r'$\epsilon^1_{\phi} $ [AU]',fontsize = fsz)        
    ax[1].set_ylabel(r'$\epsilon^2_{\phi} $ [AU]',fontsize = fsz)        
    plt.xlabel(r'Iteration No.',fontsize=fsz)
    
    filename = 'Rahnema1997_flx_MAX_err_%s_I%d_RMitr%d.pdf' %(core_typ,I,itr)
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    os.system("pdfcrop %s %s" %(filename,filename))    
# ============================================================== #
# =========================== k_eff ============================ #
# ============================================================== #
if Plot_k:     
    
    core_typ = 'C1'
    kCMFD = k_RM_C1
    kpCMFD = k_pRM_C1
    
    fig, ax = plt.subplots(figsize=(14,2.5))
    
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    It = np.array([0,1,2,5,10,25,50,75,100,125,150,175,200,225,250])
    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.set_xlabel('Iteration No.',fontsize=18)
    ax.set_ylabel('$k_{eff}$',fontsize=18)
    ax.tick_params(axis='y')
    ax.set_axisbelow(True) 
    plt.ylim(min(kCMFD)-0.00005, max(kCMFD)+0.0005)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.yticks(np.linspace(kCMFD[0],kCMFD[-1],4))
    
    for j in range(0,np.size(It)):
        ax.scatter(It[j], kCMFD[j], color='r', marker='x', label='CMFD' if j == 0 else "" )
        ax.scatter(It[j], kpCMFD[j], color='b',marker='^', label='pCMFD' if j == 0 else "")
    
    ax.legend(loc='upper right',fontsize=fsz//2,ncol=2)
    
    filename = 'Rahnema1997_keff_%s_%d_RMitr%d.pdf' %(core_typ,I,itr)
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
# ===================== Delta k [pcm] ====================== #
# ============================================================ #
if Plot_dkeff:     
    
    core_typ = 'C1'
    
    kCMFD = k_RM_C1
    kpCMFD = k_pRM_C1
    k_SN = k_SN_C1
    
    dkCMFD = np.zeros(itr+1)
    dkpCMFD = np.zeros(itr+1)
    
    dkCMFD = (1./kCMFD - 1/k_SN )*1e5
    dkpCMFD = (1./kpCMFD - 1/k_SN )*1e5
    
    RMitr = np.linspace(1,250,250)
    fig, ax = plt.subplots(figsize=(14,3))
    
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.set_xlabel('Iteration No.',fontsize=18)
    ax.set_ylabel('$\Delta k_{eff}$ [pcm]',fontsize=18)
    ax.tick_params(axis='y')
    ax.set_axisbelow(True) 
    #plt.ylim(min(k_RM1)-0.0005, max(k_RM1)+0.0005)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    
    ax.set_prop_cycle(default_cycler)
    #ax.plot(RMitr, dkCMFD[0], label='$D_0$')
    ax.plot(RMitr, dkCMFD[1:], label='CMFD')
    ax.plot(RMitr, dkpCMFD[1:], label='pCMFD')
    ax.set_xscale('log')
    
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.legend(loc='lower right',fontsize=fsz//2,ncol=2)
    plt.xlim(RMitr[0], RMitr[-1]) 
    
    filename = 'Rahnema1997_dkeff_%s_%d_RMitr%d.pdf' %(core_typ,I,itr)
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    os.system("pdfcrop %s %s" %(filename,filename))


# ============================================================ #
# =================== Left Boundary err % ==================== #
# ============================================================ #
if Plot_LB:     
    
    core_typ = 'C3'
    flx_CMFD = flx_RM_C3
    flx_pCMFD = flx_pRM_C3
    flx_SN = flx_SN_C3
    
    RMitr = np.linspace(1,250,250)
    
    plt.figure(figsize=(14, 5)) 
    fig, ax = plt.subplots(G,figsize=(14, 6))

    
    for g in range (0,G):
        flx_err_CMFD =(flx_CMFD[g,0,1:] -flx_SN[g,0])/flx_SN[g,0]*100
        flx_err_pCMFD =(flx_pCMFD[g,0,1:] -flx_SN[g,0])/flx_SN[g,0]*100
        
        ax[g].set_prop_cycle(default_cycler)
        ax[g].tick_params(axis='both', which='major', labelsize=fsz)
        ax[g].tick_params(axis='both', which='minor', labelsize=fsz)
        ax[g].set_xlim(RMitr[0], RMitr[-1]) 
        ax[g].plot(RMitr, flx_err_CMFD,label='CMFD' if g == 0 else "")
        ax[g].plot(RMitr[0::5], flx_err_pCMFD[0::5],'r:x',label='pCMFD' if g == 0 else "")
        ax[g].grid(True,'both','both')
        ax[g].legend(loc='upper right',fontsize=fsz//2,ncol=2)
        ax[g].set_xscale('log')
    plt.setp(ax[0].get_xticklabels(), visible=False)
    ax[0].set_ylabel(r'$\Delta \phi_{1} $ [AU]',fontsize = fsz)        
    ax[1].set_ylabel(r'$\Delta \phi_{2} $ [AU]',fontsize = fsz)        
    plt.xlabel('Iteration No.',fontsize=fsz)
    
    
    filename = 'Rahnema1997_flx_err_LB_%s_I%d_RMitr%d.pdf' %(core_typ,I,itr)
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    os.system("pdfcrop %s %s" %(filename,filename))    
    