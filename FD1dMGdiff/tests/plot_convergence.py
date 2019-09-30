#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from data2Ghomog import *

from cycler import cycler
default_cycler = (cycler(color=['r', 'g', 'b', 'y']) *
                  cycler(linestyle=['-', '--', ':', '-.']))

plot_eflxconv = False
plot_kflxconv = False

I = 50
N = 64
xi = np.linspace(0, L, I+1)  # equidistant mesh
xim = lambda x: (x[1:] + x[:-1]) / 2. # mid-points

# fetch reference solution
ref_flx_file = "../../SNMG1DSlab/LBC1RBC0_I%d_N%d.npz" % (I, N)
ref_data = np.load(ref_flx_file)
k_SN, flx_SN = ref_data['k'], ref_data['flxm']
# normalize the reference flux
flx_SN *= (I * G) / np.sum(flx_SN[:,0,:])
print("I = %d" % I)
print("S%d k = " % N + str(k_SN))

# diff_saved = np.genfromtxt("kflx_LBC2RBC0_I%d.dat" % I, delimiter=',')
# ks = diff_saved[:,0]
# flx_D = diff_saved[:,1:].reshape(len(ks), 2, I)
ks, flx_D = np.load("kflx_LBC2RBC0_I%d.npy" % I, allow_pickle=True)
flx_D = np.moveaxis(flx_D, -1, 0)
nb_its = (ks > 0).sum()
print("nb. RM iterations %d" % (nb_its - 1))
print("FD k = " + str(ks[0]))

Drho = lambda k1, k2: (1. / k2 - 1. / k1) * 1.e+5

kRM = ks[nb_its - 1]
print("RM k = " + str(kRM))
print("rho diff S%d/RM = %.2f pcm" % (N, Drho(k_SN, kRM)))
print("rho diff S%d/DF = %.2f pcm" % (N, Drho(k_SN, ks[0])))
# residual error through the iterations
eflx = np.where(flx_D[:-1,:,:] > 0., 1. - flx_D[1:,:,:] / flx_D[:-1,:,:],
                flx_D[1:,:,:] - flx_D[-1:,:,:]) * 100.

Drhos = Drho(k_SN, ks)

# get the x axis as dimensionless distance x*st from the outer boundary
xim0 = xim(xi)
xim0 = np.outer(st, xim0[-1] - xim0)

print("\nCheck the last and first 6 F/T flux values in the slab.")
print(" --- right/fast")
print(flx_D[0,0,-6:])
print(flx_D[-1,0,-6:])
print(flx_SN[0,0,-6:])
print(" --- left/fast")
print(flx_D[0,0,:6])
print(flx_D[-1,0,:6])
print(flx_SN[0,0,:6])
print(" --- right/thermal")
print(flx_D[0,1,-6:])
print(flx_D[-1,1,-6:])
print(flx_SN[1,0,-6:])
print(" --- left/thermal")
print(flx_D[0,1,:6])
print(flx_D[-1,1,:6])
print(flx_SN[1,0,:6])

if plot_kflxconv:
    fig, ax = plt.subplots()
    eflxFpc = (flx_D[:,0,-1] / flx_SN[0,0,-1] - 1.) * 100.
    eflxTpc = (flx_D[:,1,-1] / flx_SN[1,0,-1] - 1.) * 100.
    ax.plot(eflxFpc, 'k-' , label=r'$\phi_F$ diff.')
    ax.plot(eflxTpc, 'k--', label=r'$\phi_T$ diff.')
    ax.set_ylabel(r"$\max (\phi^{(it)} / \phi_{S%d} - 1)$ (%)" % N)
    ax.set_xlabel("Its.")
    ax.legend()
    axR = ax.twinx()
    axR.plot(Drhos, 'r:')
    axR.set_ylabel(r'$\rho^{(it)} - \rho_{S%d}$ (pcm)' % N, color='r')
    axR.tick_params('y', colors='r')

    plt.tight_layout()
    fig.savefig('kflxconv.pdf')
    plt.close(fig)



if plot_eflxconv:

    # idx = (np.abs(xim0 - 10)).argmin()
    idx = -1

    fig, (ax1, ax2) = plt.subplots(2, 1) #, sharex=True)
    # ax2.set_label(r'$x \in [a/2, a]$ (cm)')
    # ax2.set_xlabel(r'$i = 1, \ldots, I$')
    ax2.set_xlabel(r'$\sigma_t(a - x)$')
    ax1.set_title(r'$1 - \phi^{(new)} / \phi^{(old)}$ (%)')
    ax1.set_ylabel('Fast flux')
    ax2.set_ylabel('Thermal flux')
    ax1.set_prop_cycle(default_cycler)
    ax2.set_prop_cycle(default_cycler)

    its = range(10,nb_its-2,10)
    its = np.array([1, 2, 5, 10, 25, 50])

    for i in its:
    # for i in range(10):
        ax1.semilogx(xim0[0,:idx], eflx[i-1,0,:idx], label='it=%d'%i)
        ax2.semilogx(xim0[1,:idx], eflx[i-1,1,:idx], label='it=%d'%i)
    # for i in range(5):
    #     ax1.plot(eflx[i,0,:], label='it=%d'%i)
    #     ax2.plot(eflx[i,1,:], label='it=%d'%i)

    # ax1.set_yscale('symlog')
    # ax2.set_yscale('symlog')
    # ax1.set_xscale('log')
    # ax2.set_xscale('log')
    ax2.legend(ncol=3)
    plt.tight_layout()
    fig.savefig("eflxconv.pdf")
    plt.close(fig)
