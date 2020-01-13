#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A simple Discrete Ordinates S$_N$ computer code for multi energy group neutron
calculations in the slab geometry. This program is only intended for basic
education purposes of students at the Master program in Nuclear Engineering.
The solving algorithm is based on the transport sweep and the common diamond
differences scheme, see chapter 3 of [1].

Cross sections data of different materials must be entered according to
the following dictionary. Please, use zeroed arrays for chi and nsf for
non-fissile media.

xs_media = {
    'name_of_media_1':{
         'st': np.array with G elements
         'ss': np.array with G*G*(anisotropy_order + 1) elements
        'chi': np.array with G elements
        'nsf': np.array with G elements
    }
    'name_of_media_2':{ ... }
    ...
}

A list of lists is used to assign the materials to the geometry cells
of the mesh, like for example:

media = [
    ['name_of_media_1', x_right_medium_1],
    ['name_of_media_2', x_right_medium_2],
    ...
    ['name_of_media_N', x_right_medium_N]
]
where by definition it is always x_left_medium_1 = 0, and
x_right_medium_(i) = x_left_medium_(i+1) for all i < N.


...

Bibliography
------------

[1] Lewis, Elmer E and Miller, Warren F, Computational methods of neutron
    transport (1984).
}
"""

# Owned
__author__ = "Daniele Tomatis"
__email__ = "daniele.tomatis@cea.fr"
__company__ = "DEN/DANS/DM2S/SERMA/LPEC CEA Saclay, France"
__date__ = "03/04/2019"
__license__ = "MIT License"
__copyright__ = "Copyright 2019, CEA Saclay France"
__status__ = "Dev"
__credits__ = [__author__]
__version__ = "0.1.0"

import logging as lg
import sys, os
import warnings as wrn
import numpy as np
from scipy.special import roots_legendre, legendre

# INFO: the following makes execution working only when running the
# module within this same directory
sys.path.append('..XFD1dMGdiff'.replace('X', os.path.sep))
# WARNING: before importing snmg1dslab, check the pythonpath for the
# following import to properly work.
from FDsDiff1D import input_data, solver_options, unfold_xs


def get_dirs_and_weights(N=None, L=0, qtype="Gauss-Legendre"):
    """Compute directions and weights of the S_N quadrature set, and the
    (polynomial) functions for the flux expansion."""
    if N == None: raise ValueError("Missing input nb. of directions.")
    if (N <= 0): raise ValueError("Invalid N <= 0")
    print("Input nb. of directions is "+str(N))
    Nh, isodd = N//2, N%2 == 1
    # mn and wn are N directions and weights
    if(N == 1):
        mn, wn = np.full(1, 0.), np.full(1, 1.)
    else:
        if isodd: wrn.warn("Odd nb. of directions detected")
        if qtype == "Gauss-Legendre":
            mn, wn = roots_legendre(N, False)
            pl = [legendre(l) for l in range(L+1)]
            # remind that the weights must sum to 2
        else: raise ValueError("Unsupported quadrature type")
    return mn, wn, pl


class quad_data:
    """Object collecting quadrature data."""
    
    def __init__(self, N=16, L=0, qtype='Gauss-Legendre'):
        self.N, self.L, self.qtype = N, L, qtype
        self.mn, self.wn, self.pl = get_dirs_and_weights(N, L, self.qtype)
        #self.check_input()


def differencing_by_SC(aflx, bflx, hm):
    """Differencing scheme similar to step characteristics; aflx and bflx are
    the cell-averaged and the cell-edge fluxes, respectively; hm is dx over the
    direction cosine mn. See Azmy, lowest order nodal integral method."""
    betan = np.exp(- hm)
    betan = (1 + betan) / (1 - betan) - 2. / hm
    betan *= np.sign(hm)  # to account for both +/- directions
    return (2 * aflx - (1 - betan) * bflx) / (1 + betan)


def differencing_by_DD(aflx, bflx):
    """Differencing scheme based on unweighted diamond difference; aflx and
    bflx are the cell-averaged and the cell-edge fluxes, respectively."""
    return 2 * aflx - bflx


def tr_sweep(dx, st, qli, qdata, lbc=0, rbc=0):
    """Get new flux moments by the transport sweep. Boundary conditions: 0 for
    vacuum, 1 specular reflection, periodic translation otherwise. r/lbc, right
    or left."""
    mn, wn, pl = qdata.mn, qdata.wn, qdata.pl # qdata, quadrature data
    N = len(mn)
    L, I = qli.shape
    Nh = N // 2
    maxits = 4 if (lbc == rbc == 1) else 2

    # compute the angular source once
    qmi = np.zeros((N,I),)
    for j, m in enumerate(mn):
        #qmi[j,:] = np.dot( pl[:](m), qli )
        for l in range(L):
            qmi[j,:] += qli[l,:] * pl[l](m)

    aflx, afbc = np.zeros_like(qmi), np.zeros((N,2),)
    for it in range(maxits):
        for m in range(Nh):
            # positive directions
            mp = N-1-m
            if lbc == 0:
                aflxm1 = 0.
            elif lbc == 1:
                aflxm1 = afbc[m,0]
            else:
                aflxm1 = afbc[mp,1]
            for i in range(I): # advance forward, abs(mn[mp]) = mn[mp]
                aflx[mp, i] = qmi[mp, i] + 2 * mn[mp] * aflxm1
                aflx[mp, i] /= (2 * mn[mp] / dx[i] + st[i])
                # aflxm1 = 2 * aflx[mp, i] / dx[i] - aflxm1  # simple DD scheme
                aflxm1 = differencing_by_DD(aflx[mp, i] / dx[i], aflxm1)
                # aflxm1 = differencing_by_SC(aflx[mp, i] / dx[i], aflxm1,
                #                             st[i] * dx[i] / mn[mp])
                if aflxm1 < 0.: raise ValueError("negative aflx!")
            afbc[mp,1] = aflxm1

            # negative directions
            if rbc == 0:
                aflxm1 = 0.
            elif rbc == 1:
                aflxm1 = afbc[mp,1]
            else:
                aflxm1 = afbc[m,0]
            for i in range(I-1,-1,-1): # advance backward, abs(mn[m]) = mn[mp]
                aflx[m, i] = (qmi[m, i] + 2 * mn[mp] * aflxm1)
                aflx[m, i] /= (2 * mn[mp] / dx[i] + st[i])
                # aflxm1 = 2 * aflx[m, i] / dx[i] - aflxm1  # simple DD scheme
                aflxm1 = differencing_by_DD(aflx[m, i] / dx[i], aflxm1)
                # aflxm1 = differencing_by_SC(aflx[m, i] / dx[i], aflxm1,
                #                             st[i] * dx[i] / mn[m])
                if aflxm1 < 0.: raise ValueError("negative aflx!")
            afbc[m,0] = aflxm1

        if it == 0:
            # leave the loop if vacuum at both sides or reflection at right
            if (lbc == 0) and (0 <= rbc <= 1): break

    # compute the new flux moments
    flxm = np.zeros((L,I),)
    for l in range(L):
        flxm[l,:] = np.dot(wn * pl[l](mn), aflx)
    return flxm


def solve_inners(dx, st, ss, qdata, flxm, nsff, lbc, rbc, \
                 itsmax = 10, toll=1.e-5, vrbs=True):
    "Solve the inner iterations on groups to update the scattering source."
    G, L, I = flxm.shape

    # compute the source terms, starting from scattering
    src = np.zeros_like(flxm)
    for g in range(G):
        for l in range(L):
            src[g,l,:] = np.sum(ss[g,:,l,:] * flxm[:,l,:], axis=0) \
                       * (2 * l + 1.) / 2.
    # add the fission contribution
    src[:,0,:] += nsff

    emax_inners, it = 1.e+20, 0
    while (emax_inners > toll) and (it < itsmax):
        old_flxm = np.array(flxm, copy=True)

        for g in range(G):
            # apply the transport sweep per each group equation
            flxm[g,:,:] = tr_sweep(dx, st[g,:], src[g,:,:], qdata, lbc, rbc)

            # update the scattering source
            for gg in range(g,G):
                for l in range(L):
                    src[gg,l,:] += (2 * l + 1.) / 2. * ss[gg,g,l,:] \
                        * (flxm[g,l,:] - old_flxm[g,l,:])

        # compute the residual error
        eflx = np.where( flxm > 0., 1. - old_flxm / flxm, old_flxm - flxm )
        emax_inners = np.max(np.abs(eflx))
        it += 1
        if vrbs: print("it ={:3d}, emax = {:13.6g}".format(it,emax_inners))
    return flxm


def compute_fiss_src(nsf, chi, flxm):
    "Compute the fission source"
    nsff = np.sum(nsf * flxm[:, 0, :], axis=0) # sum on groups!
    # apply the emission spectrum, and add to the final source
    return 0.5 * chi * nsff


def solve_outers(dx, xs, qdata, flxm, k, oitsmax=20, toll=1.e-5, 
                 lbc=0, rbc=0, vrbs=True):
    "Solve the criticality problem by outer iterations."
    it, emax_outers = 0, 1.e+20
    st, ss, chi, nsf = xs  # cross sections data
    mn, wn, pl = qdata.mn, qdata.wn, qdata.pl # quadrature data
    #flxn = np.sum(flxm[:,0,:]) # volume-norm
    # (initial) compute the fission contribution
    nsff = compute_fiss_src(nsf, chi, flxm)
    while (it < oitsmax) and (emax_outers > toll):
        old_flxm, old_k = np.array(flxm, copy=True), k

        # solve the inner iterations taking on the scattering source
        flxm = solve_inners(dx, st, ss, qdata, flxm, nsff / k, lbc, rbc, \
                        itsmax = 100, toll=toll, vrbs=False)

        # compute the fission contribution
        old_nsff = np.array(nsff, copy=True)
        nsff = compute_fiss_src(nsf, chi, flxm)

        # get the new estimate of the eigenvalue
        k *=  np.sum(flxm[:,0,:] * nsff) / np.sum(flxm[:,0,:] * old_nsff)
        # compute the residual error
        e_k = 1.e+5 * (k - old_k)
        eflx = np.where( flxm > 0., 1. - old_flxm / flxm, old_flxm - flxm )
        emax_outers = np.max(np.abs(eflx))
        it += 1
        if vrbs:
            line = "it ={:3d}, k = {:6.5f}, e_k = {:6.1f}, eflx = {:13.6g}"
            print(line.format(it, k, e_k, emax_outers))
    return flxm, k


def solve_sn(idata, slvr_opts, qdata):
    '''Run SN solver.'''
    lg.info("Prepare input data")
    # xs = [st, ss, chi, nsf]
    xs = unfold_xs(idata, isotropic_scattering=False, diff_calc=False)
    # initialize the cell-integrated flux moments
    # (which are the only unknowns stored in memory at this higher level)
    # and the multiplication factor (eigenvalue)
    flxm = np.zeros((idata.G, qdata.L+1, idata.I),)
    flxm[:, 0, :], k = 1., 1.
    lg.info("-o"*22)

    # start SN iterations
    lg.info("Start the SN iterations")
    flxm, k = solve_outers(idata.Di, xs, qdata, flxm, k, 
                           slvr_opts.oitmax, slvr_opts.toll,
                           idata.LBC, idata.RBC)
        
    lg.info("-o"*22)
    lg.info("*** NORMAL END OF CALCULATION ***")
    return flxm, k


if __name__ == '__main__':

    # input general data
    I = 100 # nb. of cells
    a = 21.5 / 2. # slab width (cm)
    N = 16 # nb. of directions
    G = 2 # nb. of energy groups
    # L, maximum level of scattering anisotropy
    # (so that we will only compute flux moments up to the order L)
    L = 0

    # boundary conditions
    lbc, rbc = 1, 0  # left / right b.c.

    # input cross section data
    chi = np.array([1., 0.])
    st = np.array([5.3115e-1, 1.30058e+0])
    nsf = np.array([7.15848e-3, 1.41284e-1])
    ss0 = np.array([[5.04664e-1, 2.03884e-3], [1.62955e-2, 1.19134e+0]])
    finf = np.linalg.solve(np.diag(st) - ss0, chi)
    kinf = np.dot(nsf, finf)
    print("The k-infty of the homogeneous infinite slab is {:8.6f}.".format( \
        kinf))
    # (reference) k_\infty = 1.07838

    # fill input containers with the cross sections
    chi = np.zeros((G, I),)
    chi[0, :] = 1.
    st = np.tile(st, (I, 1)).T
    nsf = np.tile(nsf, (I, 1)).T
    ss = np.zeros((G, G, L+1, I),)
    for i in range(I): ss[:,:,0,i] = ss0
    xs = [st, ss, chi, nsf]

    # define the spatial mesh
    dx = np.linspace(0., a, I+1)
    dx = dx[1:] - dx[:-1]

    # calculate the directions using the Gauss-Legendre quadrature
    qdata = quad_data(N, L)

    # initialize the cell-integrated flux moments
    # (which are the only unknowns stored in memory at this higher level)
    # and the multiplication factor (eigenvalue)
    flxm = np.zeros((G, L+1, I),)
    flxm[:, 0, :], k = 1., 1.

    # solve the criticality problem by power iterations
    flxm, k = solve_outers(dx, xs, qdata, flxm, k, oitsmax=100, toll=1.e-7,
                           lbc=lbc, rbc=rbc)
    basefilen = "LBC%dRBC%d_I%d_N%d" % (lbc, rbc, I, N)
    # np.save(basefilen + ".npy", np.array([k, flxm]), allow_pickle=True)
    np.savez(basefilen + ".npz", k=k, flxm=flxm)
