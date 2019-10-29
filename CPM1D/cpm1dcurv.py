#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This module is dedicated to the calculation of collision, escape and transfer
probabilities in 1D curvilinear geometries.

.. note:: We use algo609 by D.E. AMOS, JUNE, 1982 for Bickley-Naylor functions.
          To avoid exponential underflow, the function at x is calculated by
          KMODE=2, which requires then to divide the obtained value by exp(x)
          to have the target value Ki_n(x) (n is the order of Ki). This is not
          necessary when calculating Ki_n(0) for any order n.
"""
import os
import sys
import logging as lg
import numpy as np
from scipy.special import expn as En
from scipy.special import roots_jacobi
from KinPy.algo609 import dbskin as Ki

__title__ = "Calculation of CP in 1D curvilinear geometries"
__author__ = "D. Tomatis"
__date__ = "30/08/2019"
__version__ = "1.0.0"


# useful constants
Ki1_at_0, Ki2_at_0, Ki3_at_0 = Ki(0.,1,1,1)[0], Ki(0.,2,1,1)[0], Ki(0.,3,1,1)[0]
geometry_type = "cylinder"  # cylinder or sphere only
max_float = np.finfo(float).max  # float is float64!
min_float = np.finfo(float).min
np.set_printoptions(precision=5)


def calculate_volumes(r, geometry_type="cylinder"):
    "Compute the cell volumes with the input 1d mesh in r."
    V = r[1:] - r[:-1]  # slab as default case
    if geometry_type != "slab":
        V *= np.pi
        if geometry_type == "cylinder":
            V *= (r[1:] + r[:-1])
        elif geometry_type == "sphere":
            V *= 4. / 3. * (r[1:]**2 + r[1:]*r[:-1] + r[:-1]**2)
        else:
            raise ValueError("Invalid input geometry_type")
    return V


def y(r, h):
    "Obtain the right side from the hypotenuse r and the other side h."
    if np.any(r < h):
        inputs = (" r=%.3f, h=" % r) + str(h)
        raise RuntimeError("negative side detected!" + inputs)
    rc = r if not hasattr(h, '__len__') else np.tile(r, len(h))
    return np.sqrt(rc**2 - h**2)


def get_GLquadrature(k, a=0., b=1., quad='Gauss-Legendre'):
    """Get the weights and points of the k-th order Gauss-Legendre quadrature.
    Return the points shifted in (a, b)."""

    bma = b - a
    if quad == 'Gauss-Legendre':
        # Gauss-Legendre (default interval is [-1, 1])
        x, w = np.polynomial.legendre.leggauss(k)
    elif quad == 'Gauss-Jacobi':
        # Gauss-Jacobi, overcoming the singularity at the boundary
        # warning: weights and points differ from Hebert's book!
        # the first parameters used by Hebert is unknown
        x, w = roots_jacobi(k, -0.5, 0.)
    elif quad == 'Rectangular':
        x, w = np.linspace(-1., 1., k + 1), np.full(k, 2. / k)
        x = (x[1:] + x[:-1]) / 2.
    else:
        raise ValueError("Unknown type of numerical quadrature.")

    # Translate x values from the interval [-1, 1] to [a, b]
    return 0.5*(x + 1) * bma + a, w * bma / 2.


def calculate_tracking_data(r, ks, sphere=False):
    """Calculate the quadrature weights for each ring described by r[i-1] and
    r[i]. The array r contains I+1 radii defining I rings in the cylinder.
    ks contains the k-th order of the quadrature in each ring. So ks has
    I values. The tracks are stored in a 1D array starting from the those
    which are closer to the center."""
    I = len(r) - 1
    lg.info("Nb of rings in the cylinder I = %d" % I)
    if len(ks) != I:
        raise ValueError("Invalid input, len(ks) != I = %d" % I)
    if np.any(ks <= 0):
        raise ValueError("Invalid input, ks has zero or negative values")
    if not np.all(np.diff(r) > 0):
        raise ValueError("Invalid input, r is not in increasing order")
    # the total number of tracks is sum_{i=0}^{I-1} ks(i)*(I-i)
    nb_tracks = I * sum(ks) - np.dot(np.arange(I), ks)
    tracks = np.zeros(nb_tracks)

    # array for n dot Omega_p
    ndotOp = np.zeros_like(tracks)
    # quadrature points and weights
    nb_total_quadrature_points = np.sum(ks)
    rks = np.zeros(nb_total_quadrature_points)
    wks = np.zeros_like(rks)

    pos, beg = 0, 0
    for i in range(I):
        # Dr = r[i+1] - r[i]
        # print('i=',i, 'ks(i)=', ks[i])
        rng = np.arange(beg, beg + ks[i])
        beg += ks[i]  # update value for next ring
        # get points and weights in the i-th ring
        rks[rng], wks[rng] = get_GLquadrature(ks[i], a=r[i], b=r[i+1],
                                              quad='Gauss-Legendre')

        for k in range(ks[i]):
            lt = 0.
            for j in np.arange(i, I) + 1:
                tracks[pos] = y(r[j], rks[k])
                ndotOp[pos] = tracks[pos] / r[j]
                tracks[pos], lt = tracks[pos] - lt, tracks[pos]
                pos += 1

    # define functions to address the elements in 1D arrays
    # get the j-th element along the k-th line in the i-th ring (for tracks)
    jki = lambda j, k, i: np.dot(ks[:i], np.arange(I,I-i,-1)) + k * (I-i) + j
    # get the k-th element in the i-th ring (for wks or xks)
    ki = lambda k, i: sum(ks[:i]) + k

    return {"quadrature_data": (rks, wks), "tracks": tracks,
            "cosines": ndotOp, "tracks_in_rings": ks,
            "address_funcs": (jki, ki)}


def calculate_escape_prob(r, st, tr_data, geometry_type="cylinder"):
    """Calculate the reduced escape probabilities in the given geometry type.
    These quantities are needed to calculate the partial currents in the 1D
    curvilinear frames."""
    lg.info("Calculate the reduced escape probabilities.")
    I = r.size - 1  # nb of rings
    if not np.all(np.diff(r) > 0):
        raise ValueError("Input values in r are not in increasing order")
    if st.size != I:
        raise ValueError("size mismatch of input sigma_t")

    # unpack tracking data
    ks = tr_data["tracks_in_rings"]
    hks, wks = tr_data["quadrature_data"]
    tracks = tr_data["tracks"]
    jki, ki = tr_data["address_funcs"]

    # compute the optical lengths by multiplying the tracks by the
    # corresponding total cross sections (remind that the tau data relate
    # still to a rectangular sector in the circle).
    tau = np.array(tracks, copy=True)
    for i in range(I):
        for k in range(ks[i]):
            for j in range(I-i):
                tau[jki(j,k,i)] *= st[i]

    # select the kernel function according to the geometry type
    if geometry_type == "cylinder":
        # the following function definition should be improved in future
        # versions to achieve higher performances
        def K(ell):
            if not hasattr(ell, '__len__'):
                Ki3_KODE2_M1 = Ki(ell, 3, 2, 1)
                if Ki3_KODE2_M1[1] != 0:
                    raise RuntimeError("error in DBSKIN")
                else:
                    Ki3_at_ell = Ki3_KODE2_M1[0]
            else:
                Ki3_KODE2_M1 = [Ki(l, 3, 2, 1) for l in ell]
                if any([v[1] != 0 for v in Ki3_KODE2_M1]):
                    raise RuntimeError("error in DBSKIN")
                else:
                    # the Ki part should be improved for higher efficiency
                    Ki3_at_ell = np.asarray([v[0][0] for v in Ki3_KODE2_M1])
            return Ki3_at_ell / np.exp(ell)
        K_at_0 = Ki3_at_0
        w = lambda wk, r, hk: wk * y(r, hk)
    elif geometry_type == "sphere":
        K = lambda ell: np.exp(ell)
        K_at_0 = 1.
        w = lambda wk, r, hk: wk * hk * y(r, hk)
    else:
        raise ValueError("Unsupported geometry type")

    # calculate the recuced escape probabilities varepsilon (times the volume)
    # remind that the current vanishes at the center, and likewise the partial
    # currents (assumption, but it shall be enough to say that they're equal!).
    vareps = np.zeros((I, I, 2),)
    # reference the partial currents to the main data container
    varepsp = vareps[:,:,0]  # first index is i, second index is j (sources)
    varepsm = vareps[:,:,1]  # (defined as negative!)
    # varepsp is for escape whereas varepsm is for in-scape
    cumsum_ks = np.insert(np.cumsum(ks), 0, 0)
    get_ks_in_ring = lambda i0: np.arange(cumsum_ks[i0], cumsum_ks[i0+1])
    for i in range(I):
        idx = get_ks_in_ring(i)
        hk, wk = hks[idx], wks[idx]
        wgts = w(wk, r[i+1], hk)

        ji_allk = lambda j, i0=i: \
            np.asarray([jki(j,k,i0) for k in range(ks[i0])])

        # escape within (r_i-1/2, r_i+1/2)
        # contribution from the i-th ring through its outer surface
        idx_0 = ji_allk(0)
        tii = 2 * tau[idx_0]  # for all k tracking lines
        varepsp[i,i] += np.dot(wgts, K_at_0 - K(tii)) / st[i]

        # contributions from the rings that are around
        tij = np.zeros(ks[i])
        for j in range(i+1,I):
            idx_j = ji_allk(j-i)
            a = tii + tij
            diffK = K(a) - K(a + tau[idx_j])
            varepsp[i,j] += np.dot(wgts, diffK) / st[j]
            diffK = K(tij) - K(tij + tau[idx_j])
            varepsm[i,j] -= np.dot(wgts, diffK) / st[j]
            tij += tau[idx_j]

        # escape within (0, r_i-1/2)
        for j in range(i,0,-1):
            # the index of the ring proceeding backwards is j-1
            jm1 = j - 1
            idx = get_ks_in_ring(jm1)
            hk, wk = hks[idx], wks[idx]
            wgts = w(wk, r[i+1], hk)
            # contribution from the same ring, whose index is still i
            idx_i = ji_allk(i - jm1, jm1)
            diffK = K_at_0 - K(tau[idx_i])
            tii = np.zeros_like(hk)
            for n in range(j):
                tii += tau[ji_allk(n, jm1)]
            a = tau[idx_i] + 2 * tii
            diffK += K(a) - K(a + tau[idx_i])
            varepsp[i,i] += np.dot(wgts, diffK) / st[i]

            # contribution from the outer rings (new index n)
            a += tau[idx_i]
            tij = np.zeros_like(hk)
            for n in range(i+1, I):
                a += tij
                idx_n = ji_allk(n, jm1)
                diffK = K(a) - K(a + tau[idx_n])
                varepsp[i,n] += np.dot(wgts, diffK) / st[n]
                diffK = K(tij) - K(tij + tau[idx_n])
                varepsm[i,n] -= np.dot(wgts, diffK) / st[n]
                tij += tau[idx_n]

            # contribution from the inner rings (again with the new index n)
            tij = np.zeros_like(hk)
            for n in range(i-1,-1,-1):
                # upper quadrant (closest to the outer surface i+1/2)
                idx_n = ji_allk(n, jm1)
                a = tau[idx_i] + tij
                diffK = K(a) - K(a + tau[idx_n])

                # lower quadrant (farer to the outer surface i+1/2)
                tjj = np.zeros_like(hk)
                for m in range(n):
                    tjj += tau[ji_allk(m, jm1)]
                a += tau[idx_n] + 2 * tjj
                diffK += K(a) - K(a + tau[idx_n])
                varepsp[i,n] += np.dot(wgts, diffK) / st[n]

                tij += tau[idx_n]

    rinv = 1. / r[1:]
    Rinv = np.tile(rinv, (I, 1)).T
    varepsp *= 2 * Rinv
    varepsm *= 2 * Rinv
    if geometry_type == "sphere":
        varepsp *= np.pi * Rinv
        varepsm *= np.pi * Rinv
    return vareps


def ep2cp(ep):
    "Derive first flight collision probabilities by input escape ones."
    # ep are organized as (G, I, I, 2) where the last index is used for
    # positive and negative currents in order. Remind that the reduced
    # escape terms (wrongly called probability sometimes) for the incoming
    # (negative-minus) currents are already stored as negative quantities.
    cp = np.zeros((G, I, I),)
    if np.any(ep > 1):
        lg.warn("Detected escape probabilities greater than 1!")
    ep_minus, ep_plus = ep[:,:,:,1], ep[:,:,:,0]
    cp[:, 0,:] = ep_plus[:,0,:] + ep_minus[:,0,:]
    cp[:,1:,:] = ep_plus[:,:-1,:] - ep_plus[:,1:,:] \
               + ep_minus[:,:-1,:] - ep_minus[:,1:,:]
    return cp


if __name__ == "__main__":

    logfile = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    lg.basicConfig(level=lg.INFO)  # filename = logfile
    lg.info("* verbose output only in DEBUG logging mode *")

    # nb of rings in the cylinder
    L = .5  # cm, body thickness
    I = 4  # number of cells in the spatial mesh
    r = np.linspace(0, L, I+1)
    # r = np.array([0., .075, .15]); I = r.size - 1;  # test
    V = calculate_volumes(r, geometry_type)
    ks = np.full(I, 1)  # quadrature order for each spatial cell

    G = 1  # nb of energy groups
    c = 0.5  # number of secondaries by scattering
    st = np.ones(G)
    st_r = np.tile(st, (I, 1)).T
    ss_r = c * st_r
    # we use st / 8 for nsf
    nsf_r = 0.125 * st_r

    tr_data = calculate_tracking_data(r, ks)

    # WARNING: storage of ndotOp should not be necessary

    # INFO: in curvilinear coordinates we assume a vanishing current at the
    # center because of the central symmetry. There is no need to store this.
    J = np.zeros((G, I),)

    # reduced escape probabilities for partial currents (2 for plus and minus)
    vareps = np.zeros((G, I, I, 2),)
    for g in range(G):
        vareps[g,:,:,:] = calculate_escape_prob(r, st_r[g,:], tr_data,
                                                geometry_type)

    # derive collision probabilities from escape probabilities
    # first multiply by the volumes of starting cells to get the probabilities
    ep = vareps / np.moveaxis(np.tile(V, (G, 2, I, 1)), 1, -1)
    cp = ep2cp(ep)
