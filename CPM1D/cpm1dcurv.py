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
# Owned
__title__ = "Calculation of CP in 1D curvilinear geometries"
__author__ = "D. Tomatis"
__date__ = "30/08/2019"
__version__ = "1.0.0"

import os
import sys
import logging as lg
import numpy as np
from scipy.special import expn as En
from scipy.special import roots_jacobi
from KinPy.algo609 import dbskin as Ki

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'FD1dMGdiff')
)
from FDsDiff1D import input_data, unfold_xs, compute_cell_volumes, \
                      compute_cell_surfaces


# useful constants
Ki1_at_0, Ki2_at_0, Ki3_at_0 = Ki(0.,1,1,1)[0], Ki(0.,2,1,1)[0], Ki(0.,3,1,1)[0]
geometry_type = "cylinder"  # cylinder or sphere only
max_float = np.finfo(float).max  # float is float64!
min_float = np.finfo(float).min
np.set_printoptions(precision=10)


class solver_options:
    """Object collecting (input) solver options."""
    toll = 1.e-6  # default tolerance
    nbitsmax = 100  # default nb. of max iterations (its)

    def __init__(self, iitmax=nbitsmax, oitmax=nbitsmax,
                 otoll=toll, itoll=toll, ks=None, GQ="Gauss-Jacobi"):
        self.oitmax = oitmax  # max nb of outer iterations
        self.iitmax = iitmax  # max nb of inner iterations
        self.otoll = otoll  # tolerance on fiss. rates at outer its.
        self.itoll = itoll  # tolerance on flx at inner its.
        self.ks = ks  # nb. of quadrature point per cell
        self.GaussQuadrature = GQ  # type of Guass quadrature along
                                   # the h axis in curvilinear coords.
        self.betaL = 0  # partial current albedo left
                        # (not used in curv. geoms)
        self.betaR = 0  # must be <= 1, see RBC in input_data
        self.check_input()

    def check_input(self):
        if self.ks is None:
            raise InputError('Missing input nb. of quadrature points')
        if self.oitmax < 0:
            raise InputError('Detected negative max nb. of outer its.')  
        if self.iitmax < 0:
            raise InputError('Detected negative max nb. of inner its.')
        if self.GaussQuadrature != "Gauss-Jacobi" and \
           self.GaussQuadrature != "Gauss-Legendre":
            raise InputError('Unsupported type of Gauss quadrature')           

    @property
    def itsmax(self):
        "pack nb. of outer and inner iterations"
        return self.oitmax, self.iitmax

    @property
    def tolls(self):
        "pack tolerances on residual errors in outer and inner iterations"
        return self.otoll, self.itoll


def equivolume_mesh(I, a=0, b=1, geometry_type="cylinder"):
    "Make an equivolume mesh of I elements within [a, b]."
    if I <= 0:
        raise InputError("Invalid nb. of mesh elements")
    if a >= b:
        raise InputError("Invalid input bounds.")
    Vi = (b - a) / float(I)
    if "cylind" in geometry_type:
        Vi *= b + a
    elif "spher" in geometry_type:
        Vi *= b**2 + a * b + a**2
    elif geometry_type != "slab":
        raise InputError("Unsupported geometry type")
    
    r = np.cumsum(Vi * np.ones(I))
    if "cylind" in geometry_type:
        r = np.sqrt(r - a**2)
    elif "spher" in geometry_type:
        r = np.cbrt(r - a**3)
    
    return np.insert(r, 0, a)


def calculate_volumes(r, geometry_type="cylinder"):
    "Compute the cell volumes with the input 1d mesh r."
    return compute_cell_volumes(r, geometry_type, per_unit_angle=False)


def calculate_surfaces(r, geometry_type="cylinder"):
    "Compute the cell outer surfaces with the input 1d mesh r."
    return compute_cell_surfaces(r, geometry_type, per_unit_angle=False)


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

    bmao2 = .5 * (b - a)
    if quad == 'Gauss-Legendre':
        # Gauss-Legendre (default interval is [-1, 1])
        x, w = np.polynomial.legendre.leggauss(k)
    elif quad == 'Gauss-Jacobi':
        # Gauss-Jacobi, overcoming the singularity of Ki3 at 0 (tangent)
        # warning: weights and points differ from Hebert's book!
        # the first parameters used by Hebert is unknown
        x, w = roots_jacobi(k, 1, 0)
    elif quad == 'Rectangular':
        x, w = np.linspace(-1., 1., k + 1), np.full(k, 2. / k)
        x = (x[1:] + x[:-1]) / 2.
    else:
        raise ValueError("Unknown type of numerical quadrature.")

    # Translate x values from the interval [-1, 1] to [a, b]
    ## return (x + 1) * bmao2 + a, w * bmao2
    c = 1 - x
    if quad == 'Gauss-Jacobi':
        c *= .5 * (1 - x)
    return b - c * bmao2, w * bmao2


def calculate_tracking_data(r, ks, sphere=False, quadrule='Gauss-Jacobi'):
    """Calculate the quadrature weights for each ring described by r[i-1] and
    r[i]. The array r contains I+1 radii defining I rings in the cylinder.
    ks contains the k-th order of the quadrature in each ring. So ks has
    I values. The tracks are stored in a 1D array starting from the those
    which are closer to the center."""
    I = len(r) - 1
    lg.info("Nb of rings in the %s, I = %d" %
        ('sphere' if sphere else 'cylinder', I))
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
                                              quad=quadrule)

        for k in range(ks[i]):
            lt = 0.
            for j in np.arange(i, I) + 1:
                tracks[pos] = y(r[j], rks[rng][k])
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


def calculate_escape_prob(r, st, tr_data, geometry_type="cylinder",
                          wcos=True):
    """Calculate the reduced escape probabilities in the given geometry type.
    These quantities are needed to calculate the partial currents in the 1D
    curvilinear frames. The flag wcos allows to get escape probabilities
    weighted with the director cosines."""
    lg.debug("Calculate the reduced escape probabilities.")
    if wcos == True:
        raise RuntimeError("Attention, the cosine weight seems mistaken!")
    I = r.size - 1  # nb of rings
    if not np.all(np.diff(r) > 0):
        raise ValueError("Input values in r are not in increasing order")
    if st.size != I:
        raise InputError("size mismatch of input sigma_t")

    # unpack tracking data
    ks, tracks = tr_data["tracks_in_rings"], tr_data["tracks"]
    hks, wks = tr_data["quadrature_data"]
    jki, ki = tr_data["address_funcs"]

    # compute the optical lengths by multiplying the tracks by the
    # corresponding total cross sections (remind that the tau data relate
    # still to a rectangular sector, or quadrant, of the circle).
    tau = np.array(tracks, copy=True)
    for i in range(I):
        for k in range(ks[i]):
            for j in range(I-i):
                tau[jki(j,k,i)] *= st[i]

    # select the kernel function according to the geometry type
    if "cylind" in geometry_type:
        # the following function definition should be improved in future
        # versions to achieve higher performances
        def K(ell, n=3):
            if not hasattr(ell, '__len__'):
                Ki3_KODE2_M1 = Ki(ell, n, 2, 1)
                if Ki3_KODE2_M1[1] != 0:
                    raise RuntimeError("error in DBSKIN")
                else:
                    Ki3_at_ell = Ki3_KODE2_M1[0]
            else:
                Ki3_KODE2_M1 = [Ki(l, n, 2, 1) for l in ell]
                if any([v[1] != 0 for v in Ki3_KODE2_M1]):
                    raise RuntimeError("error in DBSKIN")
                else:
                    # the Ki part should be improved for higher efficiency
                    Ki3_at_ell = np.asarray([v[0][0] for v in Ki3_KODE2_M1])
            return Ki3_at_ell / np.exp(ell)
        K_at_0 = Ki3_at_0
        w = lambda wk, r, hk: wk * y(r, hk) if wcos else wk
    elif "spher" in geometry_type:
        K = lambda ell: np.exp(-ell)
        K_at_0 = 1.
        w = lambda wk, r, hk: wk * hk * y(r, hk) if wcos else wk * hk
    else:
        raise ValueError("Unsupported geometry type " + geometry_type)

    # calculate the recuced escape probabilities varepsilon (times the volume)
    # remind that the current vanishes at the center, and likewise the partial
    # currents (assumption, but it shall be enough to say that they're equal!).
    vareps = np.zeros((I, I, 2),)
    # reference the partial currents to the main data container
    varepsp = vareps[:,:,0]  # first index is i, second index is j (sources)
    varepsm = vareps[:,:,1]  # (defined as negative!)
    # varepsp is for escape whereas varepsm is for in-scape

    # local functions
    cumsum_ks = np.insert(np.cumsum(ks), 0, 0)
    get_ks_in_ring = lambda i0: np.arange(cumsum_ks[i0], cumsum_ks[i0+1])
    ji_allk = lambda j, i: np.asarray([jki(j,k,i) for k in range(ks[i])])
    def get_weights(i, r):
        idx = get_ks_in_ring(i)
        hk, wk = hks[idx], wks[idx]
        # print('idx',idx)
        # print('hk', hk)
        # print('wk', wk)
        return w(wk, r, hk)

    # process the surfaces outwards
    # print('r', r)
    # print('V', calculate_volumes(r, geometry_type))
    for i in range(I):
        wgts = get_weights(i, r[i+1])

        # escape within (r_i-1/2, r_i+1/2)
        # contribution from the i-th ring through its outer surface
        idx_0, invsti = ji_allk(0, i), 1. / st[i]
        tii = 2 * tau[idx_0]  # for all k tracking lines
        varepsp[i,i] += np.dot(wgts, K_at_0 - K(tii)) * invsti
        # varepsp[i,i] += np.dot(wgts, 2 * tracks[idx_0])  # * K(0, n=2)=1 !
        # print('ell', tii * invsti, 'i', i)
        # print('tracks', tracks[idx_0])
        # print(np.dot(wgts, 4 * tracks[idx_0] * np.pi), varepsp[i,i])
        # print('wgts', wgts)

        # contributions from the rings that are around
        tij = np.zeros_like(wgts)
        for j in range(i+1,I):
            idx_j, invstj = ji_allk(j-i, i), 1. / st[j]
            a = tii + tij
            diffK = K(a) - K(a + tau[idx_j])
            varepsp[i,j] += np.dot(wgts, diffK) * invstj
            diffK = K(tij) - K(tij + tau[idx_j])
            varepsm[i,j] -= np.dot(wgts, diffK) * invstj
            tij += tau[idx_j]

        # escape within (0, r_i-1/2)
        for j in range(i-1,-1,-1):
            # the index decreases inwards
            # escape within (r_j-1/2, r_j+1/2)
            wgts = get_weights(j, r[i+1])
            # w.r.t. the jik notation, the same ring is at the ji-th position
            # starting from radial zero level
            ji = i - j
            # contribution from the same ring, whose index is still i
            idx_i = ji_allk(ji, j)
            diffK = K_at_0 - K(tau[idx_i])
            # varepsp[i,i] += np.dot(wgts, diffK) * invsti  # ??
            tii = np.zeros_like(wgts)
            for n in range(ji):
                tii += tau[ji_allk(n, j)]
            a = tau[idx_i] + 2 * tii
            diffK += K(a) - K(a + tau[idx_i])
            varepsp[i,i] += np.dot(wgts, diffK) * invsti

            # contribution from the outer rings (new index n)
            a += tau[idx_i]
            tij = np.zeros_like(wgts)
            for n in range(i+1, I):
                idx_n, invstn = ji_allk(n - j, j), 1. / st[n]
                a += tij
                diffK = K(a) - K(a + tau[idx_n])
                varepsp[i,n] += np.dot(wgts, diffK) * invstn
                diffK = K(tij) - K(tij + tau[idx_n])
                varepsm[i,n] -= np.dot(wgts, diffK) * invstn
                tij += tau[idx_n]

            # contribution from the inner rings (again with the new index n)
            tij = np.zeros_like(wgts)
            for n in range(i-1,j-1,-1):
                # upper quadrant (closest to the outer surface i+1/2)
                idx_n, invstn = ji_allk(n - j, j), 1. / st[n]
                a = tau[idx_i] + tij
                diffK = K(a) - K(a + tau[idx_n])
                # varepsp[i,n] += np.dot(wgts, diffK) * invstn  # ??

                # lower quadrant (farer to the outer surface i+1/2)
                tjj = np.zeros_like(wgts)
                for m in range(n - j):
                    tjj += tau[ji_allk(m, j)]
                a += tau[idx_n] + 2 * tjj
                diffK += K(a) - K(a + tau[idx_n])
                varepsp[i,n] += np.dot(wgts, diffK) * invstn

                tij += tau[idx_n]

    Rinv = np.tile(1. / r[1:], (I, 1)).T
    if "spher" in geometry_type:
        varepsp *= np.pi  # * Rinv
        varepsm *= np.pi  # * Rinv
    if wcos:
        varepsp *= Rinv
        varepsm *= Rinv
    return vareps * 2


def ep2cp(ep, check=False, Vst=None):
    "Derive first flight collision probabilities by input escape ones."
    # ep are organized as (G, I, I, 2) where the last index is used for
    # positive and negative currents in order. Remind that the reduced
    # escape terms (wrongly called probability sometimes) for the incoming
    # (negative-minus) currents are already stored as negative quantities.
    G, I = ep.shape[:2]
    cp = np.zeros((G, I, I),)
    for g in range(G):  # check for negative values
        if np.any(abs(ep[g,:,:]) > 1):
            msg = ("Detected escape probabilities greater than 1 " + 
                   "in group %d!\n" % (g + 1))
            msg += "ep_minus =\n" + str(ep[g,:,:,1]) + '\n'
            msg += " ep_plus =\n" + str(ep[g,:,:,0])
            lg.warning(msg)
    ep_minus, ep_plus = ep[:,:,:,1], ep[:,:,:,0]
    cp[:, 0,:] = - ep_plus[:,0,:] - ep_minus[:,0,:]
    cp[:,1:,:] = ep_plus[:,:-1,:] - ep_plus[:,1:,:] \
               + ep_minus[:,:-1,:] - ep_minus[:,1:,:]
    for g in range(G):
        cp[g][np.diag_indices(I)] += 1
        if np.any(cp[g] < 0):
            msg = ("Detected negative collision probabilities " +
                   "in group %d:\n" % (g + 1) + str(cp[g]))
            # raise RuntimeError(msg)
            lg.warning(msg)
    
    if check:
        if Vst is None:
            raise InputError("V times st is needed to check reciprocity")
        for g in range(G):
            pij = cp[g,:,:]
            # check particle conservation on the full domain
            np.testing.assert_allclose(np.ones(I),
                np.sum(pij, axis=0) + ep_plus[g,-1,:], rtol=1.e-06,
                err_msg="conservation on the full domain not satisfied" +
                    " for group %d" % (g + 1))
    
            # check reciprocity
            Vjstj = np.tile(Vst[g, :], (I, 1))
            symm = pij * Vjstj
            # print('pij * Vjstj = \n' + str(symm) + '\n ---')
            np.testing.assert_allclose(symm, symm.T, rtol=1.e-06,
                err_msg="reciprocity of CPs not satisfied" +
                    " for group %d" % (g + 1))
    return cp


def calculate_probs(r, st, tr_data=None, Vj=None, geometry_type="cylinder"):
    """Calculate first flight escape and collision probabilities in the
    input geometry_type."""
    G, I = st.shape
    if Vj is None:
        Vj = calculate_volumes(r, geometry_type)
    # reduced escape probabilities for partial currents (not weighted by
    # director cosines)
    varepS = np.zeros((G, I, I, 2),)
    for g in range(G):
        varepS[g,:,:,:] = calculate_escape_prob(r, st[g,:], tr_data,
                                                geometry_type, wcos=False)

    # divide by the volumes of starting cells to get the probabilities
    eP = varepS / np.moveaxis(np.tile(Vj, (G, 2, I, 1)), 1, -1)
    # derive collision probabilities from escape probabilities and
    # return the f.f. collision probabilities
    return eP, ep2cp(eP, check=True, Vst=st * Vj)


def check_xs(xs):
    st, ss, chi, nsf = xs
    G, I = nsf.shape
    if st.shape != nsf.shape:
        raise InputError("st and nsf shapes mismatch")
    if chi.shape != nsf.shape:
        raise InputError("nsf and chi shapes mismatch")
    if ss.shape[:2] != (G, G):
        raise InputError("expected ss with G=%d" % G)
    if ss.shape[-1] != I:
        raise InputError("expected ss with I=%d" % I)
    pass


def calculate_full_spectrum(xs, cp, ep=None, beta=0, data=None):
    """Direct solution of the k-eigenvalue problem in integral transport
    by the collision probability method. Input data are the xs list and
    the collision probabilities in cp. Only isotropic scattering is
    allowed. A relation of albedo for the partial currents can be used
    at the boundary."""
    st, ss, chi, nsf = xs
    G, I = nsf.shape
    check_xs(xs)
    if (beta < 0) or (beta > 1):
        raise InputError("beta is not in valid range")
    else:
        if ep is None:
            raise InputError("beta > 0, but no input escape probs")
        if data is None:
            raise InputError("no input VjoSb, input mesh data is needed")
        else:
            r, geo, V, Sb = data.xi, data.geometry_type, data.Vi, data.Si[-1]
            # V = calculate_volumes(r, geo)
            # Sb = calculate_surfaces(r, geo)[-1]
            VjoSb = V / Sb

    GI = G * I
    PS = np.zeros((GI, GI),)
    PX, F = np.zeros((GI, I),), np.zeros((I, GI),)
    # X = np.zeros_like(PX)
    Is = np.arange(I)
    for g in range(G):
        idx = slice(I * g, I * (g + 1))
        pji = np.transpose(cp[g,:,:] / st[g,:])  # reduced CP
        if beta > 0:
            ebj = ep[g,-1,:,0]
            pjb = VjoSb * ebj * 4.  # times is very important!
            transmission = 1 - np.dot(pjb, st[g,:])
            coef = beta / (1 - beta * transmission)
            pji += coef * np.dot(np.diag(ebj), np.tile(pjb, (I, 1)))
            
        # X[Is + g * I, Is] = chi[g,:]
        F[Is, Is + g * I] = nsf[g,:]
        PX[idx,:] = pji * chi[g,:]
        for gg in range(G):
            jdx = slice(I * gg, I * (gg + 1))
            PS[idx, jdx] = pji * ss[g,gg,0,:]
    PS *= -1
    PS[np.diag_indices_from(PS)] += 1
    H = np.dot(F, np.dot(np.linalg.inv(PS), PX))
    return np.linalg.eig(H)  # (K, flx_modes)


def solve_outers(xs, cp, slvr_opts, ep=None, flx=None, k=1, kappa=1.5,
                 data=None):
    """Calculate the fundamental eigenpair of the CPM integral transport
    problem by the power method. Convergence is accelerated by the 
    Wielandt acceleration. A boundary condition of albedo on the partial
    currents can be selected by 0 leq beta leq 1."""
    st, ss, chi, nsf = xs
    G, I = nsf.shape
    GI = G * I
    
    check_xs(xs)
    if flx is None:
        flx = np.ones(GI)
    beta = slvr_opts.betaR
    if (beta < 0) or (beta > 1):
        raise InputError("beta is not in valid range")
    else:
        if ep is None:
            raise InputError("beta > 0, but no input escape probs")
        if data is None:
            raise InputError("no input VjoSb, input mesh data is needed")
        else:
            r, geo, V, Sb = data.xi, data.geometry_type, data.Vi, data.Si[-1]
            # V = calculate_volumes(r, geo)
            # Sb = calculate_surfaces(r, geo)[-1]
            VjoSb = V / Sb
    
    PS, PX = np.zeros((GI, GI),), np.zeros((GI, I),)
    # X, F = np.zeros_like(PX), np.zeros((I, GI),)
    PXF = np.zeros_like(PS)
    Is = np.arange(I)
    for g in range(G):
        idx = slice(I * g, I * (g + 1))
        pji = np.transpose(cp[g,:,:] / st[g,:])  # reduced CP
        if beta > 0:
            ebj = ep[g,-1,:,0]
            pjb = VjoSb * ebj * 4.  # times is very important!
            transmission = 1 - np.dot(pjb, st[g,:])
            coef = beta / (1 - beta * transmission)
            pji += coef * np.dot(np.diag(ebj), np.tile(pjb, (I, 1)))
        # X[Is + g * I, Is] = chi[g,:]
        # F[Is, Is + g * I] = nsf[g,:]
        PX[idx,:] = pji * chi[g,:]
        for gg in range(G):
            jdx = slice(I * gg, I * (gg + 1))
            PS[idx, jdx] = pji * ss[g,gg,0,:]
            # PS[idx, jdx] = pji * \
                 # (ss[g,gg,0,:] - chi[g,:] * nsf[gg,:] / kappa)
            PXF[idx, jdx] = PX[idx,:] * nsf[gg,:]
    PS *= -1
    PS[np.diag_indices_from(PS)] += 1
    H = np.dot(np.linalg.inv(PS), PXF)
    shiftedH = lambda k, dk: np.dot(np.linalg.inv(PS - PXF / (k + dk)),
        dk / k / (k + dk) * PXF)
    
    # function to calculate the Rayleigh quotient
    Rq = lambda M, x: np.dot(x, np.dot(M, x)) / np.dot(x, x)
    # function to calculate the source from fission and emission x
    fsrc = lambda flx, x=chi, n=nsf: (x * 
        np.sum(n * flx.reshape(G, I), axis=0)).flatten()

    # start outer iterations
    err, ito, ferr = 1.e+20, 0, np.zeros_like(flx)
    # flx /= k  # in case we do not calculate k throughout the its
    src = fsrc(flx)
    while (err > slvr_opts.otoll) and (ito < slvr_opts.oitmax):
        flx_old = np.array(flx, copy=True)
        src_old, k_old = np.array(src, copy=True), k
        flx = np.dot(H, flx) / k
        dk = kappa - k if ito < 5 else (0.05 / (ito // 5)) 
        flx = np.dot(shiftedH(k, dk), flx)
        src = fsrc(flx)
        # flx /= np.linalg.norm(flx)
        # without acceleration
        pratio = np.dot(flx, src) / np.dot(flx, src_old)
        # k *= pratio
        # with acceleration
        dkratio = dk / k_old
        k *= pratio * (1 + dkratio) / (pratio + dkratio)
        
        # ferr = np.where(flx > 0, 1 - flx_old / flx, flx - flx_old)
        # or the following
        mask = flx > 0
        ferr[mask] = 1. - flx_old[mask] / flx[mask]
        ferr[~mask] = flx[~mask] - flx_old[~mask]
        err = abs(ferr).max()
        ito += 1
        lg.info("<- it={:^4d}, k={:<13.6g}, err={:<+13.6e}".format(
            ito, k, err
        ))
    
    return Rq(H, flx), flx.reshape(G, I)


def solve_cpm1D(idata, slvr_opts):
    '''Run the CPM solver.'''
    lg.info("Prepare input data")
    r, geo = idata.xi, idata.geometry_type
    # xs = [st, ss, chi, nsf]
    xs = unfold_xs(idata, diff_calc=False)
    if idata.RBC == 2:
        slvr_opts.betaR = 1
    if (idata.LBC != 2) and (geo == "sphere" or geo == "cylinder" ):
        raise InputError("Reflection at L(eft)BC is requested in " + geo)
    lg.info("-o"*22)
    
    lg.info("Calculate the collision probabilities")
    tr_data = calculate_tracking_data(r, slvr_opts.ks,
        sphere=True if "spher" in geo else False, 
        quadrule=slvr_opts.GaussQuadrature)
    V = calculate_volumes(r, geo)
    np.testing.assert_allclose(V, idata.Vi,
        err_msg="Cell volumes in input are not correct")
    ep, cp = calculate_probs(r, xs[0], tr_data, V, geo)
    lg.info("-o"*22)

    # direct solution for the full spectrum
    chk_sign = lambda x: -x if np.all(x < 0) else x
    lg.info("Call the direct solution by CPM to get the full spectrum.")
    K, nfss = calculate_full_spectrum(xs, cp, ep, slvr_opts.betaR, data=idata)
    # ...nfss is nu-fiss rate
    ik = np.argmax(K)
    keff, nfss_rates = K[ik], chk_sign(nfss[:,ik])
    lg.info("Multiplication factor from the fission matrix is %.6f" % keff)
    lg.info("Fission rate distribution:\n" + str(nfss_rates))
    
    # compute the fundamental eigenpair by the power method
    lg.info("Solve iteratively by the Wielandt-accelerated power\n" +
            " method the CPM transport equation.")
    keff, flx = solve_outers(xs, cp, slvr_opts, ep, data=idata)
    lg.info("Multiplication factor from the power method is %.6f" % keff)
    lg.info("Flux distribution (G/I):\n" + str(flx))
    lg.info("-o"*22)
    lg.info("*** NORMAL END OF CALCULATION ***")
    return keff, chk_sign(flx)


if __name__ == "__main__":

    logfile = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    lg.basicConfig(level=lg.INFO)  # filename = logfile
    lg.info("* verbose output only in DEBUG logging mode *")
    geometry_type = 'sphere'

    if 'spher' in geometry_type:
        L = np.cbrt(3. / 4. / np.pi)
    elif 'cylind' in geometry_type:
        L = 1. / np.sqrt(np.pi)  # L in cm, body thickness
    I = 2  # number of cells (rings) in the spatial mesh
    # r = np.linspace(0, L, I+1)
    r = equivolume_mesh(I, 0, L, geometry_type)
    # r = np.array([0., .075, .15]); I = r.size - 1;  # test
    V = calculate_volumes(r, geometry_type)
    ks = np.full(I, 2)  # quadrature order for each spatial cell

    G = 1  # nb of energy groups
    c = 0.5  # number of secondaries by scattering
    st = np.ones(G) * 1.e-6
    st_r = np.tile(st, (I, 1)).T
    ss_r = c * st_r
    # we use st / 8 for nsf
    nsf_r = 0.125 * st_r

    tr_data = calculate_tracking_data(r, ks, quadrule="Gauss-Jacobi",
        sphere=True if "spher" in geometry_type else False)

    # WARNING: storage of ndotOp should not be necessary

    # INFO: in curvilinear coordinates we assume a vanishing current at the
    # center because of the central symmetry. There is no need to store this.
    J = np.zeros((G, I),)

    # # reduced escape probabilities for partial currents (2 for plus and minus)
    # vareps = np.zeros((G, I, I, 2),)
    # for g in range(G):
        # vareps[g,:,:,:] = calculate_escape_prob(r, st_r[g,:], tr_data,
                                                # geometry_type)

    # # derive collision probabilities from escape probabilities
    # # first multiply by the volumes of starting cells to get the probabilities
    # ep = vareps / np.moveaxis(np.tile(V, (G, 2, I, 1)), 1, -1)
    # cp = ep2cp(ep, check=True, Vst=st_r * V)
    # print('almost-CP: \n', cp[0, :, :])

    ep, cP = calculate_probs(r, st_r, tr_data, V, geometry_type)
    print('V',V)
    print('CP:\n' + str(cP[0, :, :]))
    
    # start solving the integral transport equation by the CPM
