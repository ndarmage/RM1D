#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This module resolves diffusion in 1D geometry by finites differences
and with the multigroup energy formalism. Boundary conditions use a
fictitious extrapolation length in the generalized form of the kind:

\[ J = -D \phi_{bnd} / (\Delta_{bnd} + \zeta). \]

$\zeta$ is the extrapolated length equal to $2.13 D$ in case of vacuum.
If it is zero, we have the zero flux b.c., while reflection is reproduced
by $\zeta \rightarrow \infty$. The b.c. code is in order 0, 1 and 2,
respectively.

Input data are set in two main objects. The first stores geometry and
material input data, whereas the second contains the solver options. The
Ronen Method is implemented on top of the diffusion solver through the
standard CMFD or its pCMFD version.

The Cartesian (slab) is the default geometry option. Curvilinears can be
chosen by selecting 'cylindrical' or 'spherical' for the global variable
geometry_type.

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

.. note:: Ronen iterations are accelerated by Anderson's method as shown by
          Walker and Ni [Walker2011]_ and [Henderson2019]_. 

.. [Walker2011] Walker, H. F., & Ni, P. (2011). Anderson acceleration for
                fixed-point iterations. SIAM Journal on Numerical Analysis,
                49(4), 1715-1735.
                
.. [Henderson2019] Henderson, N. C., & Varadhan, R. (2019). Damped Anderson
                   acceleration with restarts and monotonicity control for 
                   accelerating EM and EM-like algorithms. Journal of
                   Computational and Graphical Statistics, 1-13.
"""
# Owned
__title__ = "Multigroup diffusion and RM in 1D geometries by finite differences"
__author__ = "D. Tomatis"
__date__ = "15/11/2019"
__version__ = "1.4.0"

import os
import sys
import logging as lg
import numpy as np
from scipy.special import expn as En
from scipy.optimize import brentq
from GeoMatTools import *

sys.path.append(
    os.path.join(os.getcwd(), os.path.dirname(__file__), '..', 'CPM1D')
)
from cpm1dcurv import calculate_tracking_data, calculate_sprobs, \
                      calculate_eprobs, calculate_tprobs

max_float = np.finfo(float).max  # float is float64!
min_float = np.finfo(float).min
np.set_printoptions(precision=5)

# log file settings
logfile = os.path.splitext(os.path.basename(__file__))[0] + '.log'
# verbose output only with lg.DEBUG mode
lg.basicConfig(level=lg.INFO)  # filename = logfile

fix_reflection_by_flx2 = True


opt_theta_coeffs = np.array( \
    [-2.284879e-7, +1.222516e-5, -2.683648e-4, \
     +3.145553e-3, -2.152599e-2, +8.740501e-2, -5.542780e-2])


def opt_theta(tau):
    """Polynomial fit of optimal theta in odCMFD by Zhu et al. ANE 95 (2016)
    116-124."""
    if tau >= 1:
        t, poly_degree = opt_theta_coeffs[0], opt_theta_coeffs.size - 1
        for i in range(poly_degree):
            t *= tau
            t += opt_theta_coeffs[i + 1]
    elif tau >= 14:
        t = 0.127
    else:
        t = 0.
    return t


def roll_matrix(M, c):
    return np.concatenate([M[:, 1:],
                           np.expand_dims(c, axis=1)], axis=1)


class solver_options:
    """Object collecting (input) solver options. INFO: set ritmax to 0 to
    skip Ronen iterations."""
    toll = 1.e-6  # default tolerance
    nbitsmax = 100  # default nb. of max iterations (its)

    def __init__(self, iitmax=nbitsmax, oitmax=nbitsmax, ritmax=10,
                 pCMFD=False, otoll=toll, itoll=toll, rtoll=toll,
                 CMFD=True, wSOR=None, Aitken=False, Anderson_depth=5,
                 Anderson_relaxation=1, noacc_rit=0, ks=0, GQ="Gauss-Jacobi"):
        self.ritmax = ritmax  # set to 1 to skip Ronen iterations
        self.oitmax = oitmax  # max nb of outer iterations
        self.iitmax = iitmax  # max nb of inner iterations
        self.otoll = otoll  # tolerance on fiss. rates at outer its.
        self.itoll = itoll  # tolerance on flx at inner its.
        self.rtoll = rtoll  # tolerance on flx at RM its.
        self.CMFD = CMFD  # use CMFD, new D by Fick's law is False
        self.pCMFD = pCMFD  # classic CMFD is False
        self.noacc_rit = noacc_rit
        self.wSOR = wSOR  # SOR relaxation parameter (opt 1.7 -- 2^-)
        self.Aitken = Aitken  # poor performance noticed
        self.Anderson_depth = Anderson_depth  # dim of subspace for residuals
                                              # 0 to disable, not None
        self.Anderson_relaxation = Anderson_relaxation  # set to 1 to disable
        # opts for escape and collision probabilities in curv coords.
        self.ks = ks  # nb. of quadrature point per cell
        self.GaussQuadrature = GQ  # type of Guass quadrature along
                                   # the h axis in curvilinear coords.
        self.check_input()

    def check_input(self):
        if self.CMFD:
            info = "Use CMFD in Ronen iterations"
            info += " with " if self.pCMFD else " without "
            lg.info(info + "pCMFD.")
        else:
            lg.info("Recalculate diff. coeff. by Fick's law in " +
                    "Ronen iterations.")        
        if self.ritmax < 0:
            raise ValueError('Detected negative max nb. of RM its.')  
        if self.oitmax < 0:
            raise ValueError('Detected negative max nb. of outer its.')  
        if self.iitmax < 0:
            raise ValueError('Detected negative max nb. of inner its.')
        if self.noacc_rit < 0:
            raise ValueError('Detected negative nb. of unaccelerated rits.')
        if isinstance(self.Anderson_depth, (int, float)):
            if self.Anderson_depth < 0:
                raise ValueError('Detected negative dim of Anderson subspace.')
        if not (0 < self.Anderson_relaxation <= 1):
            raise ValueError('relaxation for Anderson out of allowed bounds.')
        if np.any(self.ks < 0):
            raise ValueError('Detected negative nb. of quadrature points' +
                             ' (set 0 with slab)')
        if self.GaussQuadrature != "Gauss-Jacobi" and \
           self.GaussQuadrature != "Gauss-Legendre":
            raise ValueError('Unsupported type of Gauss quadrature')

    @property
    def itsmax(self):
        "pack nb. of outer and inner iterations"
        return self.oitmax, self.iitmax

    @property
    def tolls(self):
        "pack tolerances on residual errors in outer and inner iterations"
        return self.otoll, self.itoll
    
    @property
    def Anderson(self):
        if isinstance(self.Anderson_depth, str):
            r = 'auto' in self.Anderson_depth
        else:
            r = self.Anderson_depth > 0
        return r

    @property
    def SOR(self):
        return True if self.wSOR is not None else False


class AndersonAcceleration:
    "Object storing data for Anderson Acceleration (AA)."
    
    # DAAREM parameters
    alpha = 1.2
    kappa = 30  # "half-life" of relative dumping (25 by Henderson2019)
    sk = 0  # used in the relative damping parameter deltak
    
    @property
    def Anderson_depth(self):
        return self.m
    
    @classmethod
    def delta(cls, sk):
        e = max(0, cls.kappa - sk)
        return 1 / (1 + cls.alpha**e)
    
    def __init__(self, opts=None, m=-1, betak=-1., size=0):
        if isinstance(opts, solver_options):
            self.m, self.betak = opts.Anderson_depth, opts.Anderson_relaxation
        else:
            self.m, self.betak = m, betak
        if self.m == 'auto':
            self.set_automatic_depth(size)
        self.check_input()
        self.Fk = np.zeros((size, self.m),)
        self.Xk = np.zeros_like(self.Fk)
    
    def check_input(self):
        if self.m < 0:
            raise ValueError('Detected negative dim of Anderson subspace.')
        if not (0 < self.betak <= 1):
            raise ValueError('relaxation for Anderson out of allowed bounds.')
        if self.alpha <= 1:
            raise ValueError('DAAREM alpha parameter must be > 1')
        if self.kappa < 0:
            raise ValueError('DAAREM kappa parameter must be >= 0')

    def set_automatic_depth(self, size, check=False):
        self.m = min(int(size / 2), 10)
        if check and hasattr(self, 'Fk'):
            if self.Fk.shape[0] != size:
                lg.warning("Fk and Xk must be redefined")
                self.Fk = np.zeros((size, self.m),)
                self.Xk = np.zeros_like(self.Fk)

    def __call__(self, k, fk, xk, xkp1, constrainedLS=False, k_restart=1):
        """Call AA with fk, xk, xkp1 = flxres, flxold, flx to be flattened by
        (*).flatten(); k is the iteration index decreased of noacc_rit. A
        restart can be enabled with k_restart > depth."""
        mk, orig_shape = min(np.mod(k - 1, k_restart) + 1, self.m), xkp1.shape
        fk, xk, xkp1 = map(np.ravel, [fk, xk, xkp1])
        Fk, Xk, betak = self.Fk, self.Xk, self.betak  # reference to obj attrs
        # ------------------------------------------------------------------
        if constrainedLS:  # version (a) - constrained L2 minimization
            if k > 0:
                Fr = Fk[:, -mk:] - np.tile(fk, (mk, 1)).T
                # alphm1 = np.dot(np.linalg.inv(np.dot(Fr.T, Fr)
                                # # + 0.05 * np.eye(mk)  # regularization
                                # ), np.dot(Fr.T, -fk))
                alphm1 = np.linalg.lstsq(Fr, -fk, rcond=None)[0]
                # from scipy.optimize import nnls, lsq_linear
                # gams = nnls(DFk, fk)[0]  # non-negative LS, or bounded as:
                # gams = lsq_linear(DFk, fk, bounds=(0, np.inf),
                                  # lsq_solver='exact', verbose=1).x
                alphmk = 1 - np.sum(alphm1)
                Gk = (Fk + Xk)[:, -mk:]
                xkp1 = betak * (alphmk * xkp1 + np.dot(Gk, alphm1)) \
                     + (1 - betak) * (alphmk * xk + np.dot(Xk[:, -mk:], alphm1))
                # print(mk, np.insert(alphm1, -1, alphmk))
            self.Fk, self.Xk = roll_matrix(Fk, fk), roll_matrix(Xk, xk)
        # ------------------------------------------------------------------
        else:  # version (b) - unconstrained L2 minimization
            Fkp1, Xkp1 = roll_matrix(Fk, fk), roll_matrix(Xk, xk)
            if k > 0:
                DFk = (Fkp1 - Fk)[:, -mk:]  # to start earlier for k < m
                DXk = (Xkp1 - Xk)[:, -mk:]
                # # Anderson Type I
                # # gams = np.dot(np.linalg.inv(np.dot(DXk.T, DFk)
                # #             # + 0.05 * np.eye(mk)  # regularization
                # #              ), np.dot(DXk.T, fk))
                # # Anderson Type II
                # gams = np.dot(np.linalg.inv(np.dot(DFk.T, DFk)
                               # # + 1e-13 * np.eye(mk)  # regularization
                              # ), np.dot(DFk.T, fk))
                # N.B.: regularization in previous schemes does not lead to
                # successful iterations
                # Implementation of DAAREM without merit function of interest
                Uk, dk, Vk = np.linalg.svd(DFk)
                # find lambdak
                uf = np.dot(fk, Uk[:, :mk])
                s = lambda lmbda: np.dot(np.dot(Vk.T,
                    np.diag(dk / (dk**2 + lmbda))), uf)
                self.sk = min(self.sk + 1, type(self).kappa - mk)
                deltak = self.delta(self.sk)
                vk = np.sqrt(deltak) * np.linalg.norm(s(0))
                phi = lambda lmbda: np.linalg.norm(s(lmbda)) - vk
                lmbdak = brentq(phi, 0, 1e+3)
                gams = s(lmbdak)
                # Walker2011 - no relaxation
                # xkp1 -= np.dot(DXk + DFk, gams).reshape(xkp1.shape)
                # Henderson2019 - betak is the relaxation parameter
                # (no significant improvement noticed)                
                xkp1 = betak * (xkp1 - np.dot(DXk + DFk, gams)) \
                     + (1 - betak) * (xk - np.dot(DXk, gams))
                # print('lmbdak:', lmbdak, '\ngammas:', gams)
                # input('wait')
            self.Xk, self.Fk = Xkp1, Fkp1
        # ------------------------------------------------------------------
        return xkp1.reshape(orig_shape)


def get_zeta(bc=0):
    "Return the zeta boundary condition according to the input bc code."
    if bc == 0:
        # vacuum
        zeta = 2.13
    elif bc == 1:
        # zero flux
        zeta = 0.
    elif bc == 2:
        # reflection (i.e. zero current)
        zeta = max_float
    else:
        raise ValueError("Unknown type of b.c.")
    return zeta


def solveTDMA(a, b, c, d):
    """Solve the tridiagonal matrix system equations by the
    Thomas algorithm. a, b and c are the lower, central and upper
    diagonal of the matrix to invert, while d is the source term."""
    n = len(d)
    cp, dp = np.ones(n - 1), np.ones(n)
    x = np.ones_like(dp)  # the solution

    cp[0] = c[0] / b[0]
    for i in range(1, n-1):
        cp[i] = c[i] / (b[i] - a[i-1] * cp[i-1])

    dp[0] = d[0] / b[0]
    dp[1:] = b[1:] - a * cp
    for i in range(1, n):
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / dp[i]

    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x


def compute_fission_source(chi, nsf, flx):
    # warning: the input flux can be volume-integrated or not
    return (chi * np.sum(nsf * flx, axis=0)).flatten()


def compute_scattering_source(ss0, flx):
    # warning: the input flux can be volume-integrated or not
    return np.sum(ss0 * flx, axis=1).flatten()


def compute_source(ss0, chi, nsf, flx, k=1.):
    "Return (flattened) scattering plus fission sources."
    qs = compute_scattering_source(ss0, flx)
    return (qs + compute_fission_source(chi, nsf, flx) / k)


def first_order_coeff_at_interfaces(f, Vi):
    """Compute the diffusion coefficient by 1st order finite-difference
    currents determined at both sides of an interface."""
    G, I = f.shape
    Im1 = I - 1
    
    fb = 2 * f[:, 1:] * f[:, :-1]
    return fb / (f[:, 1: ] * np.tile(Vi[:-1], G).reshape(G, Im1) +
                 f[:, :-1] * np.tile(Vi[1: ], G).reshape(G, Im1))


def set_diagonals(st, Db, data, dD=None):
    "Build the three main diagonals of the solving system equations."
    # Remind that we solve for the volume-integrated flux
    LBC, RBC = data.BC
    # cell bounds, widths and reduced volumes
    xi, Di, Vi, geo = data.xi, data.Di, data.Vi, data.geometry_type
    G, I = st.shape
    GI = G * I
    # a, b, c are lower, center and upper diagonals respectively
    a, b, c = np.zeros(GI-1), np.zeros(GI), np.zeros(GI-1)

    # take into account the delta-diffusion-coefficients
    if isinstance(dD, tuple) or isinstance(dD, list):
        lg.debug('Apply corrections according to pCMFD.')
        dDp, dDm = dD
    else:
        lg.debug('Apply corrections according to classic CMFD.')
        if dD is None:
            dD = np.zeros((G, I+1),)
        elif isinstance(dD, np.ndarray):
            if dD.shape != (G, I+1):
                raise ValueError('Invalid shape of input dD')
        else:
            raise ValueError('Invalid input dD (delta D coefficients).')
        dDp = dDm = dD

    # compute the coupling coefficients by finite differences
    iDm = 2. / (Di[1:] + Di[:-1])  # 1 / (\Delta_i + \Delta_{i+1}) / 2
    xb0, xba = 1., 1.
    if geo == 'cylindrical' or geo == 'cylinder':
        iDm *= xi[1:-1]
        dD *= xi
        xb0, xba = xi[0], xi[-1]
    if geo == 'spherical' or geo == 'sphere':
        iDm *= xi[1:-1]**2
        dD *= xi**2
        xb0, xba = xi[0]**2, xi[-1]**2

    # d = D_{i+1/2}/(\Delta_i + \Delta_{i+1})/2
    d = Db[:, 1:-1].flatten() * np.tile(iDm, G)

    # print('coeffs from diffusion')
    # print (d.reshape(G,I-1))
    # print(dDp, dDm)
    # get extrapolated length
    zetal, zetar = get_zeta(LBC), get_zeta(RBC)
    # c1l, c2l = quadratic_fit_zD(xi[0],  # option (c), see below
                                # np.insert(data.xim[:2], 0, xi[0]),
                                # zetal) if zetal < max_float else (0, 0)
    # c1r, c2r = quadratic_fit_zD(xi[-1],  # option (c), see below
                                # np.insert(data.xim[-2:][::-1], 0, xi[-1]),
                                # -zetar) if zetar < max_float else (0, 0)
    for g in range(G):
        # d contains only I-1 elements, flattened G times
        idd = np.arange(g*(I-1), (g+1)*(I-1))
        id0, ida = g*I, (g+1)*I-1
        idx = np.arange(id0, ida)  # use only the first I-1 indices

        coefp = d[idd] + dDm[g, 1:-1]  # \phi_{i+1}
        coefm = d[idd] - dDp[g, 1:-1]  # \phi_{i-1}
        a[idx] = -coefm / Vi[:-1]
        b[idx] = coefm
        b[idx+1] += coefp
        c[idx] = -coefp / Vi[1:]

        # add b.c. (remind to be consistent with compute_diff_currents)
        # different attempts to get higher order accuracy
        # option (a) use J_b = \pm flx_b / zeta, with flx_b = flx_0 (1st order) 
        # option (b) central fin. diff. with extrap distance, at 2nd order
        #  *** (case b must be debugged because it doesn't work)
        # option (c) fit by 2nd order polynomial
        if zetal < max_float:
            # b[id0] += xb0 / zetal  # (a)
            b[id0] += xb0 * Db[g,0] / (0.5 * Di[0] + zetal * Db[g, 0])  # (b)
            # b[id0] += xb0 * Db[g,0] * c1l  # (c)
            # c[id0] += xb0 * Db[g,0] * c2l / Vi[1]  # (c)
        b[id0] += dDm[g, 0]
        if zetar < max_float:
            # b[ida] += xba / zetar  # (a)
            b[ida] += xba * Db[g,-1] / (0.5 * Di[-1] + zetar * Db[g,-1])  # (b)
            # b[ida] -= xba * Db[g,-1] * c1r  # (c)
            # a[ida-1] -= xba * Db[g,-1] * c2r / Vi[-2]  # (c)
        b[ida] -= dDp[g,-1]
        # N.B.: the division by Vi are needed because we solve for the
        # volume-integrated flux
        idx = np.append(idx, ida)
        b[idx] /= Vi
        b[idx] += st[g, :]

    return a, b, c


def compute_tran_currents(flx, k, Di, xs, PP, BC=(0, 0), curr=None,
                          isSlab=False):
    """Compute the partial currents given by the integral transport equation
    with the input flux flx and using the reduced f.f. escape probabilities
    veP and the transmission probabilities needed in case of reflection at the
    boundary (both packed in PP = veP, tP). An artificial second moment is 
    used to match vanishing current at the boundary in case of reflection.
    When the current curr is provided in input, the linearly anisotropic term
    is accounted in the source if anisotropic scattering cross section data
    are available (currently only available for the slab)."""
    LBC, RBC = BC
    st, ss, chi, nsf, Db = xs
    G, I = st.shape
    veP, tP = PP

    if isSlab:
        # compute the total removal probability per cell
        Pt = np.tile(Di, G).reshape(G, I) * st

    ss0 = ss[:, :, 0, :]
    q0 = compute_source(ss0, chi, nsf, flx, k).reshape(G, I)
    # remind that the source must not be volume-integrated
    # q /= np.tile(Di, G).reshape(G, I)
    if curr is not None:
        # the following will raise IndexError if ss1 is not available
        # and the program will stop to inform the user of missing data
        ss1 = ss[:, :, 1, :]
        # mind that coeffs are already in escape probs!
        q1 = 1.5 * compute_scattering_source(ss1, curr)

    # J = np.zeros((G,I+1),)  # currents at the cell bounds
    Jp = np.zeros((G, I+1),)  # plus partial currents at the cell bounds
    Jm = np.zeros_like(Jp)  # minus partial currents at the cell bounds

    for g in range(G):
        # unpack red. e-probs
        vePp = veP[g, :,:,0]  # 2nd index is i-surface, 3rd index is j-sources
        vePm = veP[g, :,:,1]  # (defined as negative!)
        
        Jp[g, :] = np.dot(vePp, q0[g, :])
        Jm[g, :] = np.dot(-vePm, q0[g, :])  # return a positive Jm
        
        if curr is not None:
            if not isSlab:
                raise RuntimeError('Anisotropic sources are not yet ' +
                                   'supported in 1d curv. geoms.')
            for i in range(I+1):
                for j in range(i):
                    Ajigp[j, i] = (En(4, opl(j+1, i-1, Pt[g, :]))
                                 - En(4, opl( j , i-1, Pt[g, :])))
                for j in range(i, I):
                    Ajigm[j, i] = (En(4, opl(i,  j , Pt[g, :]))
                                 - En(4, opl(i, j-1, Pt[g, :])))
            Jp[g, :] += np.dot(q1[g, :], Ajigp)
            Jm[g, :] += np.dot(q1[g, :], Ajigm)  # check the sign!

        # add bnd. cnds.
        # Zero flux and vacuum have zero incoming current; a boundary term is
        # needed only in case of reflection. Please note that limiting the
        # flux expansion to P1, that is having the flux equal to half the
        # scalar flux plus 3/2 of the current times the mu polar angle, does
        # not reproduce a vanishing total current on the reflected boundary.
        # Therefore, we obtain the second moment to enforce the vanishing of
        # the current. This must be intended as a pure numerical correction,
        # since the real one remains unknown.

        # Apply the boundary terms acting on the 0-th moment (scalar flux)
        # Note: the version quadratic_extrapolation_0 is not used because
        # overshoots the flux estimates
        if LBC == 2 and isSlab:
            # N.B.: reflection at left is used only for the slab geometry,
            # since escape probabilities take already into account ot the
            # geometry effect in the 1d curv. geoms.
            bflx = quadratic_extrapolation(flx[g, :3], Di[:3])
            # bflx = flx[g,0]  # accurate only to 1st order
            # print ('L',g,bflx, flx[g,:3])
            # get the 'corrective' 2-nd moment
            # bflx2_5o8 = -J[g, 0] - 0.25 * bflx  ## it may be negative!
            # ...commented for considering also the contributions from the
            #    right below
            trL = lambda n: np.array([En(n, opl(0, i-1, Pt[g, :]))
                                      for i in range(I+1)])
            # # J[g, :] += 0.5 * np.tile(flx[g,0] / Di[0], I+1) * trL(3)
            # # J[g, :] += 0.5 * bflx * trL(3)
            # Jp[g, :] += 0.5 * bflx * trL(3)
            Jp[g, :] += 0.25 * bflx * tP[g, 0, :]
        if RBC == 2:
            bflx = quadratic_extrapolation(flx[g, -3:][::-1], Di[-3:][::-1])
            # bflx = flx[g,-1]  # accurate only to 1st order
            # print ('R',g, bflx, flx[g,-3:], flx[g,-1])
            trR = lambda n: np.array([En(n, opl(i, I-1, Pt[g, :]))
                                      for i in range(I+1)])
            # # J[g, :] -= 0.5 * np.tile(flx[g,-1] / Di[-1], I+1) * trR(3)
            # # J[g, :] -= 0.5 * bflx * trR(3)
            # Jm[g, :] += 0.5 * bflx * trR(3)
            Jm[g, :] += 0.25 * bflx * tP[g, 1, :]

        # Fix the non-vanishing current at the boundary by propagating the
        # error through the second moments. This is done after transmitting
        # the terms on the 0-th moment to account for possible contributions
        # coming from the opposite boundary. The second moment is already
        # multiplied by 5/8, and it will be multiplied by 5/2 in the equation
        # for the current (and so see below 8 / 5 * 5 / 2 = 4). This 2nd moment
        # may be negative (and unphysical), but we use it only as a numerical
        # correction.
        if isSlab:
            if LBC == 2 and fix_reflection_by_flx2:
                # bflx2_5o16 = -J[g, 0]  # for total curr
                # J[g,:] += 4. * bflx2_5o16 * (3 * trL(5) - trL(3))
                bflx2_5o16 = Jm[g, 0] - Jp[g, 0]  # for partial curr
                Jp[g, :] += 4 * bflx2_5o16 * (3 * trL(5) - trL(3))
            if RBC == 2 and fix_reflection_by_flx2:
                # bflx2_5o16 = J[g, -1]  # for total curr
                # J[g, :] -= 4. * bflx2_5o16 * (3 * trR(5) - trR(3))
                bflx2_5o16 = Jp[g, -1] - Jm[g, -1]  # for partial curr
                Jm[g, :] += 4. * bflx2_5o16 * (3 * trR(5) - trR(3))

        # and here one can check that J[g, :] = Jp[g, :] - Jm[g, :]
    return Jp, Jm


def compute_tran_currents_old(flx, k, Di, xs, BC=(0, 0), curr=None):
    """Compute the partial currents given by the integral transport equation
    with the input flux flx. An artificial second moment is used to match
    vanishing current at the boundary in case of reflection. When the input
    current curr is provided, the linearly anisotropic term is accounted in
    the source if anisotropic scattering cross section data are available."""
    LBC, RBC = BC
    st, ss, chi, nsf, Db = xs
    G, I = st.shape

    ss0 = ss[:, :, 0, :]
    q0 = 0.5 * compute_source(ss0, chi, nsf, flx, k).reshape(G, I)
    # divide the volume-integrated source by the cell volumes if the input
    # flux is volume-integrated
    # q /= np.tile(Di, G).reshape(G, I)
    # warning: divide the source by st for the next numerical integration
    q0 /= st  # term-by-term division
    if curr is not None:
        # the following will raise IndexError if ss1 is not available
        # and the program will stop to inform the user of missing data
        ss1 = ss[:, :, 1, :]
        q1 = 1.5 * compute_scattering_source(ss1, curr) / st

    # J = np.zeros((G,I+1),)  # currents at the cell bounds
    Jp = np.zeros((G, I+1),)  # plus partial currents at the cell bounds
    Jm = np.zeros_like(Jp)  # minus partial currents at the cell bounds

    # compute the total removal probability per cell
    Pt = np.tile(Di, G).reshape(G, I) * st

    for g in range(G):
        # We use here a matrix to store the transfer probabilities, though
        # only the elements on one colums should be stored by the recipro-
        # city theorem.

        # # Net current
        # l = 0  # left as reminder of old implementation with all moments
        # Ajig = np.zeros((I, I+1),)
        # for i in range(I+1):
        #     for j in range(I):
        #         Ajig[j,i] = (En(3, opl(j+1, i-1, Pt[g, :]))
        #                    - En(3, opl( j , i-1, Pt[g, :])))
        #                     if j < i else \
        #                     (En(3, opl(i,  j , Pt[g, :]))
        #                    - En(3, opl(i, j-1, Pt[g, :]))) * (-1)**l
        # J[g, :] += np.dot(q0[g, :], Ajig)

        # Partial currents
        Ajigp = np.zeros((I, I+1),)
        Ajigm = np.zeros_like(Ajigp)
        for i in range(I+1):
            for j in range(i):
                Ajigp[j, i] = (En(3, opl(j+1, i-1, Pt[g, :]))
                             - En(3, opl( j , i-1, Pt[g, :])))
            for j in range(i, I):
                Ajigm[j, i] = (En(3, opl(i,  j , Pt[g, :]))
                             - En(3, opl(i, j-1, Pt[g, :])))
                # the following is not needed for l = 0
                # Ajigm[j, i] *= (-1)**l or if l%2 != 0: Ajigm[j, i] *= -1
        Jp[g, :] += np.dot(q0[g, :], Ajigp)
        Jm[g, :] -= np.dot(q0[g, :], Ajigm)
        
        if curr is not None:
            for i in range(I+1):
                for j in range(i):
                    Ajigp[j, i] = (En(4, opl(j+1, i-1, Pt[g, :]))
                                 - En(4, opl( j , i-1, Pt[g, :])))
                for j in range(i, I):
                    Ajigm[j, i] = (En(4, opl(i,  j , Pt[g, :]))
                                 - En(4, opl(i, j-1, Pt[g, :])))
            Jp[g, :] += np.dot(q1[g, :], Ajigp)
            Jm[g, :] += np.dot(q1[g, :], Ajigm)  # check the sign!

        # add bnd. cnds.
        # Zero flux and vacuum have zero incoming current; a boundary term is
        # needed only in case of reflection. Please note that limiting the
        # flux expansion to P1, that is having the flux equal to half the
        # scalar flux plus 3/2 of the current times the mu polar angle, does
        # not reproduce a vanishing total current on the reflected boundary.
        # Therefore, we obtain the second moment to enforce the vanishing of
        # the current. This must be intended as a pure numerical correction,
        # since the real one remains unknown.

        # Apply the boundary terms acting on the 0-th moment (scalar flux)
        # Note: the version quadratic_extrapolation_0 is not used because
        # overshoots the flux estimates
        if LBC == 2:
            bflx = quadratic_extrapolation(flx[g, :3], Di[:3])
            # bflx = flx[g,0]  # accurate only to 1st order
            # print ('L',g,bflx, flx[g,:3])
            # get the 'corrective' 2-nd moment
            # bflx2_5o8 = -J[g, 0] - 0.25 * bflx  ## it may be negative!
            # ...commented for considering also the contributions from the
            #    right below
            trL = lambda n: np.array([En(n, opl(0, i-1, Pt[g, :]))
                                      for i in range(I+1)])
            # # J[g, :] += 0.5 * np.tile(flx[g,0] / Di[0], I+1) * trL(3)
            # J[g, :] += 0.5 * bflx * trL(3)
            Jp[g, :] += 0.5 * bflx * trL(3)
        if RBC == 2:
            bflx = quadratic_extrapolation(flx[g, -3:][::-1], Di[-3:][::-1])
            # bflx = flx[g,-1]  # accurate only to 1st order
            # print ('R',g, bflx, flx[g,-3:], flx[g,-1])
            trR = lambda n: np.array([En(n, opl(i, I-1, Pt[g, :]))
                                      for i in range(I+1)])
            # # J[g, :] -= 0.5 * np.tile(flx[g,-1] / Di[-1], I+1) * trR(3)
            # J[g, :] -= 0.5 * bflx * trR(3)
            Jm[g, :] += 0.5 * bflx * trR(3)

        # Fix the non-vanishing current at the boundary by propagating the
        # error through the second moments. This is done after transmitting
        # the terms on the 0-th moment to account for possible contributions
        # coming from the opposite boundary. The second moment is already
        # multiplied by 5/8, and it will be multiplied by 5/2 in the equation
        # for the current (and so see below 8 / 5 * 5 / 2 = 4). This 2nd moment
        # may be negative (and unphysical), but we use it only as a numerical
        # correction.
        if LBC == 2 and fix_reflection_by_flx2:
            # bflx2_5o16 = -J[g, 0]  # for total curr
            # J[g,:] += 4. * bflx2_5o16 * (3 * trL(5) - trL(3))
            bflx2_5o16 = Jm[g, 0] - Jp[g, 0]  # for partial curr
            Jp[g, :] += 4 * bflx2_5o16 * (3 * trL(5) - trL(3))
        if RBC == 2 and fix_reflection_by_flx2:
            # bflx2_5o16 = J[g, -1]  # for total curr
            # J[g, :] -= 4. * bflx2_5o16 * (3 * trR(5) - trR(3))
            bflx2_5o16 = Jp[g, -1] - Jm[g, -1]  # for partial curr
            Jm[g, :] += 4. * bflx2_5o16 * (3 * trR(5) - trR(3))

        # and here one can check that J[g, :] = Jp[g, :] - Jm[g, :]
    return Jp, Jm


def compute_diff_currents(flx, Db, Di, BC=(0, 0), xi=None):
    """Compute the currents by Fick's law using the volume-averaged input
    diffusion cofficients."""
    LBC, RBC = BC
    G, I = flx.shape
    # Db, diff. coeff. on cell borders
    J = -2. * Db[:, 1:-1] * (flx[:, 1:] - flx[:, :-1])
    J /= np.tile(Di[1:] + Di[:-1], G).reshape(G, I-1)
    # add b.c.
    zetal, zetar = get_zeta(LBC), get_zeta(RBC)
    if zetal < max_float:
        # JL = flx[:, 0] / zetal  # (a)
        JL = -Db[:,  0] * flx[:, 0] / (0.5 * Di[ 0] + zetal * Db[:, 0])  # (b)
        # c1l, c2l = quadratic_fit_zD(xi[0],
                                    # np.insert(xim(xi[:3]), 0, xi[0]),
                                    # zetal)  # (c)
        # JL = -Db[:, 0] * (c1l * flx[:, 0] + c2l * flx[:, 1])  # (c)
    else:
        JL = np.zeros_like(flx[:, 0])
    if zetar < max_float:
        # JR = -flx[:,-1] / zetar  # (a)
        JR =  Db[:, -1] * flx[:,-1] / (0.5 * Di[-1] + zetar * Db[:,-1])  # (b)
        # c1r, c2r = quadratic_fit_zD(xi[-1],
                                    # np.insert(xim(xi[-3:])[::-1], 0, xi[-1]),
                                    # -zetar)  # (c)
        # JR = -Db[:, -1] * (c1r * flx[:,-1] + c2r * flx[:,-2])  # (c)
    else:
        JR = np.zeros_like(flx[:, -1])
    # avoid possible numerical issues
    # if LBC == 2: JL.fill(0)  # not needed anymore
    # if RBC == 2: JR.fill(0)  # not needed anymore
    J = np.insert(J, 0, JL, axis=1)
    J = np.insert(J, I, JR, axis=1)
    return J


def compute_delta_diff_currents(flx, dD, Di, BC=(0, 0), pCMFD=False):
    """Compute the correction currents by delta diffusion coefficients dD;
    this function is valid only with a CMFD scheme."""
    LBC, RBC = BC
    G, I = dD.shape
    I -= 1
    if pCMFD:
        dDp, dDm = dD
        dJp, dJm = -dDp[:, 1:] * flx, -dDm[:, :-1] * flx
        dJ = dJp, dJm
    else:
        dJ = -dD[:, 1:-1] * (flx[:, 1:] + flx[:, :-1])
        # add b.c.
        dJ = np.insert(dJ, 0, -dD[:,  0] * flx[:,  0], axis=1)
        dJ = np.insert(dJ, I, -dD[:, -1] * flx[:, -1], axis=1)
    return dJ


def solve_inners(flx, ss0, diags, sok, toll=1.e-5, iitmax=10):
    "Solve inner iterations on scattering."
    a, b, c = diags
    G, I = flx.shape
    irr, iti = 1.e+20, 0
    while (irr > toll) and (iti < iitmax):
        # backup local unknowns
        flxold = np.array(flx, copy=True)
        src = sok + compute_scattering_source(ss0, flx)
        flx = solveTDMA(a, b, c, src).reshape(G, I)
        ferr = np.where(flx > 0., 1. - flxold / flx, flx - flxold)
        irr = abs(ferr).max()
        iti += 1
        lg.debug(" +-> it={:^4d}, err={:<+13.6e}".format(iti, irr))
    return flx


def compute_delta_D(flx, J_diff, pJ_tran, pCMFD=False, vrbs=False):
    """Compute the delta diffusion coefficients (already divided by the cell
    width, plus the possible extrapolated length); the input currents are used
    differently accoring to pCMFD."""
    Jp, Jm = pJ_tran
    J_tran = Jp - Jm
    dD = J_diff - J_tran
    if vrbs or (lg.DEBUG >= lg.root.level):
        print('currents...')
        print('flux: ' + str(flx))
        print('diff: ' + str(J_diff))
        print('tran: ' + str(J_tran))
    
    if np.any(flx[:, 0] <= 0.):
        raise RuntimeError('Detected flx at LB <= 0: ' + str(flx[:, 0]))
    if np.any(flx[:,-1] <= 0.):
        raise RuntimeError('Detected flx at RB <= 0: ' + str(flx[:,-1]))
    
    if pCMFD:
        half_Jdiff = 0.5 * J_diff
        dDp, dDm = half_Jdiff - Jp, Jm + half_Jdiff
        dDm[:, 1:-1] /= flx[:, 1:]
        dDp[:, 1:-1] /= flx[:, :-1]
        dDm[:, 0], dDp[:, -1] = dD[:, 0] / flx[:, 0], dD[:, -1] / flx[:,-1]
        dDm[:,-1].fill(np.nan)  # N.B.: these values must not be used!
        dDp[:, 0].fill(np.nan)
        dD = dDp, dDm
    else:
        # use the classic CMFD scheme
        dD[:, 1:-1] /= (flx[:, 1:] + flx[:, :-1])
        dD[:,  0] /= flx[:,  0]
        dD[:, -1] /= flx[:, -1]
    return dD


def compute_D(Di, flx, pJ, BC=(0, 0), zero=1.e-6, chk=False):
    """Derive new diffusion coefficients by Fick's law with the 
    input flux flx and partial currents J."""
    Jp, Jm = pJ
    LBC, RBC = BC
    G, I = flx.shape
    J, mflx_diff = Jp - Jm, flx[:, :-1] - flx[:, 1:]
    if J.size != G * (I + 1):
        raise ValueError('Unexpected size of input current.')
    Db = np.array(J[:,1:-1], copy=True)
    Db *= np.tile(Di[1:] + Di[:-1], G).reshape(G, I-1) / 2.
    # Because of the flux-limiting principle, the current must go to
    # zero faster than the flux. This may not happen in simple
    # diffusion. In this case, a division by zero will occur here.
    # If current is zero instead, any value of the diffusion coeff
    # will do. 
    idx = np.abs(Db) > zero
    Db[idx] /= mflx_diff[idx]
    
    # get values of D at the boundary cells
    zetal, zetar = get_zeta(LBC), get_zeta(RBC)
    #: JL = -Db[:,  0] * flx[:, 0] / (0.5 * Di[ 0] + zetal * Db[:, 0])
    #: JR =  Db[:, -1] * flx[:,-1] / (0.5 * Di[-1] + zetar * Db[:,-1])
    a, b = J[:, 0] / flx[:, 0], J[:,-1] / flx[:,-1]
    DbL = - a * 0.5 * Di[0] / (1. + a * zetal)
    DbR = b * 0.5 * Di[-1] / (1. - b * zetar)
    if LBC == 2: DbL.fill(0.)  # but any value would be fine
    if RBC == 2: DbR.fill(0.)  # ...same as above
    Db = np.insert(Db, 0, DbL, axis=1)
    Db = np.insert(Db, I, DbR, axis=1)
    
    if chk:
        c = np.tile(Di[1:] + Di[:-1], G).reshape(G, I-1) / 2.
        Jcmp = Db[:,1:-1] * mflx_diff / c
        if not np.allclose(Jcmp, J[:, 1:-1]):
            lg.debug("Computed/input currents mismatch")
            lg.debug('Jout', Jcmp)
            lg.debug('Jin ', J[:, 1:-1])
        JL = -Db[:,  0] * flx[:, 0] / (0.5 * Di[ 0] + zetal * Db[:, 0])
        JR =  Db[:, -1] * flx[:,-1] / (0.5 * Di[-1] + zetar * Db[:,-1])
        if not np.allclose(JL, J[:, 0]):
            lg.debug("Computed/input currents mismatch at LB")
            lg.debug('JLout', JL)
            lg.debug('JLin ', J[:, 0])
        if not np.allclose(JR, J[:,-1]):
            lg.debug("Computed/input currents mismatch at RB")
            lg.debug('JRout', JR)
            lg.debug('JRin ', J[:,-1])
        input('ok')
    return Db


def compute_delta_J(J_diff, pJ_tran, pCMFD=False):  # potentially obsolete
    'Return the delta current (negative, i.e. with a change of sign).'
    Jp, Jm = pJ_tran
    if pCMFD:
        raise ValueError('not available yet')
    return J_diff - Jp + Jm


def solve_outers(flx, k, data, xs, slvr_opts, dD=None):
    "Solve the outer iterations by the power method."
    # unpack objects
    st, ss, chi, nsf, Db = xs
    ss0 = ss[:, :, 0, :]
    G, I = data.G, data.I
    LBC, RBC = data.BC
    itsmax, tolls = slvr_opts.itsmax, slvr_opts.tolls
    oitmax, iitmax = itsmax  # max nb of outer/inner iterations
    # tolerance on residual errors for the solution in outers/inners
    otoll, itoll = tolls

    # setup the tri-diagonal matrix and the source s
    # MEMO: the setup of the system eqs is made many times where only dD
    # changes - a re-code would be needed to avoid redundant setup
    diags = set_diagonals(st, Db, data, dD)

    # start outer iterations
    err, ito = 1.e+20, 0
    # evaluate the initial source
    s = compute_fission_source(chi, nsf, flx)
    while (err > otoll) and (ito < oitmax):
        # backup local unknowns
        serr, kold = np.array(s, copy=True), k
        sold = serr  # ...just a reference

        # solve the diffusion problem with inner iterations on scattering
        flx = solve_inners(flx, ss0, diags, s / k, itoll, iitmax)

        # evaluate the new source
        s = compute_fission_source(chi, nsf, flx)

        # new estimate of the eigenvalue
        k *= np.sum(flx * s.reshape(G, I)) / np.sum(flx * sold.reshape(G, I))

        # np.where seems to bug when treating flattened arrays...
        # serr = np.where(s > 0., 1. - sold / s, s - sold)
        mask = s > 0.
        serr[mask] = 1. - serr[mask] / s[mask]
        serr[~mask] = s[~mask] - serr[~mask]
        err = abs(serr).max()
        ito += 1
        lg.info("<- it={:^4d}, k={:<13.6g}, err={:<+13.6e}".format(
            ito, k, err
        ))

    # normalize the (volume-integrated) flux to the number of cells I times
    # the number of energy groups
    return flx / np.sum(flx) * I * G, k


def load_refSN_solutions(ref_flx_file, G, Di, Dbnd=0.):
    "load reference SN flux and calculate reference currents."
    k_SN, flxm_SN, I = 0, 0, Di.size
    if os.path.isfile(ref_flx_file):
        lg.debug("Retrieve reference results.")
        ref_data = np.load(ref_flx_file)
        k_SN, flxm_SN = ref_data['k'], ref_data['flxm']
        G_SN, M_SN, I_SN = flxm_SN.shape
        if G != G_SN:
            raise ValueError('Reference flux has different en. groups')
        if I != I_SN:
            raise ValueError('Reference flux has different nb. of cells')

        # normalize the reference flux
        flxm_SN *= (I * G) / np.sum(flxm_SN[:, 0, :])
        d_flxm0R = - np.array([
            estimate_derivative(flxm_SN[g, 0, -3:][::-1], Di[-3:][::-1]) /
            flxm_SN[g, 0, -1] for g in range(G)
        ])
        print('Estimates of the extrapolated lengths ' + str(-1. / d_flxm0R))
        print('Values used in the diffusion solver ' + str(2.13 * Dbnd))
    else:
        lg.debug('Missing file ' + ref_flx_file)

    return k_SN, flxm_SN


def check_current_solutions():
    # compute the corrective currents (determined only for debug)
    J_corr = compute_delta_diff_currents(flxd, dD, Di, BC, slvr_opts.pCMFD)
    print("F_ref ", flxm_SN[0, 0, -6:] / Di[-6:])
    print("F_diff", flx_save[:, :, itr][0, -6:] / Di[-6:])
    print("F_dif*", flx[0, -6:] / Di[-6:])
    print("J_ref ", flxm_SN[0, 1, -6:] / Di[-6:])  # cell-averaged!
    print("J_diff", J_diff[0, -6:])
    print("J_tran", J_tran[0, -6:])
    print("J_corr", J_corr[0, -6:])
    print("    dD", dD[0, -6:])


def plot_fluxes(xm, flx, L):
    # prepare the plot
    fig = plt.figure(0)
    # define a fake subplot that is in fact only the plot.
    ax = fig.add_subplot(111)

    # change the fontsize of major/minor ticks label
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    ax.plot(xm, flx[0, :], label='fast')
    ax.plot(xm, flx[1, :], label='thermal')
    plt.xlabel(r'$x$ $[cm]$', fontsize=16)
    plt.ylabel(r'$\phi$ $[n/cm^2\cdot s]$', fontsize=16)
    plt.title(r'Diffusion fluxes - DT', fontsize=16)
    plt.xlim(0, L)
    plt.ylim(0, max(flx.flatten()) + 0.2)
    plt.grid(True, 'both', 'both')
    ax.legend(loc='best', fontsize=12)
    # fig.savefig('diffusion_fluxes_DT.png',dpi=150,bbox_inches='tight')
    plt.show()


def solve_RMits(data, xs, flx, k, slvr_opts, filename=None):
    """Solve the Ronen Method by non-linear iterations based on CMFD and
    diffusion."""
    # unpack data
    Db = xs[-1]  # retrieve Db which does not change with CMFD
    ss = xs[1]  # check for scattering anisotropy in input xs data
    lin_anis = False
    try:
        if np.sum(ss[:, :, 1, :]) > 0.:
            lin_anis = True
    except:
        pass  # continue without raising an error
    xi, Di, Vi, Si = data.xi, data.Di, data.Vi, data.Si
    G, I, BC, geo = data.G, data.I, data.BC, data.geometry_type
    itsmax, tolls = slvr_opts.itsmax, slvr_opts.tolls
    ritmax, rtoll = slvr_opts.ritmax, slvr_opts.rtoll
    noacc_rit = slvr_opts.noacc_rit
    if slvr_opts.ritmax == 0:
        lg.warning('You called RM its, but they were disabled at input.')
    
    if slvr_opts.Anderson:
        AA = AndersonAcceleration(opts=slvr_opts, size=flx.size)
        lg.info("Reset the number of unaccelerated rits to m - 1.")
        noacc_rit = AA.Anderson_depth - 1  # noticed optimal performance
    
    lg.info("Calculate the first flight escape and transmission probabilities")
    tr_data = None if (geo == "slab") else \
        calculate_tracking_data(xi, slvr_opts.ks,
            sphere=True if "spher" in geo else False, 
            quadrule=slvr_opts.GaussQuadrature)
    # N.B.: Remind that volumes are per unit angle in diffusion calculations,
    # but full ones are needed to compute the escape and transfer probabilities
    # -> compute_cell_volumes(xi, geo, per_unit_angle=False) != Vi
    vareps = calculate_eprobs(xi, xs[0], tr_data, geometry_type=geo)
        # , Vj=compute_cell_volumes(xi, geo, per_unit_angle=False))
    Sf = compute_cell_surfaces(xi, geo, per_unit_angle=False)  # != Si
    rps = calculate_sprobs(vareps, Sf)
    tp = calculate_tprobs(vareps, xs[0], Sf)
    PP = rps, tp  # pack probs-related data
    lg.info("-o"*22)
    
    # # load reference currents
    # ref_flx_file = "../SNMG1DSlab/LBC1RBC0_I%d_N%d.npz" % (I, 64)
    # k_SN, flxm_SN = load_refSN_solutions(ref_flx_file, G, Di, Dbnd=D[:,-1])

    # keep track of partial solutions on external files
    flx_save = np.empty([G, I, ritmax + 1])
    k_save = np.full(ritmax + 1, -1.)

    err, itr, kinit = 1.e+20, 0, k
    Dflxm1, dD = np.zeros_like(flx), None
    while (err > rtoll) and (itr < ritmax):
        k_save[itr], flx_save[:, :, itr] = k, flx

        # revert to flux density
        # (this division does not seem to affect the final result though)
        flxd, Jd = flx / Vi, None  # division on last index

        # compute the currents by diffusion and finite differences
        # (not used later in isotropic problems without CMFD)
        J_diff = compute_diff_currents(flxd, Db, Di, BC, xi)
        if lin_anis:
            Jd = J_diff + compute_delta_diff_currents(flxd, dD, Di, BC,
                                                      slvr_opts.pCMFD)
            Jd = (Jd[:, 1:] + Jd[:, :-1]) / 2.  # simple cell-average

        # compute the currents by integral transport (Ronen Method)
        # #lg.warning("USE THE REFERENCE SN FLUX IN THE TRANSPORT OPERATOR")
        # # rflx = flxm_SN[:, 0, :] / Vi
        # # J_tran = compute_tran_currents(rflx, k_SN, Di, xs, BC)
        # pJ_tran = compute_tran_currents_old(flxd, k, Di, xs, BC, Jd)
        # print('Jp',pJ_tran[0][0,:],'\n','Jm',pJ_tran[1][0,:],'\n ---')
        # Remind that Jp, Jm = *pJ_tran, J = Jp - Jm
        pJ_tran = compute_tran_currents(flxd, k, Di, xs, PP, BC, Jd,
                                        isSlab=(geo=='slab'))
        # print('flxd', flxd)
        # print("J_diff", J_diff)
        # print('Jp',pJ_tran[0][0,:],'\nJm',pJ_tran[1][0,:],'\n ---')

        if slvr_opts.CMFD:
            # compute the corrective delta-diffusion-coefficients
            dD = compute_delta_D(flxd, J_diff, pJ_tran, slvr_opts.pCMFD)
        else:
            # print('before',Db)
            Db, dD = compute_D(Di, flx, pJ_tran, BC), None
            # print('after', Db)
            xs[-1] = Db  # update bnd D coeffs

        flxold, kold = np.array(flx, copy=True), k
        lg.info("Start the diffusion solver (<- outer its. / -> inner its.)")
        flx, k = solve_outers(flx, k, data, xs, slvr_opts, dD)
        Dflxm2 = np.array(Dflxm1, copy=True)
        Dflxm1 = flxres = flx - flxold  # flux residual
        # check_current_solutions()
        # print('--- it %d ---' % itr)
        # print(flxold[0, :])
        # print(flx[0, ])

        # possible techniques to accelerate the convergence rate
        itr0 = itr + 1 - noacc_rit
        # Aitken extrapolation
        if (itr0 > 0) and (err < rtoll * 100) and slvr_opts.Aitken:
            lg.info("<-- Apply Aitken extrapolation on the flux -->")
            flx -=  (Dflxm1**2 / (Dflxm1 - Dflxm2))
        # Successive Over-Relaxation (SOR)
        if (itr0 > 0) and slvr_opts.SOR:
            lg.info("<-- Apply SOR on the flux -->")
            flx = slvr_opts.wSOR * flx + (1 - slvr_opts.wSOR) * flxold
        # Anderson implementation to accelerate yet for k < m
        if slvr_opts.Anderson:
            flx = AA(itr0, flxres, flxold, flx, k_restart=AA.m)
            # print(flx[0, :])
            # input('wait')  # debug
        
        # evaluate the flux differences through successive iterations
        ferr = np.where(flx > 0., 1. - flxold / flx, flxres)
        
        err = abs(ferr[np.unravel_index(abs(ferr).argmax(), (G, I))])
        itr += 1
        lg.info("+RM+ it={:^4d}, k={:<13.6g}, err={:<+13.6e}".format(
            itr, k, err
        ))
        lg.info("{:^4s}{:^13s}{:^6s}{:^13s}".format(
            "G", "max(err)", "at i", "std(err)"
        ))
        for g in range(G):
            ierr, estd = abs(ferr[g, :]).argmax(), abs(ferr[g, :]).std()
            lg.info(
                "{:^4d}{:<+13.6e}{:^6d}{:<+13.6e}".format(
                         g, ferr[g, ierr], ierr + 1, estd)
            )
        # input('press a key to continue...')
    #
    # plot fluxes
    # plot_fluxes(xim(xi),flx,L)
    # save fluces
    # save_fluxes('diffusion_fluxes_DT.dat',xm,flx)
    if itr == ritmax != 0:
        lg.warning(' ---> !!! MAX NB. of R.ITS attained !!!')

    k_save[itr], flx_save[:, :, itr] = k, flx  # store final values
    lg.info("Initial value of k was %13.6g." % kinit)
    if filename is not None:
        # filename = "output/kflx_LBC%dRBC%d_I%d" % (*BC,I)
        np.save(filename + ".npy",
            [k_save, flx_save, xi, xs[0], dD], allow_pickle=True)
        # np.save(filename.replace('kflx','err'), [kerr_save, ferr_save],
        #         allow_pickle=True)
        # np.save(filename + ".npy", [k_save, flx_save, xi, xs[0], dD],
        #         allow_pickle=True)
        with open(filename + ".dat", 'w') as f:
            i = 0
            while k_save[i] >= 0:
                flxsv = flx_save[:, :, i].flatten()
                f.write(to_str(np.append(k_save[i], flxsv)))
                i += 1
                if i == len(k_save):
                    break

    return flx, k


def run_calc_with_RM_its(idata, slvr_opts, filename=None):
    lg.info("Prepare input xs data")            
    xs = unfold_xs(idata)
    lg.info("-o"*22)

    # Call the diffusion solver without CMFD corrections
    lg.info("Initial call to the diffusion solver without RM corrections")
    lg.info("Start the diffusion solver (<- outer its. / -> inner its.)")
    flx, k = np.ones((idata.G, idata.I),), 1.  # initialize the unknowns
    flx, k = solve_outers(flx, k, idata, xs, slvr_opts)
    lg.info("-o"*22)

    # start Ronen iterations
    if slvr_opts.ritmax > 0:
        lg.info("Start the Ronen Method by CMFD iterations")
        flx, k = solve_RMits(idata, xs,  # input data
                             flx, k,  # initial values
                             slvr_opts,  # for its opts
                             filename)
    lg.info("-o"*22)
    lg.info("*** NORMAL END OF CALCULATION ***")
    return flx, k


if __name__ == "__main__":

    lg.info("Verify the code with the test case from the M&C article")
    from tests.homogSlab2GMC2011 import Homog2GSlab_data as data
    #from tests.heterSlab2GRahnema1997 import Heter2GSlab_data as data

    slvr_opts = solver_options()
    filename = "output/kflx_LBC%dRBC%d_I%d" % (data.LBC, data.RBC, data.I)
    flx, k = run_calc_with_RM_its(data, slvr_opts, filename)
