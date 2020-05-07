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
import os
import sys
import logging as lg
import numpy as np
from scipy.special import expn as En

__title__ = "Multigroup diffusion in 1D slab by finite differences"
__author__ = "D. Tomatis"
__date__ = "15/11/2019"
__version__ = "1.3.0"

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


# get the mid-points of cells defined in the input spatial mesh x
def xim(x):
    return (x[1:] + x[:-1]) / 2.


class input_data:
    """Geometry and material input data of the 1D problem. Possible options
    of geometry_type are slab, cylindrical and spherical. Allowed boundary
    conditions: 0-vacuum, 1-zero flux and 2-reflection. Units are in cm."""

    def __init__(self, xs_media, media, xi, geometry_type='slab',
                 LBC=0, RBC=0, per_unit_angle=True):
        self.geometry_type, self.xi = geometry_type.lower(), xi
        self.LBC, self.RBC = LBC, RBC
        self.xs_media, self.media = xs_media, media
        self.check_input()
        self.compute_cell_width()
        self.compute_mid_cell_coordinate()
        self.compute_cell_surfaces(per_unit_angle)
        self.compute_cell_volumes(per_unit_angle)

    @property
    def I(self):
        return self.xi.size - 1

    @property
    def L(self):
        return self.xi[-1]

    @property
    def BC(self):
        return self.LBC, self.RBC

    @property
    def G(self):
        return self.xs_media[next(iter(self.xs_media))]['st'].size

    @property
    def Lss(self):
        random_ss, L = self.xs_media[next(iter(self.xs_media))]['ss'], 1
        if random_ss.ndim == 3:
            L = max([self.xs_media[m]['ss'].shape[-1] for m in self.xs_media])
        return L - 1
    
    def compute_cell_width(self):
        self.Di = self.xi[1:] - self.xi[:-1]
    
    def compute_mid_cell_coordinate(self):
        self.xim = xim(self.xi)
    
    def compute_cell_surfaces(self, per_unit_angle=True):
        self.Si = compute_cell_surfaces(self.xi, geo=self.geometry_type,
                                        per_unit_angle=per_unit_angle)
    
    def compute_cell_volumes(self, per_unit_angle=True):
        self.Vi = compute_cell_volumes(self.xi, geo=self.geometry_type,
                                       per_unit_angle=per_unit_angle)

    def check_input(self):
        lg.info("Geometry type is " + self.geometry_type)
        if (self.geometry_type != 'slab') and \
           (self.geometry_type != 'cylinder') and \
           (self.geometry_type != 'cylindrical') and \
           (self.geometry_type != 'spherical') and \
           (self.geometry_type != 'sphere'):
            raise ValueError('Unknown geometry_type ' + self.geometry_type)
        if not isinstance(self.LBC, int):
            raise TypeError('LBC must be integer.')
        if (self.LBC < 0) and (self.LBC > 2):
            raise ValueError('Check LBC, allowed options ')
        if not isinstance(self.RBC, int):
            raise TypeError('RBC must be integer.')
        if (self.RBC < 0) and (self.RBC > 2):
            raise ValueError('Check RBC, allowed options ')
        if (self.geometry_type != 'slab') and (self.LBC != 2):
            raise ValueError('Curvilinear geometries need LBC = 2.')
        if not isinstance(self.xs_media, dict):
            raise TypeError('The input xs_media is not a dictionary.')
        if not isinstance(self.media, list):
            raise TypeError('The input media is not a list.')
        media_set = set(m[0] for m in self.media)
        xs_media_set = set(self.xs_media.keys())
        if xs_media_set.union(media_set) != xs_media_set:
            raise ValueError('xs media dict has missing keys, check media.')
        rbnd = [m[1] for m in self.media]
        if sorted(rbnd) != rbnd:
            raise ValueError('media list must be in order from left to right!')
        if not np.isclose(max(rbnd), self.L):
            raise ValueError('Please check the right bounds of media (>L?)')

    def __str__(self):
        print("Geometry type is " + self.geometry_type)
        print("Boundary conditions: (left) LBC=%d and (right) RBC=%d." %
              (self.LBC, self.RBC))
        print("B.C. legend: 0-vacuum, 1-zero flux and 2-reflection.")
        print("Number of energy groups: %d" % self.G)
        print("Number of spatial cells: %d" % self.I)
        print("Spatial mesh xi\n" + str(self.xi))
        print("Media list:\n" + str(self.media))
        print(" with xs:\n" + str(self.xs_media))


class solver_options:
    """Object collecting (input) solver options. INFO: set ritmax to 0 to
    skip Ronen iterations."""
    toll = 1.e-6  # default tolerance
    nbitsmax = 100  # default nb. of max iterations (its)

    def __init__(self, iitmax=nbitsmax, oitmax=nbitsmax, ritmax=10,
                 pCMFD=False, otoll=toll, itoll=toll, rtoll=toll,
                 CMFD=True, wSOR=None, Aitken=False, Anderson_depth=5,
                 Anderson_relaxation=1, noacc_rit=0):
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
        self.Anderson_relaxation = Anderson_relaxation  # set to 1 to disable
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
        if self.Anderson_depth < 0:
            raise ValueError('Detected negative dim of Anderson subspace.')
        if not (0 < self.Anderson_relaxation <= 1):
            raise ValueError('relaxation for Anderson out of allowed bounds.')

    @property
    def itsmax(self):
        "pack nb. of outer and inner iterations"
        return self.oitmax, self.iitmax

    @property
    def tolls(self):
        "pack tolerances on residual errors in outer and inner iterations"
        return self.otoll, self.itoll


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


def compute_cell_volumes(xi, geo=None, per_unit_angle=True):
    """Compute the volumes of the cells in the mesh xi. These volumes
    are per unit of transversal surface of the slab or per unit of
    angle in the other frame (azimuthal angle in the cylinder or
    per cone unit in the sphere) if per_unit_angle is true. In this case
    the real volumes in the cylinder must be multiplied by 2*np.pi,
    or by 4*np.pi for the sphere."""
    # V = xi[1:] - xi[:-1]  # = Di for the slab as default case
    V = np.diff(xi)  # same as above, but possible vectorized
    if geo != 'slab':
        xm = xim(xi)
        if 'cylind' in geo:
            V *= (xm if per_unit_angle else 2 * xm)
        elif 'spher' in geo:
            cm = (4. * xm**2 - xi[1:] * xi[:-1]) / 3.
            V *= (cm if per_unit_angle else 4 * cm)
            # V *= (r[1:]**2 + r[1:]*r[:-1] + r[:-1]**2) / 3.
        else:
            raise ValueError("Unknown geometry type " + geo)
        if not per_unit_angle:
            V *= np.pi
    return V


def compute_cell_surfaces(xi, geo=None, per_unit_angle=True):
    "Compute the cell outer surfaces with the input 1d mesh xi."
    S = np.ones(xi.size - 1)
    if geo != 'slab':
        if 'cylind' in geo:
            S = (xi[1:] if per_unit_angle else 2 * xi[1:])
        elif 'spher' in geo:
            S = (xi[1:]**2 if per_unit_angle else 4 * xi[1:]**2)
        else:
            raise InputError("Unknown geometry type " + geo)
        if not per_unit_angle:
            S *= np.pi
    return S


def first_order_coeff_at_interfaces(f, Vi):
    """Compute the diffusion coefficient by 1st order finite-difference
    currents determined at both sides of an interface."""
    G, I = f.shape
    Im1 = I - 1
    
    fb = 2 * f[:, 1:] * f[:, :-1]
    return fb / (f[:, 1: ] * np.tile(Vi[:-1], G).reshape(G, Im1) +
                 f[:, :-1] * np.tile(Vi[1: ], G).reshape(G, Im1))


def vol_averaged_at_interface(f, Vi):
    "Compute surface quantities by volume averaging (slab geometry)."
    G, I = f.shape
    Im1 = I - 1

    # fb, volume-average quantity on cell borders
    fb = (f[:, 1: ] * np.tile(Vi[1: ], G).reshape(G, Im1) +
          f[:, :-1] * np.tile(Vi[:-1], G).reshape(G, Im1))
    return fb / np.tile(Vi[1:] + Vi[:-1], G).reshape(G, Im1)


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

        # add b.c.
        # # force the extrap. lengths estimated by the reference solution
        # zetar = np.array([0.67291, 0.33227])[g]
        b[id0] += Db[g, 0] * xb0 / (0.5 * Di[ 0] + zetal * Db[g, 0]) + dDm[g, 0]
        b[ida] += Db[g,-1] * xba / (0.5 * Di[-1] + zetar * Db[g,-1]) - dDp[g,-1]
        # N.B.: the division by Vi are needed because we solve for the
        # volume-integrated flux
        idx = np.append(idx, ida)
        b[idx] /= Vi
        b[idx] += np.full(I, st[g, :])

    return a, b, c


def estimate_derivative(flx, Di):
    """Fit flx by quadratic interpolation at the cell centers and return the
    value of its derivative at the initial point."""
    if len(Di) != 3:
        raise ValueError("invalid input cell-width vector Di")
    if flx.size % 3 != 0:
        raise ValueError("invalid input flux flx")
    r = Di[1] / Di[0]
    a1 = 2. + r
    b1 = a1 + r + Di[2] / Di[0]
    c = 4. / Di[0]**2 / (1. - b1) * ((flx[1] - flx[0]) / (a1 - 1.) -
                                     (flx[2] - flx[1]) / (b1 - 1.))
    b = 2. / Di[0] * (flx[1] - flx[0]) / (a1 - 1.) - c * Di[0] / 2. * (a1 + 1.)
    return b


def quadratic_extrapolation(flx, Di):
    """Extrapolate the flux at the boundary by a quadratic polynomial which
    interpolates the flux at the mid of the input three cells. This assumption
    is consistent with the centered fintite differences used in the diffusion
    solver."""
    if len(Di) != 3:
        raise ValueError("invalid input cell-width vector Di")
    if flx.size % 3 != 0:
        raise ValueError("invalid input flux flx")
    r = Di[1] / Di[0]
    a = 2. + r
    b = a + r + Di[2] / Di[0]
    df1, df2 = (flx[1] - flx[0]) / (a - 1.), (flx[2] - flx[1]) / (b - a)
    bflx = flx[0] - df1 + a / (1. - b) * (df1 - df2)
    return bflx


def quadratic_extrapolation_0(flx, Di):
    """Extrapolate the flux at the boundary by a quadratic polynomial whose
    integral over the first three cells at the boundary yields the volume-
    integrated flux (given by flx_i * Di_i thanks to the theorem of the mean).
    """
    if len(Di) != 3:
        raise ValueError("invalid input cell-width vector Di")
    if flx.size % 3 != 0:
        raise ValueError("invalid input flux flx")
    x0, x1, x2 = np.cumsum(Di) - Di[0] / 2.
    xA, xB, xC, xD = 0., Di[0], Di[0] + Di[1], np.sum(Di[:3])
    g0 = xB**2 + xA * xB + xA**2
    g1 = xC**2 + xB * xC + xB**2
    g2 = xD**2 + xC * xD + xC**2
    df1, df2 = (flx[1] - flx[0]) / (x1 - x0), (flx[2] - flx[1]) / (x2 - x1)
    a = (g2 - g1) / (g1 - g0) * (x1 - x0) / (x2 - x1) - 1.
    bflx = flx[0] - df1 + (df1 - df2) / a
    return bflx


def opl(j, i, Ptg):
    "Calculate the (dimensionless) optical length between j-1/2 and i+1/2."
    # if j > i, the slicing returns an empty array, and np.sum returns zero.
    return np.sum(Ptg[j:i+1])


def compute_tran_currents(flx, k, Di, xs, BC=(0, 0), curr=None):
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


def compute_diff_currents(flx, Db, Di, BC=(0, 0)):
    """Compute the currents by Fick's law using the volume-averaged input
    diffusion cofficients."""
    LBC, RBC = BC
    G, I = flx.shape
    # Db, diff. coeff. on cell borders
    J = -2. * Db[:, 1:-1] * (flx[:, 1:] - flx[:, :-1])
    J /= np.tile(Di[1:] + Di[:-1], G).reshape(G, I-1)
    # add b.c.
    zetal, zetar = get_zeta(LBC), get_zeta(RBC)
    JL = -Db[:,  0] * flx[:, 0] / (0.5 * Di[ 0] + zetal * Db[:, 0])
    JR =  Db[:, -1] * flx[:,-1] / (0.5 * Di[-1] + zetar * Db[:,-1])
    # avoid possible numerical issues
    if LBC == 2: JL.fill(0)
    if RBC == 2: JR.fill(0)
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


def to_str(v, fmt='%.13g'):
    return ', '.join([fmt % i for i in v]) + '\n'


def check_current_solutions():
    # compute the corrective currents (determined only for debug)
    J_corr = compute_delta_diff_currents(flxd, dD, Di, BC, pCMFD)
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


def roll_matrix(M, c):
    return np.concatenate([M[:, 1:],
                           np.expand_dims(c, axis=1)], axis=1)


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
    xi, Di, Vi = data.xi, data.Di, data.Vi
    G, I, BC = data.G, data.I, data.BC
    itsmax, tolls = slvr_opts.itsmax, slvr_opts.tolls
    ritmax, rtoll = slvr_opts.ritmax, slvr_opts.rtoll
    noacc_rit = slvr_opts.noacc_rit
    if slvr_opts.ritmax == 0:
        lg.warning('You called RM its, but they were disabled at input.')
    elif data.geometry_type != 'slab':
        raise RuntimeError('RM has not been ported to curv. geoms.')
    
    if slvr_opts.Anderson_depth is not None:
        Anderson = True
        m, betak = slvr_opts.Anderson_depth, slvr_opts.Anderson_relaxation
        Fk = np.zeros((flx.size, m),)
        Xk = np.zeros_like(Fk)
        lg.info("Reset the number of unaccelerated rits to m - 1.")
        noacc_rit = m - 1  # noticed optimal performance
    else:
        Anderson = False
    
    # # load reference currents
    # ref_flx_file = "../SNMG1DSlab/LBC1RBC0_I%d_N%d.npz" % (I, 64)
    # k_SN, flxm_SN = load_refSN_solutions(ref_flx_file, G, Di, Dbnd=D[:,-1])

    # keep track of partial solutions on external files
    flx_save = np.empty([G, I, ritmax + 1])
    k_save = np.full(ritmax + 1, -1.)

    err, itr, kinit = 1.e+20, 0, k
    Dflxm1 = np.zeros_like(flx)
    while (err > rtoll) and (itr < ritmax):
        k_save[itr], flx_save[:, :, itr] = k, flx

        # revert to flux density
        # (this division does not seem to affect the final result though)
        flxd, Jd = flx / Vi, None  # division on last index

        # compute the currents by diffusion and finite differences
        # (not used later in isotropic problems without CMFD)
        J_diff = compute_diff_currents(flxd, Db, Di, BC)
        if lin_anis:
            Jd = J_diff + compute_delta_diff_currents(flxd, dD, Di, BC,
                                                      slvr_opts.pCMFD)
            Jd = (Jd[:, 1:] + Jd[:, :-1]) / 2.  # simple cell-average

        # compute the currents by integral transport (Ronen Method)
        # #lg.warning("USE THE REFERENCE SN FLUX IN THE TRANSPORT OPERATOR")
        # # rflx = flxm_SN[:, 0, :] / Vi
        # # J_tran = compute_tran_currents(rflx, k_SN, Di, xs, BC)
        pJ_tran = compute_tran_currents(flxd, k, Di, xs, BC, Jd)
        # Remind that Jp, Jm = *pJ_tran, J = Jp - Jm

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
        # print(flxold[0, -6:])
        # print(flx[0, -6:])

        # possible techniques to accelerate the convergence rate
        itr0 = itr - noacc_rit
        # Aitken extrapolation
        if (itr0 >= 0) and (err < rtoll * 100) and slvr_opts.Aitken:
            lg.info("<-- Apply Aitken extrapolation on the flux -->")
            flx -=  (Dflxm1**2 / (Dflxm1 - Dflxm2))
        # Successive Over-Relaxation (SOR)
        if (itr0 >= 0) and (slvr_opts.wSOR is not None):
            lg.info("<-- Apply SOR on the flux -->")
            flx = slvr_opts.wSOR * flx + (1 - slvr_opts.wSOR) * flxold
        # Anderson implementation to accelerate yet for k < m
        if Anderson:
            fk, xk = flxres.flatten(), flxold.flatten()
            # ------------------------------------------------------------------
            # ---------- (uncomment only one version in the following) ---------
            # ------------------------------------------------------------------
            # # version (a) - constrained minimization
            # if itr0 >= 0:
                # mk = min(itr0 + 1, slvr_opts.Anderson_depth)
                # Fr = Fk[:, -mk:] - np.tile(fk, (mk, 1)).T
                # # alphm1 = np.dot(np.linalg.inv(np.dot(Fr.T, Fr)
                                # # # + 0.05 * np.eye(mk)  # regularization
                                # # ), np.dot(Fr.T, -fk))
                # alphm1 = np.linalg.lstsq(Fr, -fk, rcond=None)[0]
                # # from scipy.optimize import nnls, lsq_linear
                # # gams = nnls(DFk, fk)[0]  # non-negative LS, or bounded as:
                # # gams = lsq_linear(DFk, fk, bounds=(0, np.inf),
                                  # # lsq_solver='exact', verbose=1).x
                # alphmk = 1 - np.sum(alphm1)
                # Gk = (Fk + Xk)[:, -mk:]
                # flx = (1 - betak) * (alphmk * xk
                       # + np.dot(Xk[:, -mk:], alphm1)).reshape(flx.shape) \
                          # + betak * (alphmk * flx
                       # + np.dot(Gk, alphm1).reshape(flx.shape))
                # print(mk, np.insert(alphm1, -1, alphmk))
            # Fk, Xk = roll_matrix(Fk, fk), roll_matrix(Xk, xk)
            # ------------------------------------------------------------------
            # version (b) - unconstrained minimization
            Fkp1, Xkp1 = roll_matrix(Fk, fk), roll_matrix(Xk, xk)
            if itr0 >= 0:
                mk = min(itr0 + 1, slvr_opts.Anderson_depth)
                DFk = (Fkp1 - Fk)[:, -mk:]  # to start earlier for k < m
                # Anderson Type I
                # gams = np.dot(np.linalg.inv(np.dot(DXk.T, DFk)
                #              + 0.05 * np.eye(mk)  # regularization
                #              ), np.dot(DXk.T, fk))
                # Anderson Type II
                gams = np.dot(np.linalg.inv(np.dot(DFk.T, DFk)
                              # + 1e-13 * np.eye(mk)  # regularization
                              ), np.dot(DFk.T, fk))
                DXk = (Xkp1 - Xk)[:, -mk:]  # to start earlier for k < m
                # Walker2011 - no relaxation
                # flx -= np.dot(DXk + DFk, gams).reshape(flx.shape)
                # Henderson2019 - betak is the relaxation parameter
                # (no significant improvement noticed)                
                flx = betak * (flx - np.dot(DXk + DFk, gams).reshape(flx.shape))
                flx += (1 - betak) * (xk - np.dot(DXk, gams)).reshape(flx.shape)
                # print('gammas', gams)
            Xk, Fk = Xkp1, Fkp1
            # input(flx[0, -6:])  # debug
        
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


def unfold_xs(input_data, diff_calc=True):
    """Set up the spatial mesh with cross sections data. Scattering xs ss can
    be limited to the only isotropic term for diffusion calculations."""
    I, G, Lss = input_data.I, input_data.G, input_data.Lss
    Lssp1 = Lss + 1
    xs_media, media = input_data.xs_media, input_data.media
    xm = xim(input_data.xi)
    ss, st = np.zeros((G, G, Lssp1, I),), np.zeros((G, I),)
    nsf, chi, D = np.zeros_like(st), np.zeros_like(st), np.zeros_like(st)

    lbnd = 0.
    for m in media:
        media_name, rbnd = m
        idx = (lbnd < xm) & (xm < rbnd)
        n = sum(idx)  # nb. of True values or non-zero elements
        st[:, idx] = np.tile(xs_media[media_name]['st'], (n, 1)).T
        nsf[:, idx] = np.tile(xs_media[media_name]['nsf'], (n, 1)).T
        chi[:, idx] = np.tile(xs_media[media_name]['chi'], (n, 1)).T
        D[:, idx] = np.tile(xs_media[media_name]['D'], (n, 1)).T
        Lmed = xs_media[media_name]['ss'].shape[-1]
        if Lmed > Lssp1:
            raise ValueError("Media %s has ss with L > %d!" %
                             (media_name, Lss))
        tmp = np.tile(xs_media[media_name]['ss'].flatten(), (n, 1)).T
        ss[:, :, :, idx] = tmp.reshape(G, G, Lssp1, n)
        lbnd = rbnd

    # determine the diffusion coefficient on the cell borders (w/o boundaries)
    Db = vol_averaged_at_interface(D, input_data.Vi)
    # Db at the boundaries is considered equal to the one of the nearest cell
    Db = np.insert(Db, 0, D[:, 0], axis=1)
    Db = np.insert(Db, I, D[:,-1], axis=1)
#    # test odCMFD
#    Vm = np.insert(xim(input_data.Vi), 0, input_data.Vi[0])
#    Vm = np.insert(Vm, -1, input_data.Vi[-1])
#    tau_max = np.max(input_data.Vi * st)
#    opt_theta_by_Zhu = opt_theta(tau_max)
#    # NOTE: our max_tau values are much smaller than what is used in MOC 
#    # transport calculations and so opt_theta is almost always zero.
#    lg.info("Optimal theta for odCMFD = %.3f with max(tau) = %.5f" % \
#        (opt_theta_by_Zhu, tau_max))
#    Db += opt_theta_by_Zhu * np.tile(Vm, (G, 1))
    xs = [st, ss, chi, nsf]
    if diff_calc:
        xs.append(Db)
    return xs


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
