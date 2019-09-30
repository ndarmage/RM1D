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

The Cartesian (slab) is the default geometry option. Curvilinears can be
chosen by selecting 'cylindrical' or 'spherical' for the global variable
geometry_type.

.. todo:: the current version assumes a homogeneous problem. To be
          extended to heterogeneous problems.
"""
import os
import sys
import logging as lg
import numpy as np
from scipy.special import expn as En

__title__ = "Multigroup diffusion in 1D slab by finite differences"
__author__ = "D. Tomatis"
__date__ = "02/04/2019"
__version__ = "1.1.0"

max_float = np.finfo(float).max  # float is float64!
min_float = np.finfo(float).min
np.set_printoptions(precision=5)

# possible options of geometry_type are slab, cylindrical, spherical
geometry_type = "slab"
fix_reflection_by_flx2 = True


def get_zeta(bc=0, D=None):
    "Yield the zeta boundary condition according to the input bc code"
    G = D.size
    if bc == 0:
        # vacuum
        zeta = 2.13 * D
    elif bc == 1:
        # zero flux
        zeta = np.full(G, 0.)
    elif bc == 2:
        # reflection (i.e. zero current)
        zeta = np.full(G, max_float)
    else:
        raise ValueError("Unknown type of b.c.")
    return zeta


def solveTDMA(a, b, c, d):
  """Solve the tridiagonal matrix system equations by the
  Thomas algorithm. a, b and c are the lower, central and upper
  diagonal of the matrix to invert, while d is the source term."""
  n = len(d)
  cp, dp = np.ones(n - 1), np.ones(n)
  x = np.ones_like(dp) # the solution

  cp[0] = c[0] / b[0]
  for i in range(1,n-1):
      cp[i] = c[i] / (b[i] - a[i-1] * cp[i-1])

  dp[0] = d[0] / b[0]
  dp[1:] = b[1:] - a * cp
  for i in range(1,n):
      dp[i] = (d[i] - a[i-1] * dp[i-1]) / dp[i]

  x[-1] = dp[-1]
  for i in range(n-2, -1, -1):
      x[i] = dp[i] - cp[i] * x[i+1]
  return x


def compute_fission_source(chi, nsf, flx):
    # the input flux must be volume-integrated
    return np.outer(chi, np.dot(nsf, flx)).flatten()


def compute_source(ss0, chi, nsf, flx, k=1.):
    # the input flux must be volume-integrated
    qs = np.dot(ss0, flx).flatten()
    return (qs + compute_fission_source(chi, nsf, flx) / k)


def compute_cell_volumes(xi, geo=geometry_type):
    # These volumes are per unit of transversal surface of the slab or
    # per unit of angle in the other frame (azimuthal angle in the cylinder
    # or per cone unit in the sphere). The real volumes in the cylinder
    # must be multiplied by 2*np.pi, 4*np.pi for the sphere.
    Di = xi[1:] - xi[:-1]
    if geo != 'slab':
        xm = (xi[1:] + xi[:-1]) / 2.
        if geo == 'cylindrical':
            Di *= xm
        elif geo == 'spherical':
            Di *= (4. * xm**2 - xi[1:] * xi[:-1]) / 3.
        else:
            raise ValueError("Unknown geometry type " + geo)
    return Di


def vol_averaged_at_interface(D, Vi):
    "Compute surface quantities by volume averaging (slab geometry)."
    G, I = D.shape
    Im1 = I - 1

    # Db, diff. coeff. on cell borders
    Db = (D[:,1: ] * np.tile(Vi[1: ], G).reshape(G, Im1) +
          D[:,:-1] * np.tile(Vi[:-1], G).reshape(G, Im1))
    Db /= np.tile(Vi[1:] + Vi[:-1], G).reshape(G, Im1)
    # Db = np.zeros((G, I-1),)
    # for g in range(G):
    #     Db[g,:] = (D[g,1:] * Vi[1:] + D[g,:-1] * Vi[:-1]) / Dm
    return Db


def set_diagonals(st, D, xv, BC=(0, 0), dD=None):
    "Build the three main diagonals of the solving system equations."
    LBC, RBC = BC
    xi, Di, Vi, geo = xv  # cell bounds and reduced volumes
    G, I = D.shape
    GI = G * I
    a, b, c = np.zeros(GI-1), np.zeros(GI), np.zeros(GI-1)

    # take into account the delta-diffusion-coefficients
    if dD is None:
        dD = np.zeros((G, I+1),)
    elif dD.shape != (G, I+1):
        raise ValueError('Unexpected shape for the delta D coefficients.')

    # determine the diffusion coefficient on the cell borders
    Db = vol_averaged_at_interface(D, Vi)

    # compute the coupling coefficients by finite differences
    iDm = 2. / (Di[1:] + Di[:-1])
    # iDm = 2. / (Vi[2:] + Vi[:-2])
    xb0, xba = 1., 1.
    if geo == 'cylindrical':
        iDm *= xi[1:-1]
        dD *= xi
        xb0, xba = xi[0], xi[-1]
    if geo == 'spherical':
        iDm *= xi[1:-1]**2
        dD *= xi**2
        xb0, xba = xi[0]**2, xi[-1]**2

    d = Db.flatten() * np.tile(iDm, G)
    # print('coeffs from diffusion')
    # print (d.reshape(G,I-1))
    # print(dD)
    for g in range(G):
        idx = np.arange(g*I, (g+1)*I-1)
        coefp = d[idx-g] + dD[g,1:-1]
        coefm = d[idx-g] - dD[g,1:-1]
        c[idx] = -coefp / Vi[1:]
        b[idx] = coefm
        b[idx+1] += coefp
        a[idx] = -coefm / Vi[:-1]
        # add b.c.
        zetal, zetar = get_zeta(LBC, D[g, 0]), get_zeta(RBC, D[g, -1])
        # # force the extrapolated lengths estimated by the reference solution
        # zetar = np.array([0.67291, 0.33227])[g]
        b[g*I] += D[g, 0] * xb0 / (0.5 * Di[ 0] + zetal) + dD[g, 0]
        b[(g+1)*I-1] += D[g, -1] * xba / (0.5 * Di[-1] + zetar) - dD[g, -1]
        # N.B.: the division by Di are needed because we solve for the volume
        # integrated flux
        idx = np.arange(g*I, (g+1)*I)
        b[idx] /= Vi
        b[idx] += np.full(I, st[g])
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


def compute_tran_currents(flx, k, Di, xs, BC=(0, 0), L=1):
    """Compute the currents given by the integral transport equation with the
    input flux flx. The flux at the boundary used in case of reflection."""
    # daniele: Di is probably to be replaced by Vi in curvilinear geoms.
    if geometry_type != 'slab':
        raise RuntimeError('This function has not been ported to curv. geoms.')
    LBC, RBC = BC
    st, ss0, chi, nsf, D = xs
    G, I = st.size, Di.size
    J = np.zeros((G,I+1),)  # currents at the cell bounds
    q = compute_source(ss0, chi, nsf, flx, k).reshape(G, I)
    # divide the volume-integrated source by the cell volumes if the input
    # flux is volume-integrated
    # q /= np.tile(Di, G).reshape(G, I)
    # The current implementation does not support yet any flux anisotropy, so
    # q is fixed as in the following.
    q = np.expand_dims(q, axis=0)

    # compute the total removal probability per cell
    stx = np.tile(st, (I, 1)).T
    Pt = np.tile(Di, G).reshape(G, I) * stx

    for g in range(G):
        for l in range(L):
            # We use here a matrix to store the transfer probabiities, although
            # only the elements on one colums should be stored by the recipro-
            # city theorem.
            Ajig = np.zeros((I,I+1),)
            # (2 * l + 1.) / 2. = l + 0.5
            qgoStg = (l + 0.5) * q[l,g,:] / stx[g,:]
            for i in range(I+1):
                for j in range(I):
                    Ajig[j,i] = (En(l+3, opl(j+1, i-1, Pt[g,:])) \
                               - En(l+3, opl( j , i-1, Pt[g,:]))) \
                                if j < i else \
                                (En(l+3, opl(i,  j , Pt[g,:])) \
                               - En(l+3, opl(i, j-1, Pt[g,:]))) * (-1)**l
            J[g,:] += np.dot(qgoStg, Ajig)

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
            bflx = quadratic_extrapolation(flx[g,:3], Di[:3])
            # bflx = flx[g,0]  # accurate only to 1st order
            # print ('L',g,bflx, flx[g,:3])
            # get the 'corrective' 2-nd moment
            # bflx2_5o8 = -J[g, 0] - 0.25 * bflx  ## it may be negative!
            # ...commented for considering also the contributions from the
            #    right below
            trL = np.array([En(3, opl(0, i-1, Pt[g,:])) for i in range(I+1)])
            # J[g,:] += 0.5 * np.tile(flx[g,0] / Di[0], I+1) * trL
            J[g,:] += 0.5 * bflx * trL
        if RBC == 2:
            bflx = quadratic_extrapolation(flx[g,-3:][::-1], Di[-3:][::-1])
            # bflx = flx[g,-1]  # accurate only to 1st order
            # print ('R',g, bflx, flx[g,-3:], flx[g,-1])
            trR = np.array([En(3, opl(i, I-1, Pt[g,:])) for i in range(I+1)])
            # J[g,:] -= 0.5 * np.tile(flx[g,-1] / Di[-1], I+1) * trR
            J[g,:] -= 0.5 * bflx * trR

        # Fix the non-vanishing current at the boundary by propagating the
        # error through the second moments. This is done after transmitting
        # the terms on the 0-th moment to account for possible contributions
        # coming from the opposite boundary. The second moment is already
        # multiplied by 5/8, and it will be multiplied by 5/2 in the equation
        # for the current (and so see below 8 / 5 * 5 / 2 = 4). This 2nd moment
        # may be negative (and unphysical), but we use it only as a numerical
        # correction.
        if LBC == 2 and fix_reflection_by_flx2:
            bflx2_5o16 = -J[g, 0]
            trL = np.array([En(5, opl(0, i-1, Pt[g,:])) for i in range(I+1)])*3
            trL -= np.array([En(3, opl(0, i-1, Pt[g,:])) for i in range(I+1)])
            J[g,:] += 4. * bflx2_5o16 * trL
        if RBC == 2 and fix_reflection_by_flx2:
            bflx2_5o16 = J[g, -1]
            trR = np.array([En(5, opl(i, I-1, Pt[g,:])) for i in range(I+1)])*3
            trR -= np.array([En(3, opl(i, I-1, Pt[g,:])) for i in range(I+1)])
            J[g,:] -= 4. * bflx2_5o16 * trR

    return J


def compute_diff_currents(flx, D, xv, BC=(0, 0)):
    """Compute the currents by Fick's law using the volume-averaged input
    diffusion cofficients."""
    LBC, RBC = BC
    xi, Di, Vi, geo = xv
    G, I = D.shape
    # Db, diff. coeff. on cell borders
    Db = vol_averaged_at_interface(D, Vi)
    J = -2. * Db * (flx[:,1:] - flx[:,:-1])
    J /= np.tile(Di[1:] + Di[:-1], G).reshape(G, I-1)
    # add b.c.
    zetal, zetar = get_zeta(LBC, D[:, 0]), get_zeta(RBC, D[:, -1])
    JL = -D[:, 0] * flx[:, 0] / (0.5 * Di[ 0] + zetal)
    JR =  D[:,-1] * flx[:,-1] / (0.5 * Di[-1] + zetar)
    # avoid possible numerical issues
    if LBC == 2: JL = 0
    if RBC == 2: JR = 0
    J = np.insert(J, 0, JL, axis=1)
    J = np.insert(J, I, JR, axis=1)
    return J


def compute_delta_diff_currents(flx, dD, Di, BC=(0, 0)):
    """Compute the correction currents by delta diffusion coefficients dD."""
    LBC, RBC = BC
    G, I = dD.shape
    I -= 1
    J = -2. * dD[:, 1:-1] * (flx[:,1:] + flx[:,:-1])
    J /= np.tile(Di[1:] + Di[:-1], G).reshape(G, I-1)
    # add b.c.
    zetal, zetar = get_zeta(LBC, D[:, 0]), get_zeta(RBC, D[:, -1])
    J = np.insert(J, 0, -dD[:, 0] * flx[:, 0], axis=1)
    J = np.insert(J, I, +dD[:,-1] * flx[:,-1], axis=1)
    return J


def solve_inners(flx, ss0, diags, sok, toll=1.e-5, iitmax=10):
    "Solve inner iterations on scattering."
    a, b, c = diags
    G, I = flx.shape
    irr, iti = 1.e+20, 0
    while (irr > toll) and (iti < iitmax):
        # backup local unknowns
        flxold = np.array(flx, copy=True)
        src = sok + np.dot(ss0, flx).flatten()
        flx = solveTDMA(a, b, c, src).reshape(G, I)
        ferr = np.where(flx > 0., 1. - flxold / flx, flx - flxold)
        irr = abs(ferr).max()
        iti += 1
        lg.debug(" +-> it={:^4d}, k={:<13.6g}, err={:<+13.6e}".format(
           iti, k, irr
        ))
    return flx


def compute_delta_D(flx, dJ, vrbs=False):
    """Compute the delta diffusion coefficients (already divided by the cell
    width, plus the possible extrapolated length); the input dJ is - (J_tran
    - J_diff)."""
    dD = dJ
    if vrbs:
        lg.debug('currents...')
        lg.debug('diff: ', J_diff)
        lg.debug('tran: ', J_tran)
    dD[:, 1:-1] /= (flx[:,1:] + flx[:,:-1])
    if np.all(flx[:, 0] > 0.): dD[:, 0] /= flx[:, 0]
    if np.all(flx[:,-1] > 0.): dD[:,-1] /= flx[:,-1]
    return dD


def solve_outers(flx, k, xv, xs, BC=(0, 0), itsmax=(50, 50),
                 toll=(1.e-6, 1.e-6), dD=None):
    "Solve the outer iterations by the power method."
    # unpack objects
    st, ss0, chi, nsf, D = xs
    G, I = flx.shape
    LBC, RBC = BC
    oitmax, iitmax = itsmax # max nb of outer/inner iterations
    otoll, itoll = toll # rel. tolerance for the solution in outers/inners

    # setup the tri-diagonal matrix and the source s
    diags = set_diagonals(st, D, xv, BC, dD)

    # start outer iterations
    err, ito = 1.e+20, 0
    # evaluate the initial source
    s = compute_fission_source(chi, nsf, flx)
    while (err > otoll) and (ito < oitmax):
        # backup local unknowns
        serr, kold = np.array(s, copy=True), k
        sold = serr # ...just a reference

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


def to_str(v, fmt='%.13g'):
    return ', '.join([fmt%i for i in v]) + '\n'


def check_current_solutions():
    print("F_ref ", flxm_SN[0,0,-6:] / Di[-6:])
    print("F_diff", flx_save[:,:,itr][0,-6:] / Di[-6:])
    print("F_dif*", flx[0,-6:] / Di[-6:])
    print("J_ref ", flxm_SN[0,1,-6:] / Di[-6:])  # cell-averaged!
    print("J_diff", J_diff[0,-6:])
    print("J_tran", J_tran[0,-6:])
    print("J_corr", J_corr[0,-6:])
    print("    dD", dD[0,-6:])


if __name__ == "__main__":

    logfile = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    lg.basicConfig(level=lg.INFO)  # filename = logfile
    lg.info("* verbose output only in DEBUG logging mode *")

    lg.info("Verify the implementation with the following test case")
    from data2Ghomog import *
    lg.info("*** Homogeneous Test Case in " + geometry_type +
            " 1D geometry ***")

    if (geometry_type != 'slab') and (LBC != 2):
        raise ValueError('Curvilinear geometries need LBC = 2.')

    # get the cell width (where are volumes in cm in the 1D geometry)
    Di = xi[1:] - xi[:-1]
    Vi = compute_cell_volumes(xi, geo=geometry_type)
    xv = xi, Di, Vi, geometry_type  # pack mesh data

    # initialize the unknowns
    flx, k = np.ones((G, I),), 1.
    # tolerance on the fission rates during outer iterations
    toll = 1.e-6
    oitmax = 100  # max nb of outer iterations
    iitmax = 100  # max nb of inner iterations
    itsmax = oitmax, iitmax
    # pack the b.c. identifiers
    BC = LBC, RBC

    # the diffusion coefficient is here request as space-dependent even when
    # the problem is homogeneus
    D = np.tile(D, (I, 1)).T  # cell-averaged diff. coef.

    # pack the xs data in a single object
    xs = st, ss0, chi, nsf, D

    # Call the diffusion solver without CMFD corrections
    lg.info("-o"*22)
    lg.info("Initial call to the diffusion solver without CMFD corrections")
    lg.info("Start the diffusion solver (<- outer its. / -> inner its.)")
    flx, k = solve_outers(flx, k, xv, xs, BC, itsmax, (toll, toll))
    kinit = k
    lg.info("-o"*22)

    # start Ronen iterations
    ritmax = 200  # set to 1 to skip the Ronen iterations
    lg.info("Implement the Ronen Method by CMFD iterations")
    err, itr = 1.e+20, 0
    Dflxm1 = np.zeros_like(flx)

    # load reference currents
    ref_flx_file = "../SNMG1DSlab/LBC1RBC0_I%d_N%d.npz" % (I, 64)
    if os.path.isfile(ref_flx_file):
        ref_data = np.load(ref_flx_file)
        k_SN, flxm_SN = ref_data['k'], ref_data['flxm']
        # normalize the reference flux
        flxm_SN *= (I * G) / np.sum(flxm_SN[:,0,:])
        d_flxm0R = np.array([
            - (estimate_derivative(flxm_SN[g,0,-3:][::-1], Di[-3:][::-1])
            / flxm_SN[g,0,-1]) for g in range(G)
        ])
        print('Estimates of the extrapolated lengths ' + str(-1. / d_flxm0R))
        print('Values used in the diffusion solver ' + str(2.13*D[:,-1]))

    # keep track of partial solutions on external files
    flx_save = np.empty(list(flx.shape) + [ritmax + 1])
    k_save = np.full(ritmax + 1, -1.)
    while (err > toll) and (itr < ritmax):
        k_save[itr], flx_save[:,:,itr] = k, flx

        # revert to flux density
        # (this division does not seem to affect the final result though)
        flxd = flx / Vi  # division on last index
        # compute the currents by diffusion and finite differences
        J_diff = compute_diff_currents(flxd, D, xv, BC)
        # compute the currents by integral transport (Ronen Method)
        # #lg.warning("CALCULATE THE TR CURRENT WITH THE REFERENCE S16 SOLUTION!")
        ## J_tran = compute_tran_currents(flxm_SN[:,0,:] / Vi, k_SN, Di, xs, BC)
        J_tran = compute_tran_currents(flxd, k, Di, xs, BC)
        # compute the corrective delta-diffusion-coefficients
        dD = compute_delta_D(flxd, J_diff - J_tran)

        # compute the corrective currents (determined only for debug)
        J_corr = compute_delta_diff_currents(flxd, dD, Di, BC)

        flxold, kold = np.array(flx, copy=True), k
        lg.info("Start the diffusion solver (<- outer its. / -> inner its.)")
        flx, k = solve_outers(flx, k, xv, xs, BC, itsmax, (toll, toll), dD)
        # check_current_solutions()

        # try Aitken delta squared extrapolation
        # if (itr > 10) and (err < toll * 100):
        #     lg.info("<-- Apply Aitken extrapolation on the flux -->")
        #     flx -=  (Dflxm1**2 / (Dflxm1 - Dflxm2))

        # evaluate the flux difference through the iterations
        Dflxm2 = np.array(Dflxm1, copy=True)
        Dflxm1 = flx - flxold
        ferr = np.where(flx > 0., 1. - flxold / flx, Dflxm1)

        err = abs(ferr[ np.unravel_index(abs(ferr).argmax(), (G, I)) ])
        itr += 1
        lg.info("+RM+ it={:^4d}, k={:<13.6g}, err={:<+13.6e}".format(
            itr, k, err
        ))
        lg.info("{:^4s}{:^13s}{:^6s}{:^13s}".format(
            "G", "max(err)", "at i", "std(err)"
        ))
        for g in range(G):
            ierr, estd = abs(ferr[g,:]).argmax(), abs(ferr[g,:]).std()
            lg.info(
                "{:^4d}{:<+13.6e}{:^6d}{:<+13.6e}".format(
                g, ferr[g,ierr], ierr+1, estd
            ))
        input('press a key to continue...')
    k_save[itr], flx_save[:,:,itr] = k, flx  # store final values
    lg.info("Initial value of k was %13.6g." % kinit)
    ofile = "tests/kflx_LBC%dRBC%d_I%d" % (*BC,I)
    np.save(ofile + ".npy", [k_save, flx_save], allow_pickle=True)
    with open(ofile + ".dat", 'w') as f:
        for i, ki in enumerate(k_save):
            if ki < 0: break
            f.write(to_str(np.append(ki, flx_save[:,:,i].flatten())))
