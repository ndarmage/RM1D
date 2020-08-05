#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
Test cases by Sood et al. [Sood2003]_.

It has been noticed that finer meshes are needed to attain the same level of
accuracy, being in order, in the slab, in the cylinder and in the sphere.

.. [Sood2003] Sood, A., Forster, R. A., & Parsons, D. K. (2003). Analytical
              benchmark test set for criticality code verification. Progress
              in Nuclear Energy, 42(1), 55-106.
"""
import sys, os
import numpy as np
from scipy import interpolate
from scipy.optimize import brentq
from scipy.special import jv as Jv  # Bessel function of the first kind
# from scipy.special import sici  # sine and cosine integrals
# from scipy.special import struve  # Struve function
J0, J1 = lambda z: Jv(0, z), lambda z: Jv(1, z)

sys.path.append(os.path.join(os.getcwd(), ".."))
from data.SoodPNE2003 import *
from FDsDiff1D import input_data, solver_options, run_calc_with_RM_its
from GeoMatTools import geomprogr_mesh, equivolume_mesh

odir = "output"
refdir = os.path.join("..", "..", "CPM1D", "tests", "output")


# Table 10
# Critical Dimensions, rc, for One-Group Bare U-235 (c=1.30)
# Problem Identifier  Geometry  rc (mfp)      rc (cm)     Reference
# 12 Ua-1-0-SL        Slab      0.93772556    2.872934    [35]
# 13 Ua-1-0-CY        Cylinder  1.72500292    5.284935    [36],[37]
# 14 Ua-1-0-SP        Sphere    2.4248249802  7.428998    [35]

# Table 11
# Normalized Scalar Fluxes for One-Group Bare U-235 (c=1.30)
# Problem Identifier Geometry r/rc = 0.25, r/rc = 0.5, r/rc = 0.75, r/rc = 1
# 12 Ua-1-0-SL       Slab     0.9669506  0.8686259  0.7055218  0.4461912
# 13 Ua-1-0-CY       Cylinder     -      -          -          -
# 14 Ua-1-0-SP       Sphere   0.93244907 0.74553332 0.48095413 0.17177706

refflx = {
    'Ua-1-0-SL': np.array([0.9669506, 0.8686259, 0.7055218, 0.4461912]),
    'Ua-1-0-CY': np.array([np.nan, np.nan, np.nan, np.nan]),
    'Ua-1-0-SP': np.array([0.93244907, 0.74553332, 0.48095413, 0.17177706])
}


if __name__ == "__main__":

    import logging as lg
    lg.info("***          Sood's test suite          ***")
    lg.info("*** 1G homogeneous isotropic scattering ***")
    lg.info("***              Ux-1-0-x               ***")
    
    # positions (fractions r/rc) to verify the flux results
    rf = np.linspace(0, 1, 5)
    flx_tolerance = np.ones(4)  # tolerances are in percent
    flx_tolerance[-1] = 2  # RM cannot fit transport at bare boundary
    
    # -------------------------------------------------------------------------- 
    m = "Ua"
    BM2m, extrap_len = BM2(materials[m]), 2.13 * materials[m]['D'][0]
    lg.info("Material buckling %.5f" % BM2m)
    
    # [3, 4, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150]
    for I0 in [50]:

        for geo in geoms:
            case = "%s-1-0-%s" % (m, get_geoid(geo))
            lg.info("Test case: " + case)
            L = Lc[case] # rc_dict[geo]
            rs = rf * L
            # L_e = L + extrap_len
            if geo == 'slab':
                I, nks, LBC = I0*2, 0, 0  # 60
                BG = brentq(lambda b: np.tan(b*L) - 1 / extrap_len / b,
                    1.e-5, .5 * np.pi / L - 1.e-5)
                rs += L
                L *= 2
                coef = 1 / np.tan(BG * L / 2)  # for symmetric distribution 
                diffsol_ref = lambda x: np.sin(BG * x) + coef * np.cos(BG * x)
                # diffsol_ref = lambda x: np.cos(BG * (x - L/2))
                anorm = (1 - np.cos(BG * L) + coef * np.sin(BG * L)) / BG
                Dktol, Dftol = 5e-4, 1e-3  # tolerance for k and flx (diffusion)
            else:
                nks, LBC = 4, 2
                if 'cylind' in geo:
                    I = I0 #30
                    BG = brentq(lambda b: J0(b*L) - extrap_len * b * J1(b*L),
                        0, 2.4048255577 / L)
                    diffsol_ref = lambda x: J0(BG * x)
                    # primitive of indefinite \int{J_0(z)dz}
                    # primitive = lambda z: 0.5 * z * (
                        # np.pi * J1(z) * struve(0, z)
                        # + J0(z) * (2 - np.pi * struve(1, z)))
                    # anorm = primitive(BG*L) / BG  # ... primitive(0) = 0!
                    # integration of (r J0(BG*r) dr) between 0 and L:
                    anorm = L / BG * J1(BG*L)
                    Dktol, Dftol = 5e-4, 5e-3
                else:
                    I = I0
                    flx_tolerance *= 2
                    f= lambda b: b*L - (1 - L/extrap_len)*np.tan(b*L)
                    # info: add a small epsilon to the left bound to avoid
                    #       failure of root-finding algorithm
                    BG = brentq(lambda b: b*L - (1 - L/extrap_len)*np.tan(b*L),
                        .50000001 * np.pi / L, np.pi / L)
                    diffsol_ref = lambda x: np.sin(BG * x) / x
                    # integration of sin(x)/x yields 'sine integral func'
                    # anorm, _ = sici(BG * L)
                    # integration of (sin(BG*r)/r) r**2 dr yields
                    anorm = (np.sin(BG*L) - BG*L * np.cos(BG*L)) / BG**2
                    Dktol, Dftol = 2e-3, 2e-2
            # r = geomprogr_mesh(N=I, L=L, ratio=0.95)
            r = equivolume_mesh(I, 0, L, geo)
            
            lg.info('Reference critical length (L) is %.6f cm' % L)
            lg.info('Extrapolation distance (zeta*D) is %.3f cm' % extrap_len)
            xs_media, media = set_media(materials[m], L, m)
            data = input_data(xs_media, media, r, geo, LBC=LBC, RBC=0,
                              per_unit_angle=True)
            
            # *** WARNING ***
            # The extrapolation length is a quite large w.r.t. to the problem
            # width in these problems. Therefore, the numerical solution can be
            # very different from the analytical one (still an extrapolation
            # length is considered).
            lg.info(' -o-'*15)
            lg.info('analytical solution of the diffusion equation')
            ansol, DFkref = diffsol_ref(data.xim), \
                            diffk_ref(BG**2, materials[m])
            # lg.info('fund. flx\n' +
            #         str(ansol / np.sum(ansol * data.Vi) * G * I))
            lg.info('kinf = {:.6}, k_DIFF = {:.6f}, BG2 = {:.6f}'.format(
                materials[m]['kinf'], diffk_ref(BG**2, materials[m]), BG**2))
            lg.info(' -o-'*15)
            # diffusion problem
            slvr_opts = solver_options(iitmax=5, oitmax=5, ritmax=0,
                                       ks=np.full(I, nks))
            filename = os.path.join(odir, case + "_LBC%dRBC%d_I%d_diff" %
                                                              (LBC, 0, I))
            flx, k = run_calc_with_RM_its(data, slvr_opts, filename)
            flx *= anorm / np.sum(flx) / data.Vi
            np.save(filename.replace('_diff', '_andf') + '.npy',
                (DFkref, BG, (1 - flx[0, :] / ansol)*100))
            np.testing.assert_allclose(k, DFkref, atol=Dktol, err_msg=case +
                ": criticality against diffusion analytically solved not " +
                "verified")
            np.testing.assert_allclose(flx[0, :], ansol, rtol=Dftol,
                err_msg=case + ": numerical vs analytical (diffusion) fund." +
                               " flux not verified")
            
            # transport problem
            slvr_opts = solver_options(iitmax=5, oitmax=5, ritmax=300,
                                       CMFD=True, pCMFD=False, 
                                       Anderson_depth='auto',
                                       ks=np.full(I, nks))
            filename = os.path.join(odir, case + "_LBC%dRBC%d_I%d" %
                                                         (LBC, 0, I))
            flx, k = run_calc_with_RM_its(data, slvr_opts, filename)
            # get the values of the flux at the positions requested by the
            # benchmark by piecewise linear interpolation
            # flx /= np.dot(data.Vi, flx[0,:])
            f = interpolate.interp1d(data.xim, flx[0,:] / data.Vi,
                                     fill_value="extrapolate")
            flxs = f(rs)  # rs[0] is at the center
            # normalize the flux to be unitary at the center
            flxs /= flxs[0]
            lg.info(' -o-'*15)
            print('Comparison of flux distribution -- case ' + case)
            print('    Calc. flx:' +
                  ' '.join(['{:9.7f}'.format(v) for v in flxs]))
            print('     Ref. flx:' + ' '.join(['*'*9] +
                ['{:9.7f}'.format(v) for v in refflx[case]]))
            rel_err_pc = (1 - flxs[1:] / refflx[case]) * 100
            np.save(filename + '_RMre.npy', rel_err_pc)
            print('Rel. Err. (%):' + ' '.join(['*'*9] +
                ['{:+9.6f}'.format(v) for v in rel_err_pc]))
            np.testing.assert_array_less(np.nan_to_num(abs(rel_err_pc)),
                                         flx_tolerance, err_msg=case +
                                         ": flux distribution not verified")
            np.testing.assert_allclose(k, 1, atol=1.e-3, err_msg=case +
                                       ": criticality not verified")
            lg.info('*'*77)
            # input('all checks successful')