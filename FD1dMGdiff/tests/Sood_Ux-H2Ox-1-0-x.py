#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
Test cases by Sood et al. [Sood2003]_.

1G heterogeneous cases with PUa and PUb, with water as reflector, in the
slab and in the cylinder. Criticality test, reference flux is not
communicated.

.. [Sood2003] Sood, A., Forster, R. A., & Parsons, D. K. (2003). Analytical
              benchmark test set for criticality code verification. Progress
              in Nuclear Energy, 42(1), 55-106.
"""
import sys, os
import numpy as np

sys.path.append(os.path.join(os.getcwd(), ".."))
from data.SoodPNE2003 import *
from FDsDiff1D import input_data, solver_options, run_calc_with_RM_its
from GeoMatTools import geomprogr_mesh, equivolume_mesh

odir = "output"
refdir = os.path.join("..", "..", "CPM1D", "tests", "output")


# Table 12
# Critical Dimensions for One-Group U-235 Sphere
# with H20 Reflector (c=0.90)
# Problem Identifier Geometry Pu-rc H2O-thickness Pu+H2O-Radius Ref.
# 16 Ub-H2O(l)-l-O-SP Sphere (mfp) 2 1 [24], [27]
#                             (cm) 6.12745 3.063725 9.191176
# 18 Uc-H20(2)-1.O-SP  Sphere (mfp) 2 2 [24],[27]
#                             (cm) 6.12745 6.12745 12.2549
# 20 Ud-H2O(3)-l-O-SP Sphere (mfp) 2 3 [24], [27]
#                             (cm) 6.12745 9.191176 15.318626

if __name__ == "__main__":

    import logging as lg
    lg.info("***           Sood's test suite           ***")
    lg.info("*** 1G heterogeneous isotropic scattering ***")
    lg.info("***             Ux-H2Ox-1-0-x             ***")
    
    # --------------------------------------------------------------------------

    # Problem 16 
    m, geo, nks = 'Ub', 'sphere', 4
    case = "%s-H2O(1)-1-0-%s" % (m, get_geoid(geo))
    L0, L1, L = 6.12745, 3.063725, Lc[case]
    LBC, RBC = 2, 0
    I1 = I0 = 20  # to get error on k less than 10 pcm
    I = I1 + I0
    xs_media, media = set_media(materials, [L0, L], [m, 'H2O'])
    r = equivolume_mesh(I0, 0, L0, geo)
    r = np.append(r, equivolume_mesh(I1, L0, L, geo)[1:])
    
    data = input_data(xs_media, media, r, geo, LBC=LBC, RBC=RBC,
                      per_unit_angle=True)
    slvr_opts = solver_options(iitmax=5, oitmax=5, ritmax=200, CMFD=True,
                               pCMFD=False, Anderson_depth='auto',
                               ks=np.full(I, nks))
    filename = os.path.join(odir, case + "_LBC%dRBC%d_I%d" %
                            (LBC, RBC, I))
    flx, k = run_calc_with_RM_its(data, slvr_opts, filename)
    np.testing.assert_allclose(k, 1.0, atol=1.e-4, err_msg=case +
                               ": criticality not verified")
    
    # Problem 18 
    m = 'Uc'
    case = "%s-H2O(2)-1-0-%s" % (m, get_geoid(geo))
    L0, L1, L = 6.12745, 6.12745, Lc[case]
    I1 = I0 = 15  # to get error on k less than 10 pcm
    I = I1 + I0
    xs_media, media = set_media(materials, [L0, L], [m, 'H2O'])
    r = equivolume_mesh(I0, 0, L0, geo)
    r = np.append(r, equivolume_mesh(I1, L0, L, geo)[1:])
    
    data = input_data(xs_media, media, r, geo, LBC=LBC, RBC=RBC,
                      per_unit_angle=True)
    slvr_opts = solver_options(iitmax=5, oitmax=5, ritmax=200, CMFD=True,
                               pCMFD=False, Anderson_depth='auto',
                               ks=np.full(I, nks))
    filename = os.path.join(odir, case + "_LBC%dRBC%d_I%d" %
                            (LBC, RBC, I))
    flx, k = run_calc_with_RM_its(data, slvr_opts, filename)
    np.testing.assert_allclose(k, 1.0, atol=1.e-4, err_msg=case +
                               ": criticality not verified")
    
    # Problem 20 
    m = 'Ud'
    case = "%s-H2O(3)-1-0-%s" % (m, get_geoid(geo))
    L0, L1, L = 6.12745, 9.191176, Lc[case]
    LBC, RBC = 2, 0
    I1 = I0 = 15  # to get error on k less than 10 pcm
    I = I1 + I0
    xs_media, media = set_media(materials, [L0, L], [m, 'H2O'])
    r = equivolume_mesh(I0, 0, L0, geo)
    r = np.append(r, equivolume_mesh(I1, L0, L, geo)[1:])
    
    data = input_data(xs_media, media, r, geo, LBC=LBC, RBC=RBC,
                      per_unit_angle=True)
    slvr_opts = solver_options(iitmax=5, oitmax=5, ritmax=200, CMFD=True,
                               pCMFD=False, Anderson_depth='auto',
                               ks=np.full(I, nks))
    filename = os.path.join(odir, case + "_LBC%dRBC%d_I%d" %
                            (LBC, RBC, I))
    flx, k = run_calc_with_RM_its(data, slvr_opts, filename)
    np.testing.assert_allclose(k, 1.0, atol=1.e-4, err_msg=case +
                               ": criticality not verified")