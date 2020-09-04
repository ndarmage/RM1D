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

odir = "../output"
refdir = os.path.join("..", "..", "CPM1D", "tests", "output")


# Table 6
# Critical Dimensions, rc, for One-Group Bare Pu-239 (c=1.50)
# with Non-Symmetric H20 Reflector (c=0.90)
# Problem Identifier Geometry Pu-rc H2O-thickness Pu+H2O-Radius Ref.
# 3 PUa-H2O(1)-l-0-SL Slab (mfp) 0.482566 1 - [16, [55
#                           (cm) 1.478450 3.063725 4.542175
# Table 7
# Critical Dimensions, rc, for One-Group Bare Pu-239 (c=1.50)
# with Non-Symmetric H20 Reflector (c=0.90)
# Problem Identifier Geometry Pu-rc H2O-thickness Pu+H2O-Radius Ref.
# 4 PUa-H2O(0.5)-l-0-SL Slab (mfp) 0.430150 0.5 - [16, [55
#                           (cm) 1.317862 1.531863 2.849725
# Table 8
# Critical Dimensions for One-Group Pu-239 Cylinder (c=1.40)
# with H20 Reflector (c=0.90)
# Problem Identifier Geometry Pu-rc H2O-thickness Pu+H2O-Radius Ref.
# 9 Pub-H2O(l)-l-0-CY Cylinder (mfp) 1.10898 1 [38
#                               (cm) 3.397610 3.063725 6.461335
# 10 PUb-H2O(10)-l-0.CY Cylinder (mfp) 1.00452 10 [38
#                               (cm) 3.077574 30.637255 33.714829

if __name__ == "__main__":

    import logging as lg
    lg.info("***           Sood's test suite           ***")
    lg.info("*** 1G heterogeneous isotropic scattering ***")
    lg.info("***            PUx-H2Ox-1-0-x             ***")
    
    # --------------------------------------------------------------------------
    
    # Problem 3
    m, geo = 'PUa', 'slab'
    LBC, RBC = 0, 0
    case = "%s-H2O(1)-1-0-%s" % (m, get_geoid(geo))
    L0, L1, L = 1.47840, 3.063725, Lc[case]
    I0 = 80  # nb. of cells in the fuel
    I1 = I0*2
    I = I0 + I1
    
    
    widths_of_buffers = [2*L0, 2*L0+L1]
    xs_media, media = set_media(materials,
        widths_of_buffers, [m, 'H2O'])
    r = equivolume_mesh(I0, 0, widths_of_buffers[0], geo)
    r = np.append(r, equivolume_mesh(I1, *widths_of_buffers, geo)[1:])
    
    data = input_data(xs_media, media, r, geo, LBC=LBC, RBC=RBC)
    slvr_opts = solver_options(iitmax=5, oitmax=5, ritmax=200, CMFD=True,
                                pCMFD=False, Anderson_depth='auto')
    filename = os.path.join(odir, case + "_LBC%dRBC%d_I%d" %
                            (LBC, RBC, I))
    flx, k = run_calc_with_RM_its(data, slvr_opts, filename)
    np.testing.assert_allclose(k, 1.0, atol=1.e-5, err_msg=case +
                                ": criticality not verified")
    
    # # Problem 4
    # I0 = 180  # to get 5 significative digits, i.e. error below 1 pcm.
    # case = "%s-H2O(0.5)-1-0-%s" % (m, get_geoid(geo))
    # L0, L1, L = 1.317862, 1.531863, Lc[case]
    # LBC = RBC = 0
    # I1 = I0
    # I = I1 + I0
    # widths_of_buffers = [L1, L + L0, 2*L]
    # xs_media, media = set_media(materials,
    #     widths_of_buffers, ['H2O', m, 'H2O'])
    # r = equivolume_mesh(I1, 0, widths_of_buffers[0], geo)
    # for i in range(2):
    #     Lb, Le = widths_of_buffers[i], widths_of_buffers[i+1]
    #     Ix = I0 if i % 2 == 0 else I1
    #     r = np.append(r, equivolume_mesh(Ix, Lb, Le, geo)[1:])
    
    # data = input_data(xs_media, media, r, geo, LBC=LBC, RBC=RBC)
    # slvr_opts = solver_options(iitmax=5, oitmax=5, ritmax=300, CMFD=True,
    #                            pCMFD=False, Anderson_depth='auto')
    # filename = os.path.join(odir, case + "_LBC%dRBC%d_I%d" %
    #                         (LBC, RBC, I))
    # flx, k = run_calc_with_RM_its(data, slvr_opts, filename)
    # np.testing.assert_allclose(k, 1.0, atol=1.e-4, err_msg=case +
    #                            ": criticality not verified")
    
    # # Problem 9
    # m, geo = 'PUb', 'cylinder'
    # case = "%s-H2O(1)-1-0-%s" % (m, get_geoid(geo))
    # L0, L1, L = 3.397610, 3.063725, Lc[case]
    # LBC, RBC = 2, 0
    # I1 = I0 = 50  # to get error on k less than 10 pcm
    # I = I1 + I0
    # nks = 4
    # xs_media, media = set_media(materials, [L0, L], [m, 'H2O'])
    # r = equivolume_mesh(I0, 0, L0, geo)
    # r = np.append(r, equivolume_mesh(I1, L0, L, geo)[1:])
    
    # data = input_data(xs_media, media, r, geo, LBC=LBC, RBC=RBC,
    #                   per_unit_angle=True)
    # slvr_opts = solver_options(iitmax=5, oitmax=5, ritmax=200, CMFD=True,
    #                            pCMFD=False, Anderson_depth='auto',
    #                            ks=np.full(I, nks))
    # filename = os.path.join(odir, case + "_LBC%dRBC%d_I%d" %
    #                         (LBC, RBC, I))
    # flx, k = run_calc_with_RM_its(data, slvr_opts, filename)
    # np.testing.assert_allclose(k, 1.0, atol=1.e-4, err_msg=case +
    #                            ": criticality not verified")
    
    # # Problem 10 ... a big number of cells might be necessary
    # case = "%s-H2O(10)-1-0-%s" % (m, get_geoid(geo))
    # L0, L1, L = 3.077574, 30.637255, Lc[case]
    # LBC, RBC = 2, 0
    # I1 = I0 = 30  # ? to get error on k less than 10 pcm
    # I = I1 + I0
    # nks = 4
    # xs_media, media = set_media(materials, [L0, L], [m, 'H2O'])
    # r = equivolume_mesh(I0, 0, L0, geo)
    # r = np.append(r, equivolume_mesh(I1, L0, L, geo)[1:])
    
    # data = input_data(xs_media, media, r, geo, LBC=LBC, RBC=RBC,
    #                   per_unit_angle=True)
    # slvr_opts = solver_options(iitmax=5, oitmax=5, ritmax=200, CMFD=True,
    #                            pCMFD=False, Anderson_depth='auto',
    #                            ks=np.full(I, nks))
    # filename = os.path.join(odir, case + "_LBC%dRBC%d_I%d" %
    #                         (LBC, RBC, I))
    # flx, k = run_calc_with_RM_its(data, slvr_opts, filename)
    # np.testing.assert_allclose(k, 1.0, atol=1.e-4, err_msg=case +
    #                            ": criticality not verified")