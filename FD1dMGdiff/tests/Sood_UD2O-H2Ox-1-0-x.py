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


# Table 16
# Critical Dimensions for One-Group U-D20 (c=1.02) Slab and Cylinder 
# with H2O 
# Problem Identifier Geometry UD2O rc H2O-thickness UD2O + H20 radius Ref.
# 25 UD2O-H2O(1)-1-O-SL Slab (mfp) 4.6041 10 - [42],[43]
#                           (cm) 9.214139 1.830563 11.044702
# 26 UD2O-H2O(10)-1-O-SL Slab (mfp) 5.0335 1 - [42],[43]
#                           (cm) 8.428096 18.30563 26.733726
# 27 UD2O-H2O(1)-1-O-CY Slab (mfp) 8.411027 1 - [38]
#                           (cm) 15.396916 1.830563 17.227479
# 28 UD2O-H2O(10)-1-O-CY Slab (mfp) 7.979325 10 - [38]
#                           (cm) 14.606658 18.30563 32.912288

def change_H2O(materials):
    h2o, ud2o = materials['H2O'], materials['UD2O']
    h2o['st'] = h2o['sc'] = ud2o['st']
    h2o['ss'] = np.array([0.491652])  # h2o['c'] * h2o['st']
    h2o['sa'] = h2o['st'] - h2o['ss']
    h2o['D'] = 1 / 3 / h2o['st']
    return materials

if __name__ == "__main__":

    import logging as lg
    lg.info("***           Sood's test suite           ***")
    lg.info("*** 1G heterogeneous isotropic scattering ***")
    lg.info("***            UD2O-H2Ox-1-0-x            ***")
    materials = change_H2O(materials)
    
    # --------------------------------------------------------------------------
    # Problem 25
    m, geo = 'UD2O', 'slab'
    case = "%s-H2O(1)-1-0-%s" % (m, get_geoid(geo))
    I0 = 180  # to get 5 significative digits, i.e. error below 1 pcm.
    L0, L1, L = 9.214139, 1.830563, Lc[case]
    LBC = RBC = 0
    I1 = I0
    I = I1 + I0
    widths_of_buffers = [L1, L + L0, 2*L]
    xs_media, media = set_media(materials,
        widths_of_buffers, ['H2O', m, 'H2O'])
    r = equivolume_mesh(I1, 0, widths_of_buffers[0], geo)
    for i in range(2):
        Lb, Le = widths_of_buffers[i], widths_of_buffers[i+1]
        Ix = I0 if i % 2 == 0 else I1
        r = np.append(r, equivolume_mesh(Ix, Lb, Le, geo)[1:])
    
    data = input_data(xs_media, media, r, geo, LBC=LBC, RBC=RBC)
    slvr_opts = solver_options(iitmax=5, oitmax=5, ritmax=200, CMFD=True,
                                pCMFD=False, Anderson_depth='auto')
    filename = os.path.join(odir, case + "_LBC%dRBC%d_I%d" %
                            (LBC, RBC, I))
    flx, k = run_calc_with_RM_its(data, slvr_opts, filename)
    np.testing.assert_allclose(k, 1.0, atol=1.e-4, err_msg=case +
                                ": criticality not verified")
    
    # Problem 26
    m, geo = 'UD2O', 'slab'
    case = "%s-H2O(10)-1-0-%s" % (m, get_geoid(geo))
    I0 = 180  # to get 5 significative digits, i.e. error below 1 pcm.
    L0, L1, L = 8.428096, 18.30563, Lc[case]
    LBC = RBC = 0
    I1 = I0
    I = I1 + I0
    widths_of_buffers = [L1, L + L0, 2*L]
    xs_media, media = set_media(materials,
        widths_of_buffers, ['H2O', m, 'H2O'])
    r = equivolume_mesh(I1, 0, widths_of_buffers[0], geo)
    for i in range(2):
        Lb, Le = widths_of_buffers[i], widths_of_buffers[i+1]
        Ix = I0 if i % 2 == 0 else I1
        r = np.append(r, equivolume_mesh(Ix, Lb, Le, geo)[1:])
    
    data = input_data(xs_media, media, r, geo, LBC=LBC, RBC=RBC)
    slvr_opts = solver_options(iitmax=5, oitmax=5, ritmax=200, CMFD=True,
                                pCMFD=False, Anderson_depth='auto')
    filename = os.path.join(odir, case + "_LBC%dRBC%d_I%d" %
                            (LBC, RBC, I))
    flx, k = run_calc_with_RM_its(data, slvr_opts, filename)
    np.testing.assert_allclose(k, 1.0, atol=1.e-4, err_msg=case +
                                ": criticality not verified")
    
    # Problem 27
    m, geo, nks = 'UD2O', 'cylinder', 4
    case = "%s-H2O(1)-1-0-%s" % (m, get_geoid(geo))
    L0, L1, L = 15.396916, 1.830563, Lc[case]
    LBC, RBC = 2, 0
    I1 = I0 = 50  # to get error on k less than 10 pcm
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
    
    # Problem 28
    m, geo = 'UD2O', 'cylinder'
    case = "%s-H2O(10)-1-0-%s" % (m, get_geoid(geo))
    L0, L1, L = 14.606658, 18.30563, Lc[case]
    LBC, RBC = 2, 0
    I1 = I0 = 30  # to get error on k less than 10 pcm
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