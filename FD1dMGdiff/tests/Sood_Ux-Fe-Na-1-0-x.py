#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
Test cases by Sood et al. [Sood2003]_.

1G heterogeneous cases with Ue, with Fe reflector and Na as moderator, in the
slab geometry. Criticality test, reference flux is not communicated.

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


# Table 18
# Critical Dimensions, rc, for One-Group U-235 Reactor 
# Problem Identifier Geometry    Fe thick.  U-235 thick. Fe thick.   Na thick.  Ref.
# 30 Ue-Fe-Na-l-0-SL  Slab (mfp) 0.0738      2.0858098   0.0738      0.173   - [55]
#                           (cm) 0.317337461 5.119720083 0.317337461 2.002771002

# Table 19
# Critical Dimensions, rc, for One-Group U-235 Reactor
# Problem Identifier Geometry   Fe thick.   Fe+U        Fe+U+Fe     Fe+U+Fe+Na
# 30 Ue-Fe-Na-l-0-SL Slab  (cm) 0.317337461 5.437057544 5.754395005 7.757166007

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
    lg.info("***             Ux-Fe-Na-1-0-x            ***")
    materials = change_H2O(materials)
    
    # --------------------------------------------------------------------------
    # Problem 30
    m, geo = 'Ue', 'slab'
    case = "%s-Fe-Na-1-0-%s" % (m, get_geoid(geo))
    L0, L1, L2, L = 0.317337461, 5.437057544, 5.754395005, Lc[case]
    LBC = RBC = 0
    I0 = 65 # within clad - Fe
    I1 = 180 # fuel
    I2 = 100 # within moderator - Na
    I = [I0, I1, I0, I2]
    
    widths_of_buffers = [L0, L1, L2, L]
    xs_media, media = set_media(materials,
        widths_of_buffers, ['Fe', m, 'Fe', 'Na'])
    r = equivolume_mesh(I0, 0, widths_of_buffers[0], geo)
    for i in range(3):
        Lb, Le = widths_of_buffers[i], widths_of_buffers[i+1]
        Ix = I[i] #[i] = I0 if i % 2 == 0 else I1
        r = np.append(r, equivolume_mesh(Ix, Lb, Le, geo)[1:])
    
    data = input_data(xs_media, media, r, geo, LBC=LBC, RBC=RBC)
    slvr_opts = solver_options(iitmax=5, oitmax=5, ritmax=200, CMFD=True,
                                pCMFD=False, Anderson_depth='auto')
    filename = os.path.join(odir, case + "_LBC%dRBC%d_I%d" %
                            (LBC, RBC, sum(I)))
    flx, k = run_calc_with_RM_its(data, slvr_opts, filename)
    np.testing.assert_allclose(k, 1.0, atol=1.e-4, err_msg=case +
                                ": criticality not verified")
    