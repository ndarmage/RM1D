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

sys.path.append(os.path.join(os.getcwd(), ".."))
from data.SoodPNE2003 import *
from FDsDiff1D import input_data, solver_options, run_calc_with_RM_its

sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "CPM1D"))
# sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "CPM1D", "KinPy"))
# import algo609
from cpm1dcurv import equivolume_mesh

odir = "output"
refdir = os.path.join("..", "..", "CPM1D", "tests", "output")


# Table 3
# Critical Dimensions, rc, for One-Group Bare Pu-239 (c=1.50)
# Problem Identifier Geometry r, (mfp) rc (cm) Reference
# 2 PUa-l-0-SL Slab 0.605055 1.853722 [161
# Table 4
# Critical Dimensions, rc, for One-Group Bare Pu-239 (c=1.40)
# Problem Identifier Geometry rc (mfp) rc (cm) Reference
# 6 PUb-1-0-SL Slab     0.73660355   2.256751 [351
# 7 PUb-1-0-CY Cylinder 1.396979     4.279960 [361,[371
# 8 PUb-1-0-SP Sphere   1.9853434324 6.082547 [351
# Table 5
# Normalized Scalar Fluxes for One-Group Bare Pu-239 (c=1.40)
# Problem Identifier Geometry r/rc = 0.25, r/rc = 0.5, r/rc = 0.75, r/rc = 1
# 6 PUb-1-0-SL Slab     0.9701734  0.8810540  0.7318131  0.4902592
# 7 PUb-1-0-CY Cylinder     -      0.8093         -      0.2926
# 8 PUb-1-0-SP Sphere   0.93538006 0.75575352 0.49884364 0.19222603
refflx = {
    'PUb-1-0-SL': np.array([0.9701734, 0.8810540, 0.7318131, 0.4902592]),
    'PUb-1-0-CY': np.array([np.nan, 0.8093, np.nan, 0.2926]),
    'PUb-1-0-SP': np.array([0.93538006, 0.75575352, 0.49884364, 0.19222603])
}


geoms = ["slab", "cylinder", "sphere"]


if __name__ == "__main__":

    import logging as lg
    lg.info("***          Sood's test suite          ***")
    lg.info("*** 1G homogeneous isotropic scattering ***")
    
    # positions (fractions r/rc) to verify the flux results
    rf = np.linspace(0, 1, 5)
    
    m = 'PUa'  # only one case in the test suite
    L = rc = 1.853722 * 2 # cm, critical length
    # L = rc = 0.605055  # mfp
    xs_media, media = set_media(materials[m], L, m)
        
    geo = "slab"
    case = "%s-1-0-%s" % (m, get_geoid(geo))
    lg.info("Test case: " + case)
    
    # load reference results computed by CPM1D
    k_ref, flx_ref  = np.load(os.path.join(refdir, case + "_ref.npy"),
                              allow_pickle=True)

    I = 20  # number of cells in the spatial mesh
    r = equivolume_mesh(I, 0, L, geo)
    # r = np.array([0, 1 / 8., 1 / 6., 0.9, 1]) * L
    LBC, RBC = 0, 0
    data = input_data(xs_media, media, r, geo, LBC=LBC, RBC=RBC)
    
    # ks is needed anyway when validating the input solver options
    ritmax = 100
    slvr_opts = solver_options(iitmax=5, oitmax=5, ritmax=ritmax,
                               CMFD=True, pCMFD=False)
    filename = os.path.join(odir, case + "_LBC%dRBC%d_I%d_it%d" %
                            (LBC, RBC, I, ritmax))
    flx, k = run_calc_with_RM_its(data, slvr_opts, filename)
    
    
    
    
    sys.exit('stop')
    m = 'PUb'  # only one case in the test suite
    rc_dict = {'slab': 2.256751, 'cylinder': 4.279960,
               'sphere': 6.082547}  # critical lengths
    for geo in geoms:
        case = "%s-1-0-%s" % (m, get_geoid(geo))
        lg.info("Test case: " + case)
        L = rc_dict[geo]
        rs = rf * L
        if geo == 'slab':
            I, nks, LBC, ratio0 = 40, 0, 0, .975
            rs += L
            L *= 2
        else:
            nks, LBC = 4, 2
            if 'cylind' in geo:
                I, ratio = 40, .95
            else:
                I, ratio = 50, .9
        # r = equivolume_mesh(I, 0, L, geo)
        r = geomprogr_mesh(N=I, L=L, ratio=.95)
        xs_media, media = set_media(materials[m], L, m)
        data = input_data(xs_media, media, r, geo, LBC=LBC, RBC=0,
                          per_unit_angle=False)
        k, flx = solve_cpm1D(data, solver_options(ks=np.full(I, nks)),
                             full_spectrum=False, vrbs=False)
        np.testing.assert_allclose(k, 1, atol=1.e-3, err_msg=case +
                                   ": criticality not verified")
        
        # get the values of the flux at the positions requested by the
        # benchmark by piecewise linear interpolation
        # flx /= np.dot(data.Vi, flx[0,:])
        f = interpolate.interp1d(data.xim, flx[0,:], fill_value="extrapolate")
        flxs = f(rs)
        # normalize the flux to be unitary at the center
        flxs /= flxs[0]
        rel_err_pc = (1 - flxs[1:] / refflx[case]) * 100
        # print(('{:6.3f} '*4).format(*rel_err_pc))
        np.testing.assert_array_less(np.nan_to_num(rel_err_pc),
                                     np.ones(4), err_msg=case +
                                     ": flux distribution not verified")
        