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
from cpm1dcurv import *

sys.path.insert(0, os.path.join("..", "..", "FD1dMGdiff"))
from data.SoodPNE2003 import *
from FDsDiff1D import input_data


odir = 'output'

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
    
    # cases for the ictt26 article:
    # [2, 3, 4, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100]
    for I0 in [40]:
    
        m = 'PUa'  # only one case in the test suite
        L = rc = 1.853722 * 2 # cm, critical length
        # L = rc = 0.605055  # mfp
        xs_media, media = set_media(materials[m], L, m)
            
        geo = "slab"
        case = "%s-1-0-%s" % (m, get_geoid(geo)) # + 'h'
        lg.info("Test case: " + case)

        # I = I0 #* 2 # number of cells in the spatial mesh
        I = 25
        r = equivolume_mesh(I, 0, L, geo)
        # r = np.array([0, 1 / 8., 1 / 6., 0.9, 1]) * L
        data = input_data(xs_media, media, r, geo, LBC=0, RBC=0)
        
        # ks is needed anyway when validating the input solver options
        k, flx = solve_cpm1D(data, solver_options(ks=np.full(I, 0)), False)
        np.testing.assert_allclose(k, 1, atol=1.e-3, err_msg=case +
                                   ": criticality not verified")
        np.save(os.path.join(odir, case + '_ref_I%d.npy' % I), [k, flx, None])
        
        m = 'PUb'  # only one case in the test suite
        # critical lengths
        rc_dict = {'slab': 2.256751, 'cylinder': 4.279960, 'sphere': 6.082547}
        
        for geo in geoms:
            case = "%s-1-0-%s" % (m, get_geoid(geo))
            lg.info("Test case: " + case)
            L = rc_dict[geo]
            rs = rf * L
            if geo == 'slab':
                I, nks, LBC, ratio0 = I0*2, 0, 0, .975  # 25
                rs += L
                L *= 2
            else:
                nks, LBC = 4, 2
                if 'cylind' in geo:
                    I, ratio = I0, .95  # 25
                else:
                    I, ratio = I0, .9  # 40
            # r = equivolume_mesh(I, 0, L, geo)  # used for all calcs of ictt26
            
            # info: mesh by geometric progression can get higher accuracy
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
            np.save(os.path.join(odir, case + '_ref_I%d.npy' % I), [k, flx, rel_err_pc])
        