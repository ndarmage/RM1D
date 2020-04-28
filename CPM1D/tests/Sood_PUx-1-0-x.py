#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
Test cases by Sood et al. [Sood2003]_.

.. [Sood2003] Sood, A., Forster, R. A., & Parsons, D. K. (2003). Analytical
              benchmark test set for criticality code verification. Progress
              in Nuclear Energy, 42(1), 55-106.
"""
import sys, os
import numpy as np

sys.path.append(os.path.join(os.getcwd(), ".."))
from cpm1dcurv import *

sys.path.insert(0, os.path.join("..", "..", "FD1dMGdiff"))
from data.SoodPNE2003 import *
from FDsDiff1D import input_data


# Table 3
# Critical Dimensions, rc, for One-Group Bare Pu-239 (c=1.50)
# Problem Identifier Geometry r, (mfp) rc (cm) Reference
# 2 PUa-l-0-SL Slab 0.605055 1.853722 [161
# Table 4
# Critical Dimensions, rc, for One-Group Bare Pu-239 (c=1.40)
# Problem Identifier Geometry rc (mfp) rc (cm) Reference
# 6 Pub-l-0-SL Slab     0.73660355   2.256751 [351
# 7 Pub-I-0-CY Cylinder 1.396979     4.279960 [361,[371
# 8 Pub-l-0-SP Sphere   1.9853434324 6.082547 [351
# Table 5
# Normalized Scalar Fluxes for One-Group Bare Pu-239 (c=1.40)
# Problem Identifier Geometry r/rc = 0.25, r/rc = 0.5, r/rc = 0.75, r/rc = 1
# 6 Pub-l-0-SL Slab     0.9701734  0.8810540  0.7318131  0.4902592
# 7 Pub-l-0-CY Cylinder     -      0.8093         -      0.2926
# 8 Pub-l-0-SP Sphere   0.93538006 0.75575352 0.49884364 0.19222603


def get_geoid(geo):
    if geo == 'slab':
        g = 'SL'
    elif 'cylind' in geo:
        g = 'CY'
    elif 'spher' in geo:
        g = 'SP'
    else:
        raise InputError('unknown input geometry type')
    return g


def set_media(m, L, name):
    xs_media = {name:{  # homogeneous medium
        'st': m['st'], 'ss': m['ss'], 'nsf': m['nsf'],
        'chi': m['chi'], 'D': m['D']}
    }
    media = [[name, L]]  # i.e. homogeneously filled
    return xs_media, media


geoms = ["slab", "cylinder", "sphere"]


if __name__ == "__main__":

    import logging as lg
    lg.info("***          Sood's test suite          ***")
    lg.info("*** 1G homogeneous isotropic scattering ***")
    
    m = 'PUa'  # only one case in the test suite
    L = rc = 1.853722 * 2 # cm, critical length
    # L = rc = 0.605055  # mfp
    xs_media, media = set_media(materials[m], L, m)
        
    geo = "slab"
    case = "%s-1-0-%s" % (m, get_geoid(geo))
    lg.info("Test case: " + case)

    I = 20  # number of cells in the spatial mesh
    r = equivolume_mesh(I, 0, L, geo)
    # r = np.array([0, 1 / 8., 1 / 6., 0.9, 1]) * L
    data = input_data(xs_media, media, r, geo, LBC=0, RBC=0)
    
    # ks is needed anyway when validating the input solver options
    k, flx = solve_cpm1D(data, solver_options(ks=np.full(I, 0)), False)
    np.testing.assert_allclose(k, 1, atol=1.e-3, err_msg=case +
                               ": criticality not verified")
    
    m = 'PUb'  # only one case in the test suite
    rc_dict = {'slab': 2.256751, 'cylinder': 4.279960,
               'sphere': 6.082547}  # critical lengths
    for geo in geoms:
        lg.info("Test case: %s-1-0-%s" % (m, get_geoid(geo)))
        L = rc_dict[geo]
        if geo == 'slab':
            I, nks, LBC = 20, 0, 0
            L *= 2
        else:
            I, nks, LBC = 4, 2, 2
        r = equivolume_mesh(I, 0, L, geo)
        xs_media, media = set_media(materials[m], L, m)
        data = input_data(xs_media, media, r, geo, LBC=LBC, RBC=0,
                          per_unit_angle=False)
        k, flx = solve_cpm1D(data, solver_options(ks=np.full(I, nks)),
                             vrbs=False)
        np.testing.assert_allclose(k, 1, atol=1.e-3, err_msg=case +
                                   ": criticality not verified")