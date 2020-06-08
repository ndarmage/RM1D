# --*-- coding:utf-8 --*--
"""
Calculate the collision probabilities in the empty cylinder.
They must be equal to the area in the circle subtended by the
radial arcs.
"""
import os, sys
import logging as lg
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), ".."))
from cpm1dcurv import *

np.set_printoptions(precision=5)


if __name__ == "__main__":

    logfile = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    lg.basicConfig(level=lg.INFO)  # filename = logfile
    lg.info("* verbose output only in DEBUG logging mode *")
    geometry_type = "cylinder"

    # nb of rings in the cylinder
    L = 1. / np.pi**.5  # cm, body thickness
    I = 5  # number of cells in the spatial mesh
    # r = np.linspace(0, L, I+1)
    r = equivolume_mesh(I, 0, L, geometry_type)
    # r = np.array([0., .075, .15]); I = r.size - 1;  # test
    V = calculate_volumes(r, geometry_type)
    np.testing.assert_allclose(V, np.ones(I) / float(I),
        err_msg="Equal volumes are not verified")
    ks = np.full(I, 8)  # quadrature order for each spatial cell

    G = 1  # nb of energy groups
    # c = 0.5  # number of secondaries by scattering
    st = np.ones(G) * 1.e-8
    st_r = np.tile(st, (I, 1)).T
    # ss_r = c * st_r
    # # we use st / 8 for nsf
    # nsf_r = 0.125 * st_r

    tr_data = calculate_tracking_data(r, ks, quadrule="Gauss-Jacobi")

    eP, cP = calculate_probs(r, st_r, tr_data, V, geometry_type)
    
    lg.info('CP :\n' + str(cP[0, :, :]))
    lg.info('EP+:\n' + str(eP[0, :, :, 0]))
    lg.info('EP-:\n' + str(eP[0, :, :, 1]))
    
    np.testing.assert_allclose(np.zeros((I, I),), cP[0, :, :],
        atol=1.e-6, rtol=np.inf, err_msg="CP should be all zeros!")
    np.testing.assert_allclose(-np.triu(eP[0, 1:, :, 0], k=1),
        eP[0, 1:, :, 1], atol=1.e-7, err_msg="- triu(EP+) != EP-")