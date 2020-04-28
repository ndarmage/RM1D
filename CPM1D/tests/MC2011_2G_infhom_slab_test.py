# --*-- coding:utf-8 --*--
"""
Solve the integral transport equation in the cylinder by CPM
using the two group data from the MC2011 article.
"""
import os, sys
import logging as lg
import numpy as np
sys.path.append(os.path.join(os.getcwd(), ".."))
from cpm1dcurv import *

sys.path.insert(0, os.path.join("..", "..", "FD1dMGdiff"))
from data.homog2GMC2011 import *

L = 21.5 / 2.  # outer radius, equal to half pitch of a fuel assembly

xs_media = {
    'HM':{  # homogeneous medium
        'st': st, 'ss': ss, 'nsf': nsf, 'chi': chi, 'D': D
    }
}
media = [['HM', L]]  # i.e. homogeneously filled

geometry_type = "slab"

if __name__ == "__main__":

    lg.basicConfig(level=lg.INFO)  # filename = logfile
    lg.info("Reproduce the inf-hom medium with MC2011 data " +
            " by CPM in the " + geometry_type)

    # test a general mesh for the infinite homog medium
    r = np.array([0, 1 / 8., 1 / 6., 0.9, 1]) * L
    I = r.size - 1  # number of cells in the spatial mesh
    Homog2GSlab_data = input_data(xs_media, media, r,
                                  geometry_type, LBC=2, RBC=2)
    
    # ks is needed anyway when validating the input solver options
    k, flx = solve_cpm1D(Homog2GSlab_data, solver_options(ks=np.full(I, 0)))
    
    flxinf = np.dot(np.linalg.inv(np.diag(st) - ss[:,:,0]), chi)
    kinf = np.dot(nsf, flxinf)  # 1.07838136
    flxinf *= np.linalg.norm(flx[:, 0]) / np.linalg.norm(flxinf)
    np.testing.assert_allclose(flx, np.tile(flxinf, (I, 1)).T,
                               err_msg="flx-inf not verified")
    np.testing.assert_allclose(k, kinf, atol=1.e-7,
                               err_msg="k-inf not verified")

    # solve the MC2011 problem
    lg.info("Solve the MC2011 problem by CPM in the " + geometry_type)
    # L *= 2
    # media = [['HM', L]]
    I = 100  # number of cells in the spatial mesh
    r = equivolume_mesh(I, 0, L, geometry_type)
    # warning: remind that the solution in half slab use white reflection
    # at the center, introducing some error!
    Homog2GSlab_data = input_data(xs_media, media, r,
                                  geometry_type, LBC=2, RBC=0)
    
    # ks is needed anyway when validating the input solver options
    k, flx = solve_cpm1D(Homog2GSlab_data,
                         solver_options(ks=np.full(I, 0)), vrbs=False)
    kref = 0.744417
    np.testing.assert_allclose(k, kref, atol=1.e-3,
                               err_msg="ref k of MC2011 not verified")
    lg.info("k verified up to about three significant digits")
    lg.info("Reference k from S16 is %.6f" % kref)