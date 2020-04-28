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
I = 4  # number of cells in the spatial mesh
    
ks = np.full(I, 2)  # quadrature order for each spatial cell
GaussQuadrature = "Gauss-Jacobi"  # type of Gauss quadrature
slvr_opts = solver_options(GQ=GaussQuadrature, ks=ks)

xs_media = {
    'HM':{  # homogeneous medium
        'st': st, 'ss': ss, 'nsf': nsf, 'chi': chi, 'D': D
    }
}
media = [['HM', L]]  # i.e. homogeneously filled


def solve_problem(geometry_type = "cylinder"):
    lg.info("Solve the MC2011 problem by CPM in the " +
            geometry_type)
    # definition of the spatial mesh
    r = equivolume_mesh(I, 0, L, geometry_type)
    # remind that volumes in input_data are per steradiant angle and
    # differ from the following ones
    # V = calculate_volumes(r, geometry_type)

    Homog2GSlab_data = input_data(xs_media, media, r,
                                  geometry_type, LBC=2, RBC=2,
                                  per_unit_angle=False)
        
    return solve_cpm1D(Homog2GSlab_data, slvr_opts)


if __name__ == "__main__":

    lg.basicConfig(level=lg.INFO)  # filename = logfile
    
    # solve the infinite homogeneous medium problem
    flxinf = np.dot(np.linalg.inv(np.diag(st) - ss[:,:,0]), chi)
    kinf = np.dot(nsf, flxinf)  # 1.07838136
    
    for geo in ["cylinder", "sphere"]:
        k, flx = solve_problem(geo)
        flx *= np.linalg.norm(flxinf) / np.linalg.norm(flx[:, 0])
        np.testing.assert_allclose(flx, np.tile(flxinf, (I, 1)).T,
                                   err_msg="flx-inf not verified")
        np.testing.assert_allclose(k, kinf, atol=1.e-7,
                                   err_msg="k-inf not verified")
