#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This test case runs the 2G homogeneous slab problem from [Tomatis2011]_.

.. [Tomatis2011] Tomatis, D. and Dall'Osso, A., "Application of a numerical
   transport correction in diffusion calculations", Proc. Int. Conf. on
   Mathematics and Computational Methods Applied to Nuclear Science and
   Engineering (M&C 2011), Rio de Janeiro, RJ, Brazil, May 8-12, 2011.
"""
import sys
sys.path.append('..')
from data.homog2GMC2011 import *
from FDsDiff1D import input_data
import numpy as np

def MyWho():
     print([v for v in globals().keys() if not v.startswith('_')])

# definition of the spatial mesh
L = 21.5 / 2.  # slab width, equal to half pitch of a fuel assembly
I = 20  # nb of spatial cells
xi = np.linspace(0, L, I+1)  # equidistant mesh

geometry_type = 'slab'

xs_media = {
    'HM':{  # homogeneous medium
        'st': st, 'ss': ss, 'nsf': nsf, 'chi': chi, 'D': D
    }
}
media = [['HM', L]]  # i.e. homogeneously filled

# set b.c.
LBC, RBC = 2, 0

Homog2GSlab_data = input_data(xs_media, media, xi, geometry_type, LBC, RBC)

if __name__ == "__main__":

    import logging as lg
    lg.info("*** Solve the M&C 2011 problem ***")
    from FDsDiff1D import run_calc_with_RM_its, solver_options

    ritmax = 10
    CMFD, pCMFD = False, True
    slvr_opts = solver_options(ritmax=ritmax, CMFD=CMFD, pCMFD=pCMFD)
    filename = "./kflx_LBC%dRBC%d_I%d_it%d" % (LBC, RBC, I, ritmax)
    flx, k = run_calc_with_RM_its(Homog2GSlab_data, slvr_opts, filename)
