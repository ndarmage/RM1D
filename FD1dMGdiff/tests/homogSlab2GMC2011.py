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
import numpy as np

# definition of the spatial mesh
L = 21.5 / 2.  # slab width, equal to half pitch of a fuel assembly
I = 6  # nb of spatial cells
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

# solver options (to override options in the main module)
ritmax = 10 # set to 1 to skip Ronen iterations

if __name__ == "__main__":

    import logging as lg
    lg.info("*** Solve the M&C 2011 problem ***")
    from FDsDiff1D import run_calc_with_RM_its as run_calc
    filename = "output/kflx_LBC%dRBC%d_I%d" % (LBC, RBC, I)
    run_calc(filename)
