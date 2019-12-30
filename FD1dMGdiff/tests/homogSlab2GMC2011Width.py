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
# Benchmark data
L0 = 21.5
I0 = 400
dx0 = 21.5/I0
""" Calculating full slab, different width, same dx """
I = np.array([I0-240, I0-160, I0-80, I0, 480, 560, 720, 800])
L = np.multiply(dx0,I)

wditer = np.size(I) # No. of width's iterations

for wi in range(0,wditer):
    xi = np.linspace(0, L[wi], I[wi]+1)  # equidistant mesh

    geometry_type = 'slab'

    xs_media = {
    'HM':{  # homogeneous medium
        'st': st, 'ss': ss, 'nsf': nsf, 'chi': chi, 'D': D
        }
    }
    media = [['HM', L[wi]]]  # i.e. homogeneously filled

    # set b.c.
    LBC, RBC = 0, 0

    Homog2GSlab_data = input_data(xs_media, media, xi, geometry_type, LBC, RBC)

    if __name__ == "__main__":
        
        import logging as lg
        lg.info("*** Solve the M&C 2011 problem ***")
        from FDsDiff1D import run_calc_with_RM_its, solver_options
        slvr_opts = solver_options(ritmax=10)
        filename = "../output/kflx_Wd_LBC%dRBC%d_I%d_L%d_it10" % (LBC, RBC, I[wi], L[wi])
        flx, k = run_calc_with_RM_its(Homog2GSlab_data, slvr_opts, filename)
