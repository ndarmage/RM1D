#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This test case runs the 2G homogeneous slab problem from [Tomatis2011].

.. [Tomatis2011] Tomatis, D. and Dall'Osso, A., "Application of a numerical
   transport correction in diffusion calculations", Proc. Int. Conf. on
   Mathematics and Computational Methods Applied to Nuclear Science and
   Engineering (M&C 2011), Rio de Janeiro, RJ, Brazil, May 8-12, 2011.
"""
import sys, os
sys.path.append('..')
sys.path.append('..X..XFD1dMGdiff'.replace('X', os.path.sep))
from data.homog2GMC2011 import *
from snmg1dslab import input_data, solver_options, quad_data
import numpy as np

# comment for Roy: please, check the following bacause it should be redundant
# after import

L = 21.5  # Core width
I = 100  # No. of spatial cells
xi = np.linspace(0, L, I+1)  # equidistant mesh      
# definition of the spatial mesh

geometry_type = 'slab'

xs_media = {
    'HM':{  # homogeneous medium
        'st': st, 'ss': ss, 'nsf': nsf, 'chi': chi, 'D': D
    }
}
media = [['HM', L]]  # i.e. homogeneously filled


# set b.c.
LBC, RBC = 0, 0

# Initiate inpur data object
Homog2GSlab_data = input_data(xs_media, media, xi, geometry_type, LBC, RBC)
# Initiate solver options object
slvr_opts = solver_options()

# Initiate quadratures object
N, L = 16, 0
qdata = quad_data(N, L)

if __name__ == "__main__":

    import logging as lg    
    lg.info("*** Solve the MC 2011 problem ***")
    from snmg1dslab import solve_sn

    flxm, k = solve_sn(Homog2GSlab_data, slvr_opts, qdata)
    basefilen = "../output/kflx_MC2011_LBC%dRBC%d_I%d_L%d_N%d" % (
                                                        Homog2GSlab_data.LBC,
                                                        Homog2GSlab_data.RBC,
                                                        Homog2GSlab_data.I,
                                                        Homog2GSlab_data.xi[-1],
                                                        qdata.N)
    np.save(basefilen + ".npy", np.array([[k], flxm]), allow_pickle=True)
    #np.savez(basefilen + ".npz", k=k, flxm=flxm)
    
