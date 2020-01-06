#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This test case runs the 2G heterogeneous slab problem from [GIlas2003]_.

.. [Tomatis2011] Tomatis, D. and Dall'Osso, A., "Application of a numerical
   transport correction in diffusion calculations", Proc. Int. Conf. on
   Mathematics and Computational Methods Applied to Nuclear Science and
   Engineering (M&C 2011), Rio de Janeiro, RJ, Brazil, May 8-12, 2011.
"""
import sys, os
sys.path.append('..X..XFD1dMGdiff'.replace('X', os.path.sep))
from data.heter2GIlas2003 import *
from tests.heterSlab2GIlas2003 import *
from snmg1dslab import input_data, solver_options, quad_data
import numpy as np


# set b.c.
# LBC, RBC = 0, 0

# Initiate inpur data object
Heter2GSlab_data = input_data(xs_media, media, xi, geometry_type, LBC, RBC)

# Initiate solver options object
slvr_opts = solver_options()

# Initiate quadratures object
N, L = 16, 0
qdata = quad_data(N, L)

if __name__ == "__main__":

    import logging as lg
    lg.info("*** Solve the Ilas 2003 problem ***")
    from snmg1dslab import solve_sn

    flxm, k = solve_sn(Heter2GSlab_data, slvr_opts, qdata)

    basefilen = "../output/CORE%dLBC%dRBC%d_I%d_N%d" % (core_config,
                                                        Heter2GSlab_data.LBC,
                                                        Heter2GSlab_data.RBC,
                                                        Heter2GSlab_data.I,
                                                        qdata.N)
    np.save(basefilen + ".npy", np.array([k, flxm]), allow_pickle=True)
    #np.savez(basefilen + ".npz", k=k, flxm=flxm)
    
