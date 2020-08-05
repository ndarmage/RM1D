#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This test case runs the 1G heterogeneous slab problem from [Sood2003]_.
   

                        Problem 30 - Ue-Fe-Na-1-0-SL
                             Non-symmetrical 4 regions


..[Sood2003] Analytical Benchmark Test Set For Criticality code Verification,
    Los Alamos National Laboratory, Applied Physics (X) Divisions, X-5 Diagnostics
    Applications Groupm P.O. Box 1663, MS F663, Los Alamos, NM 87545,
    Avneet Sood, R. Arthur Foster, and D. Kent Parsons, 2003
"""
import sys, os
sys.path.append('..X..XFD1dMGdiff'.replace('X', os.path.sep))
from data.Sood2003_heterSlab1G_P30_data import *
from tests1.Sood2003_HeterSlab1G_P30_test import Heter1GSlab_data
sys.path.append('..')  # do not use this at the beginning!
from snmg1dslab import input_data, solver_options, quad_data
import numpy as np


# set b.c.
# LBC, RBC = 0, 0

# or initiate input data object here...
# Heter2GSlab_data = input_data(xs_media, media, xi, geometry_type, LBC, RBC)

# Initiate solver options object
slvr_opts = solver_options()

# Initiate quadratures object
N, L = 16, 0
qdata = quad_data(N, L)

if __name__ == "__main__":

    import logging as lg
    lg.info("*** Solve Sood 2003 problem 30 ***")
    from snmg1dslab import solve_sn

    flxm, k = solve_sn(Heter1GSlab_data, slvr_opts, qdata)

    basefilen = "../output/Sood2003_kflx_SN_LBC%dRBC%d_I%d_N%d_P30" % (
                                                        Heter1GSlab_data.LBC,
                                                        Heter1GSlab_data.RBC,
                                                        Heter1GSlab_data.I,
                                                        qdata.N)
    np.save(basefilen + ".npy", np.array([k, flxm]), allow_pickle=True)
    #np.savez(basefilen + ".npz", k=k, flxm=flxm)
    
