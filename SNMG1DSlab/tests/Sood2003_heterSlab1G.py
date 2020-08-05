#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This test case runs the 1G heterogeneous slab problem from [Sood2003]_.
   

                       Index - 0 = Problem 4  - PUa-1-0-SL
                       Index - 1 = Problem 25 - UD2O-H2O(1)-1-0-SL
                       Index - 2 = Problem 26 - UD2O-H2O(10)-1-0-SL
                        
                     Cross section strucre [P4, P25, P26], (P=problem)
                     
            
..[Sood2003] Analytical Benchmark Test Set For Criticality code Verification,
    Los Alamos National Laboratory, Applied Physics (X) Divisions, X-5 Diagnostics
    Applications Groupm P.O. Box 1663, MS F663, Los Alamos, NM 87545,
    Avneet Sood, R. Arthur Foster, and D. Kent Parsons, 2003
"""
import sys, os
sys.path.append('..X..XFD1dMGdiff'.replace('X', os.path.sep))
#from data.Sood2003_heterSlab1G_data import *
from tests1.Sood2003_HeterSlab1G_test import Heter1GSlab_data
sys.path.append('..')  # do not use this at the beginning!
from snmg1dslab import input_data, solver_options, quad_data
import numpy as np

P_txt = int(input("Please insert problem index for output filename \n\n"))
if (P_txt < 0) or (P_txt > 2):
    raise TypeError('Invalid test case. Choose between no. [0-2].')

# Initiate solver options object
slvr_opts = solver_options()

# Initiate quadratures object
N, L = 16, 0
qdata = quad_data(N, L)

if __name__ == "__main__":

    import logging as lg
    lg.info("*** Solve Sood 2003 problem %d ***" % P_txt)
    from snmg1dslab import solve_sn

    flxm, k = solve_sn(Heter1GSlab_data, slvr_opts, qdata)

    basefilen = "../output/Sood2003_kflx_SN_LBC%dRBC%d_I%d_N%d_P%d" % (
                                                        Heter1GSlab_data.LBC,
                                                        Heter1GSlab_data.RBC,
                                                        Heter1GSlab_data.I,
                                                        qdata.N,
                                                        P_txt)
    np.save(basefilen + ".npy", np.array([k, flxm]), allow_pickle=True)
    #np.savez(basefilen + ".npz", k=k, flxm=flxm)
    
