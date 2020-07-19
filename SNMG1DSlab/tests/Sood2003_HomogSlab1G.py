#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This test case runs the 1G homogeneous slab problem from [Sood2003].

                       Index - 0 = Problem 2  - PUa-1-0-SL
                       Index - 1 = Problem 6  - PUb-1-0-SL
                       Index - 2 = Problem 12 - Ua-1-0-SL
                       Index - 3 = Problem 22 - UD2O-1-0-SL 
                        
            Cross section strucre [P2, P6, P12, P22], (P=problem)
                        
            
Analytical Benchmark Test Set For Criticality code Verification,
Los Alamos National Laboratory, Applied Physics (X) Divisions, X-5 Diagnostics
Applications Groupm P.O. Box 1663, MS F663, Los Alamos, NM 87545,
Avneet Sood, R. Arthur Foster, and D. Kent Parsons, 2003

"""
import sys, os
sys.path.append('..')
sys.path.append('..X..XFD1dMGdiff'.replace('X', os.path.sep))
from data.Sood2003_homogSlab1G_data import *
from tests1.Sood2003_HomogSlab1G_test import Homog1GSlab_data

from snmg1dslab import input_data, solver_options, quad_data
import numpy as np

P_txt = int(input("Please insert problem index for output filename \n\n"))
if (P_txt < 0) or (P_txt > 3):
    raise TypeError('Invalid test case. Choose between no. [0-3].')

# # after import
# rc_cm = 1.853722 # Critical radius of fuel
# rc_mfp = 0.605055

# L = 2*rc_cm  # Core width
# I = 200  # No. of spatial cells
# xi = np.linspace(0, L, I+1)  # equidistant mesh      
# # definition of the spatial mesh

# geometry_type = 'slab'

# xs_media = {
#     'HM':{  # homogeneous medium
#         'st': st, 'ss': ss, 'nsf': sf*nu, 'chi': chi, 'D': D
#     }
# }
# media = [['HM', L]]  # i.e. homogeneously filled

# # set b.c.
# LBC, RBC = 0, 0

# # Initiate inpur data object
# Homog1GSlab_data = input_data(xs_media, media, xi, geometry_type, LBC, RBC)
# # Initiate solver options object
slvr_opts = solver_options()

# Initiate quadratures object
N = 16
qdata = quad_data(N, L=0)

if __name__ == "__main__":

    import logging as lg    
    lg.info("*** Solve Sood 2003 problem %d ***" % P_txt)
    from snmg1dslab import solve_sn

    flxm, k = solve_sn(Homog1GSlab_data, slvr_opts, qdata)
    basefilen = "../output/Sood2003_kflx_LBC%dRBC%d_N%d_I%d_P%d" % (
                                                        Homog1GSlab_data.LBC,
                                                        Homog1GSlab_data.RBC,
                                                        qdata.N,
                                                        Homog1GSlab_data.I,
                                                        P_txt)
    np.save(basefilen + ".npy", np.array([k, flxm]), allow_pickle=True)
    #np.savez(basefilen + ".npz", k=k, flxm=flxm)
    
