#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This test case runs the 1G homogeneous slab problem from [Sood2003]_.


                       Index - 0 = Problem 2  - PUa-1-0-SL
                       Index - 1 = Problem 6  - PUb-1-0-SL
                       Index - 2 = Problem 12 - Ua-1-0-SL
                       Index - 3 = Problem 22 - UD2O-1-0-SL 
                        
            Cross section strucre [P2, P6, P12, P22], (P=problem)
                        
            
.. [Sood2003] Sood, A., Forster, R. A., & Parsons, D. K. (2003). Analytical
              benchmark test set for criticality code verification. Progress
              in Nuclear Energy, 42(1), 55-106.
"""

import sys
sys.path.append('..')
from data.Sood2003_homogSlab1G_data import *

from FDsDiff1D import input_data
import numpy as np

def MyWho():
     print([v for v in globals().keys() if not v.startswith('_')])

P = int(input("Which problem to solve? Insert index number from 0 to 3 \n \
Index - 0 = Problem 2  - PUa-1-0-SL   \n \
Index - 1 = Problem 6  - PUb-1-0-SL   \n \
Index - 2 = Problem 12 - Ua-1-0-SL    \n \
Index - 3 = Problem 22 - UD2O-1-0-SL  \n \
===>"))
if (P < 0) or (P > 3):
    raise TypeError('Invalid test case. Choose between no. [0-3].')

P_txt = problem_for_txt[P] # Problem No. according to benchmark
########################################

# How many mfp? - No, of r_c:
irc = 1

# definition of the spatial mesh
# rc is the length from slab's center to edge |   :-->|
rc_cm = rc_cm[P] # Critical length [cm]
rc_mfp = rc_mfp[P] # Critical length [cm]

L = irc*2*rc_cm # Slab's length

I = 30  # nb of spatial cells
xi = np.linspace(0, L, I+1)  # equidistant mesh

geometry_type = 'slab'

# ------------------------------------------------ #
#change c?
c = False
if c == True:
        
    # test the CMFD scheme with different material conditions
    sd = lambda ss: ss
    sa = lambda st, ss: st - sd(ss)
    def check_c(ss, st):
        s = sd(ss)
        print('sd = ', s)
        print('sa = ', sa(st, ss))
        print('c  = ', s / st)
        pass
    
    def alter_c(ss, st, c):
        "Alter the (isotropic) scattering matrix by the input c constants."
        print('alter ss with new c = ' + str(c))
        ss0 = ss
        ss0 /= sd(ss) / (st * c)
        return ss
    
    print('Input xs set:')
    check_c(ss, st)
    print(sf)  # to check physical data
    kinf0, sa0 = calc_kinf(sf, st, ss, chi), sa(st, ss)
    print('original kinf = %.6f' % kinf0)
    c = [.8]
    ss = alter_c(ss, st, c)
    sam = sa(st, ss)
    print('modify production for physical consistency')
    sf *= sam / sa0
    kinf = calc_kinf(sf, st, ss, chi)
    # nsf /= kinf / kinf0 ?!? ...skipped
    check_c(ss, st)
    print('new kinf %.6f' % kinf)
    # input('Press a key to launch the calculation...')
else:
    c = ss[P]/st[P]
# ------------------------------------------------ #

xs_media = {
    'HM':{  # homogeneous medium
        'st': np.array([st[P]]), 'ss': np.array([ss[P]]), 'nsf': np.array([sf[P]*nu[P]]), \
                        'chi': np.array([chi[P]]), 'D': np.array([D[P]])
    }
}
media = [['HM', L]]  # i.e. homogeneously filled

# set b.c.
LBC, RBC = 0, 0

Homog1GSlab_data = input_data(xs_media, media, xi, geometry_type, LBC, RBC)

if __name__ == "__main__":

    import logging as lg
    lg.info("*** Solve Sood 2003 homogeneous problem %d ***" % P_txt)
    from FDsDiff1D import run_calc_with_RM_its, solver_options

    ritmax = 2000
    CMFD, pCMFD = True, False
    slvr_opts = solver_options(ritmax=ritmax, CMFD=CMFD, pCMFD=pCMFD)
    filename = "../output/Sood2003_kflx_CMFD_LBC%dRBC%d_I%d_irc%d_c%d_P%d_wAA1" \
                                    % (LBC, RBC, I, irc, c*100, P_txt)
    flx, k = run_calc_with_RM_its(Homog1GSlab_data, slvr_opts, filename)
    #kS16 = 0.744417, 
    #lg.info("Reference k from S16 is %.6f" % kS16)
    #np.testing.assert_allclose(k, kCPM, atol=1.e-4,
    #                         err_msg="ref k of MC2011 not verified")
    