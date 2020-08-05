#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This test case runs the 1G heterogeneous slab problem from [Sood2003]_.
   

                        Problem 3 - PUa-H2O(1)-1-0-SL
                           Non-symmetrical 2 region

..[Sood2003] Analytical Benchmark Test Set For Criticality code Verification,
    Los Alamos National Laboratory, Applied Physics (X) Divisions, X-5 Diagnostics
    Applications Groupm P.O. Box 1663, MS F663, Los Alamos, NM 87545,
    Avneet Sood, R. Arthur Foster, and D. Kent Parsons, 2003

"""
import sys
sys.path.append('..')
from data.Sood2003_heterSlab1G_P3_data import *
from FDsDiff1D import input_data
import numpy as np

def MyWho():
     print([v for v in globals().keys() if not v.startswith('_')])

def get_fa(faid, ifa, L, nz=6):
   '''Returns a list of lists of FA id materials in location ifa in core.
   ifa = 0, 1, 2, ..., nfa-1.'''

   i = ifa * nz + 1
   if faid == 1:
           media = [
               ['WATER', L[i]],
               ['FUEL', L[i+1]],
               ]
   else:
      raise TypeError('Invalid FA number [1-4].')

   return media

##################


# definition of the spatial mesh


rcf_cm = 1.47845    # critical length of half fuel region [cm]
rcw_cm = 3.063725   # critical length of water [cm]
rcf_mfp = 0.482566
rcw_mfp = 1
nw = 1  # number of water segments
nz = 2  # number of different zones per fa
# materials boundaries
LcW = rcw_cm  # cm, water
LcF = rcf_cm * 2  # cm, fuel
Dxfa = np.array([LcW, LcF])

nf = 1  # number of fuel segments
nfa = 1  # total number of FAs


Dxmat = np.tile(Dxfa, nfa)  # material width
Lmat = np.cumsum(Dxmat)
Lmat = np.insert(Lmat, 0, 0)

# fine mesh
nxW = 30  # no. of cells in water - can change this
nxF = 45  # no. of cells in fuel - can change this

dxW = np.linspace(0, LcW, nxW + 1)
dxW = dxW[1:] - dxW[:-1]
dxF = np.linspace(0, LcF, nxF + 1)
dxF = dxF[1:] - dxF[:-1]

dxfa = np.concatenate((dxF, dxW), axis=0)

dx = np.tile(dxfa, nfa)
xi = np.cumsum(dx)
xi = np.insert(xi, 0, 0)

L = xi[-1]  # core width
I = len(dx) # nb of spatial cells
#xi = np.linspace(0, L, I+1)  # equidistant mesh

geometry_type = 'slab'

# ------------------------------------------------ #
#change c?
c = False
if c == True:
        
    # test the CMFD scheme with different material conditions
    sd = lambda ss: ssf
    sa = lambda st, ssf: stf - sd(ssf)
    def check_c(ssf, stf):
        sf = sd(ssf)
        print('sd_f = ', sf)
        print('sa_f = ', sa(stf, ssf))
        print('c_f  = ', sf / stf)
        pass
    
    def alter_c(ssf, stf, cf):
        "Alter the (isotropic) scattering matrix by the input c constants."
        print('alter ss with new c = ' + str(cf))
        ss0f = ssf
        ss0f /= sd(ssf) / (stf * cf)
        return ssf
    
    print('Input xs set:')
    check_c(ssf, stf)
    print(sff)  # to check physical data
    sa0f = sa(stf, ssf)
    cf = [.8]
    ssf = alter_c(ssf, stf, cf)
    samf = sa(stf, ssf)
    print('modify production for physical consistency')
    sff *= samf / sa0f
    check_c(ssf, stf)
    # input('Press a key to launch the calculation...')
else:
    cf = ssf/stf

xs_media = {
    'WATER':{  # homogeneous medium 1
        'st': stw, 'ss': ssw, 'nsf': sfw * nuw, 'chi': chiw, 'D': Dw
    },
    'FUEL':{  # homogeneous medium 2
        'st': stf, 'ss': ssf, 'nsf': sff * nuf, 'chi': chif, 'D': Df
    }
}

core = 1

cum_sum_list = lambda l1: l1[0] + cum_sum_list(l1[1:]) if len(l1) != 1 else l1[0]
media = cum_sum_list([get_fa(1, i, Lmat) for i in range(nfa)])

# set b.c.
LBC, RBC = 0, 0

Heter1GSlab_data = input_data(xs_media, media, xi, geometry_type, LBC, RBC)

if __name__ == "__main__":

    import logging as lg
    lg.info("*** Solve Sood 2003 problem 3 ***")
    from FDsDiff1D import run_calc_with_RM_its, solver_options

    ritmax = 200
    CMFD, pCMFD = True, False
    slvr_opts = solver_options(ritmax=ritmax, CMFD=CMFD, pCMFD=pCMFD)
    filename = "../output/Sood2003_kflx_LBC%dRBC%d_I%d_itr%d_P3" % \
               (LBC, RBC, I, ritmax)
    flx, k = run_calc_with_RM_its(Heter1GSlab_data, slvr_opts, filename)
