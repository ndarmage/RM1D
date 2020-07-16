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
import sys
sys.path.append('..')
from data.Sood2003_heterSlab1G_data import *
from FDsDiff1D import input_data
import numpy as np

def MyWho():
     print([v for v in globals().keys() if not v.startswith('_')])

def get_fa(faid, ifa, L, nz=3):
   '''Returns a list of lists of FA id materials in location ifa in core.
   ifa = 0, 1, 2, ..., nfa-1.'''

   i = ifa * nz + 1
   if faid == 1:
           media = [
               ['WATER', L[i]],
               ['FUEL', L[i+1]],
               ['WATER', L[i+2]],
               ]
   else:
      raise TypeError('Invalid FA number [1-4].')

   return media


########################################
'''
Problem to solve? Insert index number from 0 to 2:
'''
P = 2                      # Problem index to solve
P_txt = problem_for_txt[P] # Problem No. according to benchmark

########################################
if (P < 0) or (P > 2):
    raise TypeError('Invalid test case. Choose between no. [0-2].')


# definition of the spatial mesh


rcf_cm = rcf_cm[P]   # critical length of half fuel region [cm]
rcf_mfp = rcf_mfp[P]  # critical length of water [cm]

rcw_cm = rcw_cm[P]
rcw_mfp = rcw_mfp[P]

nw = 2  # number of water segments
nz = 3  # number of different zones per fa
# materials boundaries
LcW = rcw_cm  # cm, water
LcF = rcf_cm * 2  # cm, fuel
Dxfa = np.array([LcW, LcF, LcW])

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

dxfa = np.concatenate((dxW, dxF, dxW), axis=0)

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
    cf = ssf[P]/stf[P]

xs_media = {
    'WATER':{  # homogeneous medium 1
        'st': np.array([stw[P]]), 'ss': np.array([ssw[P]]), 'nsf': np.array([sfw[P] * nuw[P]]) \
            , 'chi': np.array([chiw[P]]), 'D': np.array([Dw[P]])
    },
    'FUEL':{  # homogeneous medium 2
        'st': np.array([stf[P]]), 'ss': np.array([ssf[P]]), 'nsf': np.array([sff[P] * nuf[P]]), \
            'chi': np.array([chif[P]]), 'D': np.array([Df[P]])
    }
}

cum_sum_list = lambda l1: l1[0] + cum_sum_list(l1[1:]) if len(l1) != 1 else l1[0]
media = cum_sum_list([get_fa(1, i, Lmat) for i in range(nfa)])

# set b.c.
LBC, RBC = 0, 0

Heter1GSlab_data = input_data(xs_media, media, xi, geometry_type, LBC, RBC)

if __name__ == "__main__":

    import logging as lg
    lg.info("*** Solve Sood 2003 problem %d ***" % P_txt)
    from FDsDiff1D import run_calc_with_RM_its, solver_options

    ritmax = 200
    CMFD, pCMFD = True, False
    slvr_opts = solver_options(ritmax=ritmax, CMFD=CMFD, pCMFD=pCMFD)
    filename = "../output/Sood2003_kflx_LBC%dRBC%d_I%d_itr%d_P%d" % \
               (LBC, RBC, I, ritmax, P_txt)
    flx, k = run_calc_with_RM_its(Heter1GSlab_data, slvr_opts, filename)
