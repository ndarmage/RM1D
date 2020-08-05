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
import sys
sys.path.append('..')
from data.Sood2003_heterSlab1G_P30_data import *
from FDsDiff1D import input_data
import numpy as np

def MyWho():
     print([v for v in globals().keys() if not v.startswith('_')])

def get_fa(faid, ifa, L, nz=4):
   '''Returns a list of lists of FA id materials in location ifa in core.
   ifa = 0, 1, 2, ..., nfa-1.'''

   i = ifa * nz + 1
   if faid == 1:
           media = [
               ['CLAD', L[i]],
               ['FUEL', L[i+1]],
               ['CLAD', L[i+2]],
               ['NA', L[i+3]],
               ]
   else:
      raise TypeError('Invalid FA number [1-4].')

   return media

##################


# definition of the spatial mesh
rcf_cm = 5.119720083   # critical length of fuel region [cm]
rcNa_cm = 2.002771002  # critical length of water [cm]
rcc_cm = 0.317337461   # critical length of cladding [cm]

rcf_mfp = 2.0858098
rcNa_mfp = 0.173
rcc_mfp = 0.0738

nz = 4  # number of different zones per fa

# materials boundaries
LcNa = rcNa_cm  # cm, water
LcC = rcc_cm  # cm, clad
LcF = rcf_cm  # cm, fuel

Dxfa = np.array([LcC, LcF, LcC, LcNa])

nf = 1  # number of fuel segments
nfa = 1  # total number of FAs


Dxmat = np.tile(Dxfa, nfa)  # material width
Lmat = np.cumsum(Dxmat)
Lmat = np.insert(Lmat, 0, 0)

# fine mesh
nxNa = 50  # no. of cells in water - can change this
nxC = 50  # no. of cells in clad - can change this
nxF = 100  # no. of cells in fuel - can change this

dxNa = np.linspace(0, LcNa, nxNa + 1)
dxNa = dxNa[1:] - dxNa[:-1]
dxC = np.linspace(0, LcC, nxC + 1)
dxC = dxC[1:] - dxC[:-1]
dxF = np.linspace(0, LcF, nxF + 1)
dxF = dxF[1:] - dxF[:-1]

dxfa = np.concatenate((dxC, dxF, dxC, dxNa), axis=0)

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
    'NA':{  # homogeneous medium 1
        'st': stNa, 'ss': ssNa, 'nsf': sfNa * nuNa, 'chi': chiNa, 'D': DNa
    },
    'CLAD':{  # homogeneous medium 2
        'st': stc, 'ss': ssc, 'nsf': sfc * nuc, 'chi': chic, 'D': Dc
    },
    'FUEL':{  # homogeneous medium 3
        'st': stf, 'ss': ssf, 'nsf': sff * nuf, 'chi': chif, 'D': Df
    }
}

cum_sum_list = lambda l1: l1[0] + cum_sum_list(l1[1:]) if len(l1) != 1 else l1[0]
media = cum_sum_list([get_fa(1, i, Lmat) for i in range(nfa)])

# set b.c.
LBC, RBC = 0, 0

Heter1GSlab_data = input_data(xs_media, media, xi, geometry_type, LBC, RBC)

if __name__ == "__main__":

    import logging as lg
    lg.info("*** Solve Sood 2003 problem 30 ***")
    from FDsDiff1D import run_calc_with_RM_its, solver_options

    ritmax = 200
    CMFD, pCMFD = True, False
    slvr_opts = solver_options(ritmax=ritmax, CMFD=CMFD, pCMFD=pCMFD)
    filename = "../output/Sood2003_kflx_LBC%dRBC%d_I%d_itr%d_P30" % \
               (LBC, RBC, I, ritmax)
    flx, k = run_calc_with_RM_its(Heter1GSlab_data, slvr_opts, filename)
