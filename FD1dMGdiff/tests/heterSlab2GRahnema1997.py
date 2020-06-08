#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This test case runs the 2G heterogeneous slab problem from [Rahnema1997]_.

.. [Rahnema1997] F. Rahnema, and E., M., Nichita "LEAKAGE CORRECTED
   SPATIAL (ASSEMBLY) HOMOGENIZATION TECHNIQUE", Elseviel Science Ltd., 
   Proc. Ann. Nucl. Energy, Vol. 24, No. 6, pp. 477-488, 1997.
"""
import sys
sys.path.append('..')
from data.heter2GRahnema1997 import *
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
            ['FUELI', L[i+1]],
            ['FUELII', L[i+2]],
            ['FUELII', L[i+3]],
            ['FUELI', L[i+4]],
            ['WATER', L[i+5]]
              ]
   elif faid == 2:
      media = [
            ['WATER', L[i]],
            ['FUELI', L[i+1]],
            ['FUELI', L[i+2]],
            ['FUELI', L[i+3]],
            ['FUELI', L[i+4]],
            ['WATER', L[i+5]]
              ]
   elif faid == 3:
      media = [
            ['WATER', L[i]],
            ['FUELI', L[i+1]],
            ['FUELIII', L[i+2]],
            ['FUELIII', L[i+3]],
            ['FUELI', L[i+4]],
            ['WATER', L[i+5]]
              ]
   elif faid == 4:
      media = [
            ['WATER', L[i]],
            ['FUELIII', L[i+1]],
            ['FUELIII', L[i+2]],
            ['FUELIII', L[i+3]],
            ['FUELIII', L[i+4]],
            ['WATER', L[i+5]]
              ]
   else:
      raise TypeError('Invalid FA number [1-4].')

   return media


# definition of the spatial mesh

# materials boundaries
DxW = 1.158  # cm, Water
DxF = 3.231  # cm, Fuel
nw = 2  # number of segments in water region
nf = 4  # number of segments in fuel region

nz = 6  # number of different zones per FA
nfa = 7  # total number of FAs in the core

Dxfa = np.array([DxW, DxF, DxF, DxF, DxF, DxW])
Dxmat = np.tile(Dxfa, nfa)  # material width
Lmat = np.cumsum(Dxmat)
Lmat = np.insert(Lmat, 0, 0)

# fine mesh
nxW = 8  # no. of cells in water - can change this
nxF = 22  # no. of cells in fuel - can change this

dxW = np.linspace(0, DxW, nxW + 1)
dxW = dxW[1:] - dxW[:-1]
dxF = np.linspace(0, DxF, nxF + 1)
dxF = dxF[1:] - dxF[:-1]

dxfa = np.concatenate((dxW, dxF, dxF, dxF, dxF, dxW), axis=0)
dx = np.tile(dxfa, nfa)
xi = np.cumsum(dx)
xi = np.insert(xi, 0, 0)

L = xi[-1]  # core width
I = len(dx) # nb of spatial cells
#xi = np.linspace(0, L, I+1)  # equidistant mesh

geometry_type = 'slab'

xs_media = {
    'WATER':{  # homogeneous medium 1
        'st': stw, 'ss': ssw, 'nsf': nsfw, 'chi': chiw, 'D': Dw
    },
    'FUELI':{  # homogeneous medium 2
        'st': st1, 'ss': ss1, 'nsf': nsf1, 'chi': chi1, 'D': D1
    },
    'FUELII':{  # homogeneous medium 3
        'st': st2, 'ss': ss2, 'nsf': nsf2, 'chi': chi2, 'D': D2
    },
    'FUELIII':{  # homogeneous medium 4
        'st': st3, 'ss': ss3, 'nsf': nsf3, 'chi': chi3, 'D': D3
    }
}

if (core_config < 1) or (core_config > 3):
    raise TypeError('Invalid core configuration no. [1-3].')
else:
    core = np.ones(nfa, dtype=np.int)
    core[1::2] += core_config

cum_sum_list = lambda l1: l1[0] + cum_sum_list(l1[1:]) if len(l1) != 1 else l1[0]
media = cum_sum_list([get_fa(core[i], i, Lmat) for i in range(nfa)])

# set b.c.
LBC, RBC = 0, 0

Heter2GSlab_data = input_data(xs_media, media, xi, geometry_type, LBC, RBC)

if __name__ == "__main__":

    import logging as lg
    lg.info("*** Solve the Rahnema 1997 problem ***")
    from FDsDiff1D import run_calc_with_RM_its, solver_options

    ritmax = 10
    CMFD, pCMFD = True, False
    slvr_opts = solver_options(ritmax=ritmax, CMFD=CMFD, pCMFD=pCMFD)
    filename = "../output/kflx_Rahnema1997_C%d_LBC%dRBC%d_I%d_itr%d" % \
               (core_config, LBC, RBC, I, ritmax)
    flx, k = run_calc_with_RM_its(Heter2GSlab_data, slvr_opts, filename)
