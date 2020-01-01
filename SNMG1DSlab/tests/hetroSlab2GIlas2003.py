#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This test case runs the 2G homogeneous slab problem from [Tomatis2011]_.

.. [Tomatis2011] Tomatis, D. and Dall'Osso, A., "Application of a numerical
   transport correction in diffusion calculations", Proc. Int. Conf. on
   Mathematics and Computational Methods Applied to Nuclear Science and
   Engineering (M&C 2011), Rio de Janeiro, RJ, Brazil, May 8-12, 2011.
"""
import sys
sys.path.append('..')
from data.hetro2GIlas2003 import *
from snmg1dslab import input_data, solver_options, quad_data
import numpy as np

def MyWho():
     print([v for v in globals().keys() if not v.startswith('_')])

def get_fa(faid, ifa, L, nz=6):
   '''returns a list of lists of FA id materials in location ifa in core.
   ifa = 0,1,2,...,nfa-1 '''
   
   i = ifa*nz+1
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


def get_media(core,L,nfa=7):
   
   media = []
   for i in range(nfa):
      media += get_fa(core[i], i, L)
      
   return media
      
      
# definition of the spatial mesh
     
# materials boubdaries
DxW = 1.158 # cm, Water
DxF = 3.231 # cm, Fuel
nw = 2 # number of water segments per FA
nf = 4 # number of fuel segments per FA

nz = 6 # number of different zones per FA
nfa = 7 # total number of FAs in the core

Dxfa = np.array([DxW, DxF, DxF, DxF, DxF, DxW])
Dxmat = np.tile(Dxfa, nfa) # material width
Lmat = np.cumsum(Dxmat)
Lmat = np.insert(Lmat,0,0)

# fine mesh
nxW = 8 # no. of cells in water - can change this
nxF = 22 # no. of cells in fuel - can change this

dxW = np.linspace(0,DxW,nxW+1)
dxW = dxW[1:] - dxW[:-1]
dxF = np.linspace(0,DxF,nxF+1)
dxF = dxF[1:] - dxF[:-1]

dxfa = np.concatenate((dxW, dxF, dxF, dxF, dxF, dxW),axis=0)
dx = np.tile(dxfa, nfa) 
xi = np.cumsum(dx)
xi = np.insert(xi,0,0)
xm = (xi[1:] + xi[:-1]) / 2.

L = xi[-1]  # core width
I = len(dx) # nb of spatial cells
#xi = np.linspace(0, L, I+1)  # equidistant mesh

geometry_type = 'slab'

#xs_media = {
#    'HM':{  # homogeneous medium
#        'st': st, 'ss': ss, 'nsf': nsf, 'chi': chi, 'D': D
#    }
#}
#media = [['HM', L]]  # i.e. homogeneously filled

xs_media = {
    'WATER':{  # homogeneous medium 1
        'st': stw, 'ss': ssw, 'nsf': nsfw, 'chi': chiw
    },
    'FUELI':{  # homogeneous medium 2
        'st': st1, 'ss': ss1, 'nsf': nsf1, 'chi': chi1
    },
    'FUELII':{  # homogeneous medium 3
        'st': st2, 'ss': ss2, 'nsf': nsf2, 'chi': chi2
    },
    'FUELIII':{  # homogeneous medium 4
        'st': st3, 'ss': ss3, 'nsf': nsf3, 'chi': chi3
    }
}

if core_config == 1:
   core = [1,2,1,2,1,2,1]    
elif core_config == 2:
   core = [1,3,1,3,1,3,1]
elif core_config == 3:
   core = [1,4,1,4,1,4,1]
else:
   raise TypeError('Invalid core configuration no. [1-3].')
   
media = get_media(core,Lmat,nfa)

# set b.c.
LBC, RBC = 0, 0

# Initiate inpur data object
Hetro2GSlab_data = input_data(xs_media, media, xi, xm, dx, geometry_type, LBC, RBC)

# Initiate solver options object
slvr_opts = solver_options()

# Initiate quadratures object
N, L = 16, 0
qdata = quad_data(N, L)

if __name__ == "__main__":

    import logging as lg    
    lg.info("*** Solve the Ilas 2003 problem ***")
    from snmg1dslab import solve_sn

    flxm, k = solve_sn(Hetro2GSlab_data, slvr_opts, qdata)

    basefilen = "../output/Rahnema1997_CORE%dLBC%dRBC%d_I%d_N%d" % (core_config,
                                                        Hetro2GSlab_data.LBC,
                                                        Hetro2GSlab_data.RBC,
                                                        Hetro2GSlab_data.I,
                                                        qdata.N)
    np.save(basefilen + ".npy", np.array([k, flxm]), allow_pickle=True)
    #np.savez(basefilen + ".npz", k=k, flxm=flxm)
    
