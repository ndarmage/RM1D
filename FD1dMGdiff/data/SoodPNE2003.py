#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
Data by Sood et al. used to verify MCNP and DANTSYS in the past [Sood2003]_.

.. [Sood2003] Sood, A., Forster, R. A., & Parsons, D. K. (2003). Analytical
              benchmark test set for criticality code verification. Progress
              in Nuclear Energy, 42(1), 55-106.
"""

import numpy as np
import logging as lg

one_third = 1. / 3.


Table2 = \
"""
# Material nu   Sf       Sc       Ss       St      c
Pu-239 (a) 3.24 0.081600 0.019584 0.225216 0.32640 1.50
Pu-239 (b) 2.84 0.081600 0.019584 0.225216 0.32640 1.40
H2O (refl) 0.0  0.0      0.032640 0.293760 0.32640 0.90
"""

Table9 = \
"""
# Material nu       Sf       Sc       Ss       St      c
U-235  (a) 2.70     0.065280 0.013056 0.248064 0.32640 1.30
U-235  (b) 2.797101 0.065280 0.013056 0.248064 0.32640 1.3194202
U-235  (c) 2.707308 0.065280 0.013056 0.248064 0.32640 1.3014616
U-235  (d) 2.679198 0.065280 0.013056 0.248064 0.32640 1.2958396
"""

# water from Table 13 has same c, but different st
Table13 = \
"""
# Material nu   Sf       Sc       Ss       St      c
U-D2O      1.7  0.054628 0.027314 0.464338 0.54628 1.02
H2O (refl) 0.0  0.0      0.054628 0.491652 0.54628 0.90
"""

Table17 = \
"""
# Material nu   Sf         Sc         Ss          St          c
U-235 (e)  2.50 0.06922744 0.01013756 0.328042    0.407407    1.230
Fe (refl)  0.0  0.0        0.00046512 0.23209488  0.23256     0.9980
Na  (mod)  0.0  0.0        0.0        0.086368032 0.086368032 1.0
"""

Lc = {'PUa-1-0-SL': 1.853722,
      'PUb-1-0-SL': 2.256751,
      'PUb-1-0-CY': 4.279960,
      'PUb-1-0-SP': 6.082547,
      'Ua-1-0-SL' : 2.872934,
      'Ua-1-0-CY' : 5.284935,
      'Ua-1-0-SP' : 7.428998,
      'UD2O-1-0-SL': 10.371065,
      'UD2O-1-0-CY': 16.554249,
      'UD2O-1-0-SP': 22.017156,
      'PUa-H2O(1)-1-0-SL': 4.542175,
      'PUa-H2O(0.5)-1-0-SL': 2.849725,
      'PUb-H2O(1)-1-0-CY': 6.461335,
      'PUb-H2O(10)-1-0-CY': 33.714829,
      'Ub-H2O(1)-1-0-SP': 9.191176,
      'Uc-H2O(2)-1-0-SP': 12.2549,
      'Ud-H2O(3)-1-0-SP': 15.318626,
      'UD2O-1-0-SL': 10.371065, 
      'UD2O-1-0-CY': 16.554249, 
      'UD2O-1-0-SP': 22.017156,
      'UD2O-H2O(1)-1-0-SL': 11.044702,
      'UD2O-H2O(10)-1-0-SL': 26.733726,
      'UD2O-H2O(1)-1-0-CY': 17.227479,
      'UD2O-H2O(10)-1-0-CY': 32.912288,
      'Ue-Fe-Na-1-0-SL' : 7.757166007, # total critical length 
      }  # critical lengths


def get_geoid(geo):
    if geo == 'slab':
        g = 'SL'
    elif 'cylind' in geo:
        g = 'CY'
    elif 'spher' in geo:
        g = 'SP'
    else:
        raise ValueError('unknown input geometry type')
    return g


def set_media(m, L, name):
    if isinstance(L, (tuple, list, np.ndarray)):
        # hetergeneous media problem
        if len(name) != len(L):
            raise ValueError('input args of different size')
        if not isinstance(m, dict):
            raise ValueError('dict of materials needed as 1st input arg')
        media, xs_media = [], dict()
        for i, n in enumerate(name):
            xs_media[n] = {
                'st': m[n]['st'], 'ss': m[n]['ss'], 'nsf': m[n]['nsf'],
                'chi': m[n]['chi'], 'D': m[n]['D']
            }
            media.append([n, L[i]])
    else:
        # homogeneous medium problem
        xs_media = {name:{
            'st': m['st'], 'ss': m['ss'], 'nsf': m['nsf'],
            'chi': m['chi'], 'D': m['D']}
        }
        media = [[name, L]]
    return xs_media, media


def set_xs(dstr):
    d = dict()
    d['nu'], d['sf'], d['sc'], d['ss'], d['st'], d['c'] = \
        [np.array([float(v)]) for v in dstr.split()[-6:]]
    d['chi'], d['nsf'] = np.ones(1), d['nu'] * d['sf']
    d['sa'] = d['sc'] + d['sf']  # = d['st'] - d['ss']
    d['ss'] = np.ones((1,1,1),) * d['ss']
    d['kinf'] = (d['nsf'] / d['sa'])[0]
    # definition of the (P1) diff coefficient with isotropic scat.
    # for k, v in d.items():
        # v /= d['st']
    d['D'] = one_third / d['st']
    return d


def BM2(m, k=1):
    "return the material buckling in the 1g homogeneous problem"
    nsf, sa, D = m['nsf'][0], m['sa'][0], m['D'][0]
    if (nsf.size != sa.size) or (nsf.size != D.size):
        raise ValueError('Input data have different size')
    if nsf.size != 1:
        raise ValueError('One-group data are needed')
    return (nsf/k - sa) / D


def diffk_ref(BG2, m):
    "Return the multiplication factor provided the geometrical buckling"
    nsf, sa, D = m['nsf'][0], m['sa'][0], m['D'][0]
    L2, kinf = D / sa, nsf / sa
    PNL = 1 / (1 + L2 * BG2)  # non leakage probability
    return kinf * PNL
    

def analytical_sol(geo, L):
    # only problems with vacuum boundary condition!
    if geo == 'centered-slab':
        BG = brentq(lambda b: np.tan(b*L) - 1 / extrap_len / b,
                    1.e-5, .5 * np.pi / L - 1.e-5)
        diffsol_ref = lambda x: np.cos(BG * x)
    elif geo == 'full-slab' or geo == 'slab':
        BG = brentq(lambda b: np.tan(b*L) - 1 / extrap_len / b,
            1.e-5, .5 * np.pi / L - 1.e-5)
        coef = 1 / np.tan(BG * L)  # for symmetric distribution 
        diffsol_ref = lambda x: np.sin(BG * x) + coef * np.cos(BG * x)
    elif geo == 'cylinder':
        BG = brentq(lambda b: J0(b*L) - extrap_len * b * J1(b*L),
            0, 2.4048255577 / L)
        diffsol_ref = lambda x: J0(BG * x)
    elif geo == 'sphere':
        BG = brentq(lambda b: b*L - (1 - L / extrap_len) * np.tan(b*L),
            .5 * np.pi / L, np.pi / L)
        diffsol_ref = lambda x: np.sin(BG * x) / x
    else:
        raise ValueError(geo + ' is not supported yet.')
    return diffsol_ref, BG


tlines = lambda Tab: [l for l in Tab.split('\n')[1:-1]
                      if not l.startswith('#')]
get_line = lambda lines, key: [l for l in lines if key in l][0]

materials = {
    'PUa': set_xs(get_line(tlines(Table2), 'Pu-239 (a)')),
    'PUb': set_xs(get_line(tlines(Table2), 'Pu-239 (b)')),
    'H2O': set_xs(get_line(tlines(Table2), 'H2O (refl)')),
    'Ua': set_xs(get_line(tlines(Table9), 'U-235  (a)')),
    'Ub': set_xs(get_line(tlines(Table9), 'U-235  (b)')),
    'Uc': set_xs(get_line(tlines(Table9), 'U-235  (c)')),
    'Ud': set_xs(get_line(tlines(Table9), 'U-235  (d)')),
    # 'H2O': set_xs(get_line(tlines(Table13), 'H2O (refl)')),
    'UD2O': set_xs(get_line(tlines(Table13), 'U-D2O')),
    'Ue': set_xs(get_line(tlines(Table17), 'U-235 (e)')),
    'Fe': set_xs(get_line(tlines(Table17), 'Fe (refl)')),
    'Na': set_xs(get_line(tlines(Table17), 'Na  (mod)'))
}

geoms = ["slab", "cylinder", "sphere"]

G = 1  # nb of energy groups


def calc_flx_inf(st, ss0, chi):
    return np.dot(np.linalg.inv(np.diag(st) - ss0), chi)


def calc_kinf(nsf, st=None, ss0=None, chi=None, flx_inf=None):
    if flx_inf is None:
        flx_inf = calc_flx_inf(st, ss0, chi)
    return np.dot(nsf, flx_inf)


if __name__ == "__main__":

    lg.info("*** Check homogeneous cross sections data ***")
    ref_kinf = {'PUa': 2.612903,  # problem 1
                'PUb': 2.290323,  # problem 5
                'Ua': 2.250000,   # problem 11
                'Ub': 2.330917,   # problem 15
                'Uc': 2.256090,   # problem 17 (communicated 2.256083)
                'Ud': 2.232665,   # problem 19 (communicated 2.232667)
                'UD2O': 1.133333,  # problem 21
                'Ue': 2.1806667  # problem 29
               }
    
    for m, xs in materials.items():
        if "H2O" in m: continue
        case = " %s-1-0-IN" % m
        lg.info(case)
        flx_inf = calc_flx_inf(xs['st'], xs['ss'], xs['chi'])
        kinf = calc_kinf(xs['nsf'], flx_inf=flx_inf)
        np.testing.assert_almost_equal(kinf, ref_kinf[m], decimal=6,
            err_msg="kinf of case %s not verified." % case)
