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
# Material nu Sf Sc Ss St c
Pu-239 (a) 3.24 0.081600 0.019584 0.225216 0.32640 1.50
Pu-239 (b) 2.84 0.081600 0.019584 0.225216 0.32640 1.40
H2O (refl) 0.0  0.0      0.032640 0.293760 0.32640 0.90
"""

Lc = {'PUa-1-0-SL': 1.853722,
      'PUb-1-0-SL': 2.256751,
      'PUb-1-0-CY': 4.279960,
      'PUb-1-0-SP': 6.082547}  # critical lengths


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
        [np.array([float(v)]) for v in dstr.split()[2:]]
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


tlines = [l for l in Table2.split('\n')[1:-1] if not l.startswith('#')]
get_line = lambda lines, key: [l for l in lines if key in l][0]

materials = {
    'PUa': set_xs(get_line(tlines, 'Pu-239 (a)')),
    'PUb': set_xs(get_line(tlines, 'Pu-239 (b)')),
    'H2O': set_xs(get_line(tlines, 'H2O (refl)'))
}

G = 1  # nb of energy groups


def calc_flx_inf(st, ss0, chi):
    return np.dot(np.linalg.inv(np.diag(st) - ss0), chi)


def calc_kinf(nsf, st=None, ss0=None, chi=None, flx_inf=None):
    if flx_inf is None:
        flx_inf = calc_flx_inf(st, ss0, chi)
    return np.dot(nsf, flx_inf)


if __name__ == "__main__":

    lg.info("*** Check homogeneous cross sections data ***")
    ref_kinf = {'PUa': 2.612903, 'PUb': 2.290323}
    
    for m, xs in materials.items():
        if "H20" in m: continue
        lg.info(" %s-1-0-IN" % m)
        flx_inf = calc_flx_inf(xs['st'], xs['ss'], xs['chi'])
        kinf = calc_kinf(xs['nsf'], flx_inf=flx_inf)
        np.testing.assert_almost_equal(kinf, ref_kinf[m], decimal=6,
            err_msg="kinf not verified.")
