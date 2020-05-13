#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This module contains objects and methods to handle input geometry and material
data. All solvers should use the data format given by this module.
"""

__title__ = "Specifications and tools for geometry and material data"
__author__ = "D. Tomatis"
__date__ = "07/05/2020"
__version__ = "1.0.0"

import os
import sys
import logging as lg
import numpy as np

# log file settings
logfile = os.path.splitext(os.path.basename(__file__))[0] + '.log'
# verbose output only with lg.DEBUG mode
lg.basicConfig(level=lg.INFO)  # filename = logfile


def xim(x):
    "Get the mid-points of cells defined in the input spatial mesh x"
    return (x[1:] + x[:-1]) / 2.


class input_data:
    """Geometry and material input data of the 1D problem. Possible options
    of geometry_type are slab, cylindrical and spherical. Allowed boundary
    conditions: 0-vacuum, 1-zero flux and 2-reflection. Units are in cm."""

    def __init__(self, xs_media, media, xi, geometry_type='slab',
                 LBC=0, RBC=0, per_unit_angle=True):
        self.geometry_type, self.xi = geometry_type.lower(), xi
        self.LBC, self.RBC = LBC, RBC
        self.xs_media, self.media = xs_media, media
        self.check_input()
        self.compute_cell_width()
        self.compute_mid_cell_coordinate()
        self.compute_cell_surfaces(per_unit_angle)
        self.compute_cell_volumes(per_unit_angle)

    @property
    def I(self):
        return self.xi.size - 1

    @property
    def L(self):
        return self.xi[-1]

    @property
    def BC(self):
        return self.LBC, self.RBC

    @property
    def G(self):
        return self.xs_media[next(iter(self.xs_media))]['st'].size

    @property
    def Lss(self):
        random_ss, L = self.xs_media[next(iter(self.xs_media))]['ss'], 1
        if random_ss.ndim == 3:
            L = max([self.xs_media[m]['ss'].shape[-1] for m in self.xs_media])
        return L - 1
    
    def compute_cell_width(self):
        self.Di = self.xi[1:] - self.xi[:-1]
    
    def compute_mid_cell_coordinate(self):
        self.xim = xim(self.xi)
    
    def compute_cell_surfaces(self, per_unit_angle=True):
        self.Si = compute_cell_surfaces(self.xi, geo=self.geometry_type,
                                        per_unit_angle=per_unit_angle)
    
    def compute_cell_volumes(self, per_unit_angle=True):
        self.Vi = compute_cell_volumes(self.xi, geo=self.geometry_type,
                                       per_unit_angle=per_unit_angle)

    def check_input(self):
        lg.info("Geometry type is " + self.geometry_type)
        if (self.geometry_type != 'slab') and \
           (self.geometry_type != 'cylinder') and \
           (self.geometry_type != 'cylindrical') and \
           (self.geometry_type != 'spherical') and \
           (self.geometry_type != 'sphere'):
            raise ValueError('Unknown geometry_type ' + self.geometry_type)
        if not isinstance(self.LBC, int):
            raise TypeError('LBC must be integer.')
        if (self.LBC < 0) and (self.LBC > 2):
            raise ValueError('Check LBC, allowed options ')
        if not isinstance(self.RBC, int):
            raise TypeError('RBC must be integer.')
        if (self.RBC < 0) and (self.RBC > 2):
            raise ValueError('Check RBC, allowed options ')
        if (self.geometry_type != 'slab') and (self.LBC != 2):
            raise ValueError('Curvilinear geometries need LBC = 2.')
        if not isinstance(self.xs_media, dict):
            raise TypeError('The input xs_media is not a dictionary.')
        if not isinstance(self.media, list):
            raise TypeError('The input media is not a list.')
        media_set = set(m[0] for m in self.media)
        xs_media_set = set(self.xs_media.keys())
        if xs_media_set.union(media_set) != xs_media_set:
            raise ValueError('xs media dict has missing keys, check media.')
        rbnd = [m[1] for m in self.media]
        if sorted(rbnd) != rbnd:
            raise ValueError('media list must be in order from left to right!')
        if not np.isclose(max(rbnd), self.L):
            raise ValueError('Please check the right bounds of media (>L?)')

    def __str__(self):
        s = ["Geometry type is " + self.geometry_type,
             "Boundary conditions: (left) LBC=%d and (right) RBC=%d." %
              (self.LBC, self.RBC),
             "B.C. legend: 0-vacuum, 1-zero flux and 2-reflection.",
             "Number of energy groups: %d" % self.G,
             "Number of spatial cells: %d" % self.I,
             "Spatial mesh xi\n" + str(self.xi),
             "Media list:\n" + str(self.media),
             " with xs:\n" + str(self.xs_media)]
        return '\n'.join(s)


def to_str(v, fmt='%.13g'):
    return ', '.join([fmt % i for i in v]) + '\n'


def compute_cell_volumes(xi, geo=None, per_unit_angle=True):
    """Compute the volumes of the cells in the mesh xi. These volumes
    are per unit of transversal surface of the slab or per unit of
    angle in the other frame (azimuthal angle in the cylinder or
    per cone unit in the sphere) if per_unit_angle is true. In this case
    the real volumes in the cylinder must be multiplied by 2*np.pi,
    or by 4*np.pi for the sphere."""
    # V = xi[1:] - xi[:-1]  # = Di for the slab as default case
    V = np.diff(xi)  # same as above, but possibly vectorized
    if geo != 'slab':
        xm = xim(xi)
        if 'cylind' in geo:
            V *= (xm if per_unit_angle else 2 * xm)
        elif 'spher' in geo:
            cm = (4. * xm**2 - xi[1:] * xi[:-1]) / 3.
            V *= (cm if per_unit_angle else 4 * cm)
            # V *= (r[1:]**2 + r[1:]*r[:-1] + r[:-1]**2) / 3.
        else:
            raise ValueError("Unknown geometry type " + geo)
        if not per_unit_angle:
            V *= np.pi
    return V


def compute_cell_surfaces(xi, geo=None, per_unit_angle=True):
    "Compute the cell outer surfaces with the input 1d mesh xi."
    S = np.ones(xi.size)
    if geo != 'slab':
        S[0] = 0  # selected value to raise singularity when dividing by itself
        if 'cylind' in geo:
            S[1:] = (xi[1:] if per_unit_angle else 2 * xi[1:])
        elif 'spher' in geo:
            S[1:] = (xi[1:]**2 if per_unit_angle else 4 * xi[1:]**2)
        else:
            raise ValueError("Unknown geometry type " + geo)
        if not per_unit_angle:
            S *= np.pi
    return S


def vol_averaged_at_interface(f, Vi):
    "Compute surface quantities by volume averaging (slab geometry)."
    G, I = f.shape
    Im1 = I - 1

    # fb, volume-average quantity on cell borders
    fb = (f[:, 1: ] * np.tile(Vi[1: ], G).reshape(G, Im1) +
          f[:, :-1] * np.tile(Vi[:-1], G).reshape(G, Im1))
    return fb / np.tile(Vi[1:] + Vi[:-1], G).reshape(G, Im1)


def unfold_xs(input_data, diff_calc=True):
    """Set up the spatial mesh with cross sections data. Scattering xs ss can
    be limited to the only isotropic term for diffusion calculations."""
    I, G, Lss = input_data.I, input_data.G, input_data.Lss
    Lssp1 = Lss + 1
    xs_media, media = input_data.xs_media, input_data.media
    xm = xim(input_data.xi)
    ss, st = np.zeros((G, G, Lssp1, I),), np.zeros((G, I),)
    nsf, chi, D = np.zeros_like(st), np.zeros_like(st), np.zeros_like(st)

    lbnd = 0.
    for m in media:
        media_name, rbnd = m
        idx = (lbnd < xm) & (xm < rbnd)
        n = sum(idx)  # nb. of True values or non-zero elements
        st[:, idx] = np.tile(xs_media[media_name]['st'], (n, 1)).T
        nsf[:, idx] = np.tile(xs_media[media_name]['nsf'], (n, 1)).T
        chi[:, idx] = np.tile(xs_media[media_name]['chi'], (n, 1)).T
        D[:, idx] = np.tile(xs_media[media_name]['D'], (n, 1)).T
        Lmed = xs_media[media_name]['ss'].shape[-1]
        if Lmed > Lssp1:
            raise ValueError("Media %s has ss with L > %d!" %
                             (media_name, Lss))
        tmp = np.tile(xs_media[media_name]['ss'].flatten(), (n, 1)).T
        ss[:, :, :, idx] = tmp.reshape(G, G, Lssp1, n)
        lbnd = rbnd

    # --------------------------------------------------------------------------
    # option 1.
    # determine the diffusion coefficient on the cell borders (w/o boundaries)
    Db = vol_averaged_at_interface(D, input_data.Vi)
    # --------------------------------------------------------------------------
    # # option 2.
    # # use the definition of the diffusion coefficient at the boundary given
    # # the current approximated by 1st order finite differences at both sides
    # # an interface
    # DioD = 1 / (D / input_data.Di)  # to exploit element-wise division
    # Db = 2 / (DioD[:, 1:] + DioD[:, :-1]) * xim(input_data.Di)
    # --------------------------------------------------------------------------
    # Db at the boundaries is considered equal to the one of the nearest cell
    Db = np.insert(Db, 0, D[:, 0], axis=1)
    Db = np.insert(Db, I, D[:,-1], axis=1)
#    # test odCMFD
#    Vm = np.insert(xim(input_data.Vi), 0, input_data.Vi[0])
#    Vm = np.insert(Vm, -1, input_data.Vi[-1])
#    tau_max = np.max(input_data.Vi * st)
#    opt_theta_by_Zhu = opt_theta(tau_max)
#    # NOTE: our max_tau values are much smaller than what is used in MOC 
#    # transport calculations and so opt_theta is almost always zero.
#    lg.info("Optimal theta for odCMFD = %.3f with max(tau) = %.5f" % \
#        (opt_theta_by_Zhu, tau_max))
#    Db += opt_theta_by_Zhu * np.tile(Vm, (G, 1))
    xs = [st, ss, chi, nsf]
    if diff_calc:
        xs.append(Db)
    return xs


def equivolume_mesh(I, a=0, b=1, geometry_type="cylinder"):
    "Make an equivolume mesh of I elements within [a, b]."
    if I <= 0:
        raise InputError("Invalid nb. of mesh elements")
    if a >= b:
        raise InputError("Invalid input bounds.")
    Vi = (b - a) / float(I)
    if "cylind" in geometry_type:
        Vi *= b + a
    elif "spher" in geometry_type:
        Vi *= b**2 + a * b + a**2
    elif geometry_type != "slab":
        raise InputError("Unsupported geometry type")
    
    r = np.cumsum(Vi * np.ones(I))
    if "cylind" in geometry_type:
        r = np.sqrt(r - a**2)
    elif "spher" in geometry_type:
        r = np.cbrt(r - a**3)
    
    return np.insert(r, 0, a)


def geomprogr_mesh(N=None, a=0, L=None, Delta0=None, ratio=None):
    """Compute a sequence of values according to a geometric progression.
    Different options are possible with the input number of intervals in the
    sequence N, the length of the first interval Delta0, the total length L
    and the ratio of the sought geometric progression. Three of them are 
    requested in input to find a valid sequence. The sequence is drawn within
    the points a and b."""
    
    if list(locals().values()).count(None) > 1:
        raise ValueError('Insufficient number of input data for a sequence')
    if ratio is not None:
        if (ratio < 0):
            raise ValueError('negative ratio is not valid')
    if L is not None:
        if (L < 0):
            raise ValueError('negative total length is not valid')
    if Delta0 is not None:
        if (Delta0 < 0):
            raise ValueError('negative length of the 1st interval is not valid')
    if N is not None:
        if (N < 0):
            raise ValueError('negative number of intervals is not valid')
    
    if N is None:
        if ratio < 1:
            N = np.log(1 - L / Delta0 * (1 - ratio)) / np.log(ratio)
        else:
            N = np.log(1 + L / Delta0 * (ratio - 1)) / np.log(ratio)
    elif L is None:
        if ratio < 1:
            L = Delta0 * (1 - ratio**N) / (1 - ratio)
        else:
            L = Delta0 * (ratio**N - 1) / (ratio - 1)
    elif Delta0 is None:
        if not np.isclose(ratio, 1):
            Delta0 = L * (1 - ratio) / (1 - ratio**N)
        else:
            Delta0 = L / float(N)
    elif ratio is None:
        f = lambda q: q**N - L / Delta0 * q + L / Delta0 - 1
        x = L / float(N)
        if Delta0 > x:
            ratio = brentq(f, 0, 1 - 1.e-6)
        elif Delta0 < x:
            ratio = brentq(f, 1 + 1.e-6, 20)
        else:
            ratio = 1
    
    if np.isclose(ratio, 1):
        r = np.linspace(0, L, N + 1)
    else:
        r = np.insert(np.full(N - 1, ratio), 0, 1)
        r = np.cumprod(r) * Delta0
        r = np.insert(np.cumsum(r), 0, 0)
    
    return r + a


def opl(j, i, tau):
    """Calculate the (dimensionless) optical length between j-1/2 and i+1/2 in
    the slab."""
    # if j > i, the slicing returns an empty array, and np.sum returns zero.
    return np.sum(tau[j:i+1])

    
# def opl(j, i, Ptg):  # first version implemented in FD1dDiff.py
    # "Calculate the (dimensionless) optical length between j-1/2 and i+1/2."
    # # if j > i, the slicing returns an empty array, and np.sum returns zero.
    # return np.sum(Ptg[j:i+1])