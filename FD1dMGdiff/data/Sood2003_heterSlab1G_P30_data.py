# --*-- coding:utf-8 --*--
"""
Multigroup xs for the Heterogeneous 1G problem Sood-2003.

.. note:: remind that the scattering moments are already multiplied by the
   coefficients of Lagrange polynomials if data are produced by APOLLO.
   

                        Problem 30 - UD2O-H2O(10)-1-0-SL
                             Symmetrical 3 regions


..[Sood2003] Analytical Benchmark Test Set For Criticality code Verification,
    Los Alamos National Laboratory, Applied Physics (X) Divisions, X-5 Diagnostics
    Applications Groupm P.O. Box 1663, MS F663, Los Alamos, NM 87545,
    Avneet Sood, R. Arthur Foster, and D. Kent Parsons, 2003
"""

import numpy as np
import logging as lg

one_third = 1. / 3.

# Na
stNa = np.array([0.086368032])
ssNa = np.array([0.086368032])
sfNa = np.array([0.])
nuNa = np.array([0.])
chiNa = np.array([0.])

# Clad
stc = np.array([0.23256])
ssc = np.array([0.23209488])
sfc = np.array([0.])
nuc = np.array([0.])
chic = np.array([0.])

# Fuel
stf = np.array([0.407407])
ssf = np.array([0.328042])
sff = np.array([0.06922744])
nuf = 2.5
chif = np.array([1])

# definition of the diff coefficient
DNa = one_third / (stNa)
Dc = one_third / (stc)
Df = one_third / (stf)


G = 1  # nb of energy groups

if __name__ == "__main__":

    lg.info("*** Check homogeneous cross sections data ***")

    flx_inf = np.dot(np.linalg.inv(np.diag(st) - ss0), chi)
    kinf = np.dot(nsf, flx_inf)
    # np.testing.assert_almost_equal(kinf, 1.1913539017168697, decimal=7,
    #     err_msg="kinf not verified.")
    np.testing.assert_almost_equal(kinf, 1.0783813599102687, decimal=7,
        err_msg="kinf not verified.")  # M&C article
    # np.testing.assert_allclose(flx_inf, [39.10711218,  5.85183328],
    #     err_msg="fundamental flx_inf not verified.")
    np.testing.assert_allclose(flx_inf, [38.194379,  5.697515],
        err_msg="fundamental flx_inf not verified.")  # M&C article
