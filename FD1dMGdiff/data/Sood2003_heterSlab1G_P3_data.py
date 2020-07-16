# --*-- coding:utf-8 --*--
"""
Multigroup xs for the Heterogeneous 1G problem Sood-2003.

.. note:: remind that the scattering moments are already multiplied by the
   coefficients of Lagrange polynomials if data are produced by APOLLO.
   

                        Problem 3 - PUa-H2O(1)-1-0-SL
                           Non-symmetrical 2 region


..[Sood2003] Analytical Benchmark Test Set For Criticality code Verification,
    Los Alamos National Laboratory, Applied Physics (X) Divisions, X-5 Diagnostics
    Applications Groupm P.O. Box 1663, MS F663, Los Alamos, NM 87545,
    Avneet Sood, R. Arthur Foster, and D. Kent Parsons, 2003
"""

import numpy as np
import logging as lg

one_third = 1. / 3.

# Water
stw = np.array([0.3264])
ssw = np.array([0.29376])
sfw = np.array([0.])
nuw = np.array([0.])
chiw = np.array([0.])

# Fuel PUa
# Fuel
stf = np.array([0.3264])
ssf = np.array([0.225216])
sff = np.array([0.081600])
nuf = 3.24 
chif = np.array([1])

# definition of the diff coefficient
Dw = one_third / (stw)
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
