# --*-- coding:utf-8 --*--
"""
Multigroup xs for the Heterogeneous 1G problem Sood-2003.

.. note:: remind that the scattering moments are already multiplied by the
   coefficients of Lagrange polynomials if data are produced by APOLLO.


                        Problem 4  - PUa-H2O(0.5)-1-0-SL
                        Problem 25 - UD2O-H2O(1)-1-0-SL
                        Problem 26 - UD2O-H2O(10)-1-0-SL
                        
            Cross section strucre [P4, P25, P26], (P=problem)
                        
..[Sood2003] Analytical Benchmark Test Set For Criticality code Verification,
    Los Alamos National Laboratory, Applied Physics (X) Divisions, X-5 Diagnostics
    Applications Groupm P.O. Box 1663, MS F663, Los Alamos, NM 87545,
    Avneet Sood, R. Arthur Foster, and D. Kent Parsons, 2003
"""

import numpy as np
import logging as lg

problem_for_txt = [4,25,26]
one_third = 1. / 3.

# Water
stw = np.array([0.3264, 0.54628, 0.54628])
ssw = np.array([0.29376, 0.491652, 0.491652])
sfw = np.array([0., 0., 0. ])
nuw = np.array([0., 0., 0. ])
chiw = np.array([0., 0., 0. ])


# Fuel PUa
# Fuel
stf = np.array([0.3264, 0.54628, 0.54628])
ssf = np.array([0.225216, 0.464338, 0.464338])
sff = np.array([0.081600, 0.054628, 0.054628])
nuf = np.array([3.24, 1.7, 1.7])
chif = np.array([1., 1., 1.])

# definition of the diff coefficient
Dw = one_third / (stw)
Df = one_third / (stf)


# Critical length
rcf_cm = np.array([1.317862, 9.214139, 8.428096])          # Units of [cm]
rcf_mfp = np.array([0.43015, 5.0335, 4.6041])   # Units of mean free path (mfp)

rcw_cm = np.array([1.531863, 1.830563, 18.30563])          # Units of [cm]
rcw_mfp = np.array([0.5, 1.0, 10.0])   # Units of mean free path (mfp)


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
