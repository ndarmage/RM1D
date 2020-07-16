# --*-- coding:utf-8 --*--
"""
Homogeneous xs for the homogeneous 1G problem [Sood2003].


                        Problem 2  - PUa-1-0-SL
                        Problem 6  - PUb-1-0-SL
                        Problem 12 - Ua-1-0-SL
                        Problem 22 - UD2O-1-0-SL 
                        
            Cross section strucre [P2, P6, P12, P22], (P=problem)
                        
Analytical Benchmark Test Set For Criticality code Verification,
Los Alamos National Laboratory, Applied Physics (X) Divisions, X-5 Diagnostics
Applications Groupm P.O. Box 1663, MS F663, Los Alamos, NM 87545,
Avneet Sood, R. Arthur Foster, and D. Kent Parsons, 2003


"""

import numpy as np
import logging as lg

one_third = 1. / 3.
problem_for_txt = [2,6,12,22]
# Fuel
st = np.array([0.3264, 0.3264, 0.3264, 0.54628])
ss = np.array([0.225216, 0.225216, 0.248064, 0.464338])
sf = np.array([0.0816, 0.0816, 0.06528, 0.054628])
nu = np.array([3.24, 2.84, 2.7, 1.7])
chi = np.array([1.0, 1.0, 1.0, 1.0])

# definition of the (P1) diff coefficient
D = one_third / (st)
G = 1  # nb of energy groups

# Critical radius
rc_cm = np.array([1.853722, 2.256751, 2.872934, 10.371065])          # Units of [cm]
rc_mfp = np.array([0.605055, 0.73660355, 0.93772556,5.6655054562])   # Units of mean free path (mfp)



def calc_flx_inf(st, ss, chi):
    return nu*sf/(st-ss)


def calc_kinf(nsf, st=None, ss=None, chi=None, flx_inf=None):
    if flx_inf is None:
        flx_inf = calc_flx_inf(st, ss, chi)
    return np.dot(nsf, flx_inf)


if __name__ == "__main__":

    lg.info("*** Check homogeneous cross sections data ***")

    flx_inf = calc_flx_inf(st, ss)
    kinf = calc_kinf(nsf, flx_inf=flx_inf)
    # np.testing.assert_almost_equal(kinf, 1.1913539017168697, decimal=7,
    #     err_msg="kinf not verified.")
    np.testing.assert_almost_equal(kinf, 1.0783813599102687, decimal=7,
        err_msg="kinf not verified.")  # M&C article
    # np.testing.assert_allclose(flx_inf, [39.10711218,  5.85183328],
    #     err_msg="fundamental flx_inf not verified.")
    np.testing.assert_allclose(flx_inf, [38.194379,  5.697515],
        err_msg="fundamental flx_inf not verified.")  # M&C article
