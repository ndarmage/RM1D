# --*-- coding:utf-8 --*--
"""
Multigroup xs for the homogeneous 2G problem [Tomatis2011]_.

.. note:: remind that the scattering moments are already multiplied by the
   coefficients of Lagrange polynomials if data are produced by APOLLO.

.. [Tomatis2011] Tomatis, D. and Dall'Osso, A., "Application of a numerical
   transport correction in diffusion calculations", Proc. Int. Conf. on
   Mathematics and Computational Methods Applied to Nuclear Science and
   Engineering (M&C 2011), Rio de Janeiro, RJ, Brazil, May 8-12, 2011.
"""

import numpy as np
import logging as lg

one_third = 1. / 3.

st = np.array([0.531150, 1.30058])
ss = np.zeros((2, 2, 1),)
ss0 = np.array([[0.504664, 0.00203884], [0.0162955, 1.19134]])
ss1 = np.array([[0., 0.], [0., 0.]]) * one_third
ss[:,:,0] = ss0  # ss[:,:,0], ss[:,:,1] = ss0, ss1
nsf = np.array([0.00715848, 0.141284])
chi = np.array([1., 0.])

# definition of the (P1) diff coefficient
D = one_third / (st - np.sum(ss1, axis=1))
G = st.size  # nb of energy groups


def calc_flx_inf(st, ss0, chi):
    return np.dot(np.linalg.inv(np.diag(st) - ss0), chi)


def calc_kinf(nsf, st=None, ss0=None, chi=None, flx_inf=None):
    if flx_inf is None:
        flx_inf = calc_flx_inf(st, ss0, chi)
    return np.dot(nsf, flx_inf)


if __name__ == "__main__":

    lg.info("*** Check homogeneous cross sections data ***")

    flx_inf = calc_flx_inf(st, ss0, chi)
    kinf = calc_kinf(nsf, flx_inf=flx_inf)
    # np.testing.assert_almost_equal(kinf, 1.1913539017168697, decimal=7,
    #     err_msg="kinf not verified.")
    np.testing.assert_almost_equal(kinf, 1.0783813599102687, decimal=7,
        err_msg="kinf not verified.")  # M&C article
    # np.testing.assert_allclose(flx_inf, [39.10711218,  5.85183328],
    #     err_msg="fundamental flx_inf not verified.")
    np.testing.assert_allclose(flx_inf, [38.194379,  5.697515],
        err_msg="fundamental flx_inf not verified.")  # M&C article
