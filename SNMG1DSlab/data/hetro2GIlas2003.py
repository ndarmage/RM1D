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

# Water
stw = np.array([0.1890, 1.4633])
ssw = np.zeros((2, 2, 2),)
ssw_0 = np.array([[0.1507, 0.], [0.0380, 1.4536]])
ssw_1 = np.array([[0., 0.], [0., 0.]])
ssw_1 *= one_third
ssw[:,:,0], ssw[:,:,1] = ssw_0, ssw_1
nsfw = np.array([0., 0.])
chiw = np.array([1., 0.])

# Fuel I
st1 = np.array([0.2263, 1.0119])
ss1 = np.zeros((2, 2, 2),)
ss1_0 = np.array([[0.2006, 0.], [0.0161, 0.9355]])
ss1_1 = np.array([[0., 0.], [0., 0.]])
ss1_1 *= one_third
ss1[:,:,0], ss1[:,:,1] = ss1_0, ss1_1
nsf1 = np.array([0.0067, 0.1241])
chi1 = np.array([1., 0.])

# Fuel II
st2 = np.array([0.2252, 0.9915])
ss2 = np.zeros((2, 2, 2),)
ss2_0 = np.array([[0.1995, 0.], [0.0156, 0.9014]])
ss2_1 = np.array([[0., 0.], [0., 0.]])
ss2_1 *= one_third
ss2[:,:,0], ss2[:,:,1] = ss2_0, ss2_1
nsf2 = np.array([0.0078, 0.1542])
chi2 = np.array([1., 0.])

# Fuel III
st3 = np.array([0.2173, 1.0606])
ss3 = np.zeros((2, 2, 2),)
ss3_0 = np.array([[0.1902, 0.], [0.0136, 0.5733]])
ss3_1 = np.array([[0., 0.], [0., 0.]])
ss3_1 *= one_third
ss3[:,:,0], ss3[:,:,1] = ss3_0, ss3_1
nsf3 = np.array([0.0056, 0.0187])
chi3 = np.array([1., 0.])

# definition of the (P1) diff coefficient
#Dw = one_third / (stw - np.sum(ssw_1, axis=1))
#D1 = one_third / (st1 - np.sum(ss1_1, axis=1))
#D2 = one_third / (st2 - np.sum(ss2_1, axis=1))
#D3 = one_third / (st3 - np.sum(ss3_1, axis=1))
# according to Rahnema-1997
#Dw = np.array([1.7639, 0.2278])
#D1 = np.array([1.4730, 0.3294])
#D2 = np.array([1.4804, 0.3362])
#D3 = np.array([1.5342, 0.3143])

G = st1.size  # nb of energy groups

core_config = 1

#if __name__ == "__main__":

    #lg.info("*** Check homogeneous cross sections data ***")

    #flx_inf = np.dot(np.linalg.inv(np.diag(st) - ss0), chi)
    #kinf = np.dot(nsf, flx_inf)
    # np.testing.assert_almost_equal(kinf, 1.1913539017168697, decimal=7,
    #     err_msg="kinf not verified.")
    #np.testing.assert_almost_equal(kinf, 1.0783813599102687, decimal=7,
    #    err_msg="kinf not verified.")  # M&C article
    # np.testing.assert_allclose(flx_inf, [39.10711218,  5.85183328],
    #     err_msg="fundamental flx_inf not verified.")
    #np.testing.assert_allclose(flx_inf, [38.194379,  5.697515],
    #    err_msg="fundamental flx_inf not verified.")  # M&C article
