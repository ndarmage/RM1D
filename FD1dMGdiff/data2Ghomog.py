import numpy as np

one_third = 1. / 3.

# multigroup xs for the homogeneous 2G problem
# st = np.array([0.52610422, 1.25161422])
# ss = np.zeros((2, 2, 2),)
# ss0 = np.array([[0.5002712, 0.00175241], [0.01537974, 1.14883323]])
# ss1 = np.array([[0.66171011, 0.00236677], [0.01330484, 0.91455725]])
# multigroup xs for the homogeneous 2G problem (M&C article)
st = np.array([0.531150, 1.30058])
ss = np.zeros((2, 2, 2),)
ss0 = np.array([[0.504664, 0.00203884], [0.0162955, 1.19134]])
ss1 = np.array([[0., 0.], [0., 0.]])
# Nuclear lib integrates already the coefficient of Lagrange polynomial
ss1 *= one_third
ss[:,:,0], ss[:,:,1] = ss0, ss1
nsf = np.array([0.00715848, 0.141284])
chi = np.array([1., 0.])

# definition of the (P1) diff coefficient
D = one_third / (st - np.sum(ss1, axis=1))

# set b.c.
LBC, RBC = 2, 0

# definition of the spatial mesh
L = 21.5 / 2.  # slab width, equal to half pitch of a fuel assembly
I = 20  # nb of spatial cells
G = st.size  # nb of energy groups
xi = np.linspace(0, L, I+1)  # equidistant mesh
# xm = (xi[1:] + xi[:-1]) / 2. # mid-points

# tolerance on the fission rates during outer iterations
ritmax = 1  # set to 1 to skip the Ronen iterations
toll = 1.e-6
oitmax = 50  # max nb of outer iterations
iitmax = 100  # max nb of inner iterations
itsmax = oitmax, iitmax

# choose the type of CMFD scheme
pCMFD = True  # classic CMFD scheme is False

# choose geometry
geometry_type = 'slab' # 'cylindrical'/'spherical'

# reflective BC fix
fix_reflection_by_flx2 = True

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
