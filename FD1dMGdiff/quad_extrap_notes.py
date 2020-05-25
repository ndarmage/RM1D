def quadratic_fit_3p(rs, fs):
    """
    fit a function by quadratic polynomial passing for three points

    f(r) = (
      f0*(r**2*r1 - r**2*r2 - r*r1**2 + r*r2**2 + r1**2*r2 - r1*r2**2)
    - f1*(r**2*r0 - r**2*r2 - r*r0**2 + r*r2**2 + r0**2*r2 - r0*r2**2)
    + f2*(r**2*r0 - r**2*r1 - r*r0**2 + r*r1**2 + r0**2*r1 - r0*r1**2)
    ) / ((r0 - r1)*(r0 - r2)*(r1 - r2))
    """
    r0, r1, r2 = rs
    f0, f1, f2 = fs
    rp2 = r**2
    r0p2, r1p2, r2p2 = r0**2, r1**2, r2**2
    d = (r0 - r1)*(r0 - r2)*(r1 - r2)
     
    # f(r) = (
    #   f0*((rp2 + r1*r2)*(r1 - r2) - r*(r1p2 + r2p2))
    # - f1*((rp2 + r0*r2)*(r0 - r2) - r*(r0p2 + r2p2))
    # + f2*((rp2 + r0*r1)*(r0 - r1) - r*(r0p2 + r1p2))
    # ) / ((r0 - r1)*(r0 - r2)*(r1 - r2))
    
    # f(r) = c0(r) * f0 + c1(r) * f1 + c2(r) * f2
     
    # coefficients to use in the system eqs.
    c0 = ((rp2 + r1*r2)*(r1 - r2) - r*(r1p2 + r2p2)) / d
    c1 = ((rp2 + r0*r2)*(r0 - r2) - r*(r0p2 + r2p2)) / -d
    c2 = ((rp2 + r0*r1)*(r0 - r1) - r*(r0p2 + r1p2)) / d
    return c0, c1, c2


# .. note:: the following function was implemented to get more accurate flux at
#           the vacuum boundary but it does not work yet as expected. Possible
#           bug. More investigations are necessary. 

def quadratic_fit_zD(rs, fs, zD):
    """
    fit a function by quadratic polynomial passing for two points and with
    prescribed derivative in a distinct point.
    >>> f  = a*r**2 + b*r + c
    >>> fp = 2*a*r + b
    >>> zD = symbols('zD')
    >>> syseqs = [Eq(f.subs(r, r0), zD * fp.subs(r, r0)), Eq(f.subs(r, r1), f1),
    ... Eq(f.subs(r, r2), f2)]
    >>> sol = solve(syseqs, (a, b, c))
    >>> g = sol[a]*r**2 + sol[b]*r + sol[c]
    """
    r0, r1, r2 = rs
    f1 ,f2 = fs
    rp2 = r**2
    r0p2, r1p2, r2p2 = r0**2, r1**2, r2**2
    d = (r1 - r2)*(r0*(r0 - r1 - r2) + r1*r2 + (r1 - 2*r0 + r2)*zD)

    # f(r) = (
    # - f1*(r**2*r0 - r**2*r2 - r**2*zD - r*r0**2 + 2*r*r0*zD + r*r2**2
    #       + r0**2*r2 - r0*r2**2 - 2*r0*r2*zD + r2**2*zD)
    # + f2*(r**2*r0 - r**2*r1 - r**2*zD - r*r0**2 + 2*r*r0*zD + r*r1**2
    #       + r0**2*r1 - r0*r1**2 - 2*r0*r1*zD + r1**2*zD)
    # ) / ((r1 - r2)*(r0**2 - r0*r1 - r0*r2 - 2*r0*zD + r1*r2
    #                                         + r1*zD + r2*zD))

    # f(r) = (
    # - f1*(rp2*(r0 - r2 - zD) - r*(r0*(r0 + 2*zD) + r2p2)
    #       + r2*(r0*(r0 - r2 - 2*zD) + r2*zD))
    # + f2*(rp2*(r0 - r1 - zD) - r*(r0*(r0 + 2*zD) + r1p2)
    #       + r1*(r0*(r0 - r1 - 2*zD) + r1*zD))
    # ) / ((r1 - r2)*(r0**2 - r0*r1 - r0*r2 - 2*r0*zD + r1*r2
    #                                         + r1*zD + r2*zD))
    # f(r) = c1(r) * f1 + c2(r) * f2
    
    # coefficients to use in the system eqs.
    c1 = (rp2*(r0 - r2 - zD) - r*(r0*(r0 + 2*zD) + r2p2)
          + r2*(r0*(r0 - r2 - 2*zD) + r2*zD)) / -d
    c2 = (rp2*(r0 - r1 - zD) - r*(r0*(r0 + 2*zD) + r1p2)
          + r1*(r0*(r0 - r1 - 2*zD) + r1*zD)) / d
    return c1, c2







fprime(r) = (  # fp, first derivative
   f1*(-2*r*r0 + 2*r*r2 + 2*r*zD + r0**2 - 2*r0*zD - r2**2)
 + f2*(2*r*r0 - 2*r*r1 - 2*r*zD - r0**2 + 2*r0*zD + r1**2)
) / ((r1 - r2)*(r0**2 - r0*r1 - r0*r2 - 2*r0*zD + r1*r2
                                        + r1*zD + r2*zD))