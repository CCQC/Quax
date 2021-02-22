import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from integrals_utils import boys0 

def eri_ssss(A, B, C, D, aa, bb, cc, dd, coeff):
    """ 
    Computes a primitive electron repulsion integral over 4 s-orbital Gaussian basis functions.
    See Head-Gordon, Pople J. Chem. Phys. 89, 1988 for notation hints.

    Parameters
    ----------
    A,B,C,D : 3 element np.array()
        Cartesian coordinates of four centers of ERI
    aa,bb,cc,dd : float
        Orbital exponents of basis functions on the four centers of ERI
    coeff : float 
        Normalization constant fused with contraction coefficient.
    Returns
    -------
    ssss : float
        The value of thei integral
    """
    zeta = aa + bb 
    eta = cc + dd
    K = (1/zeta) * jax.lax.exp((-aa * bb * (1/zeta)) * jax.lax.dot(A-B,A-B)) * (1/eta) * jax.lax.exp((-cc * dd * (1/eta)) * jax.lax.dot(C-D,C-D))
    boys_arg = (zeta * eta / (zeta + eta)) * np.dot((aa * A + bb * B) / zeta - (cc * C + dd * D) / eta, (aa * A + bb * B) / zeta - (cc * C + dd * D) / eta)
    ssss = 2 * np.pi**(10/4) * coeff * jax.lax.rsqrt(zeta + eta) * K * boys0(boys_arg)
    return ssss

def eri_psss(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Returns all 3 integrals of psss class'''
    oot_aa = 1 / (2 * aa)
    return oot_aa * jax.jacfwd(eri_ssss, 0)(A, B, C, D, aa, bb, cc, dd, coeff)

def eri_psps(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Returns all 9 integrals (shape=(3,3)) of psps class'''
    oot_cc = 1 / (2 * cc)
    return oot_cc * jax.jacfwd(eri_psss, 2)(A, B, C, D, aa, bb, cc, dd, coeff)

def eri_ppss(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Returns all 9 integrals (shape=(3,3)) of ppss class'''
    oot_bb = 1 / (2 * bb)
    return oot_bb * jax.jacfwd(eri_psss, 1)(A, B, C, D, aa, bb, cc, dd, coeff)

def eri_ppps(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Returns all 27 integrals (shape=(3,3,3)) of ppps class'''
    oot_cc = 1 / (2 * cc)
    return oot_cc * jax.jacfwd(eri_ppss, 2)(A, B, C, D, aa, bb, cc, dd, coeff)

def eri_pppp(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Returns all 81 integrals (shape=(3,3,3,3)) of pppp class'''
    oot_dd = 1 / (2 * dd)
    return oot_dd * jax.jacfwd(eri_ppps, 3)(A, B, C, D, aa, bb, cc, dd, coeff)

#jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(boys0)))))(1e-5)

# Create four distinct cartesian centers of atoms
#H       -0.4939594255     -0.2251760374      0.3240754142                 
#H        0.4211401526      1.8106751596     -0.1734137286                 
#H       -0.5304044183      1.5987236612      2.0935583523                 
##H        1.9190079941      0.0838367286      1.4064021040                 
#A = np.array([-0.4939594255,-0.2251760374, 0.3240754142])
#B = np.array([ 0.4211401526, 1.8106751596,-0.1734137286])
#C = np.array([-0.5304044183, 1.5987236612, 2.0935583523])
#D = np.array([ 1.9190079941, 0.0838367286, 1.4064021040])
#
#alpha = 0.2
#beta = 0.3
#gamma = 0.4
#delta = 0.5
#coeff = 1.0
##

#args = (A, B, C, D, alpha, beta, gamma, delta, coeff)

#print(eri_ssss(*args))
#print(eri_psss(*args))
#
#
#jaxpr = jax.make_jaxpr(eri_pppp)(*args)
#print(jaxpr)
#jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(eri_pppp,0),1),2),3))(*args)
#print(len(str(jaxpr).splitlines()))

#jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(eri_ssss,0),1),2),3))(*args)
#print(jaxpr)
##jaxpr = jax.make_jaxpr(jax.jacfwd(eri_pppp,0))(*args)
##jaxpr = jax.make_jaxpr(eri_pppp)(*args)
#print(jaxpr)
#
#print('psss', eri_psss(*args))
#print('psps', eri_psps(*args))


# Psi4 input
