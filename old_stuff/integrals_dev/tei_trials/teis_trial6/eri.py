import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from integrals_utils import boys0, boys1, boys_old, taylor

def eri_ssss(A, B, C, D, aa, bb, cc, dd, coeff):
    zeta = aa + bb
    eta = cc + dd
    zetainv = 1/zeta
    etainv = 1/eta
    AmB = A-B
    CmD = C-D
    K_ab = zetainv * jax.lax.exp((-aa * bb * zetainv) * jax.lax.dot(AmB,AmB))
    K_cd = etainv * jax.lax.exp((-cc * dd * etainv) * jax.lax.dot(CmD,CmD))
    P = (aa * A + bb * B) * zetainv
    Q = (cc * C + dd * D) * etainv
    PmQ = P - Q
    boys_arg = (zeta * eta / (zeta + eta)) * jax.lax.dot(PmQ,PmQ)
    #ssss = 2 * np.pi**(10/4) * coeff * (zeta + eta)**(-1/2) * K_ab * K_cd * boys0(boys_arg)
    ssss = 2 * np.pi**(10/4) * coeff * (zeta + eta)**(-1/2) * K_ab * K_cd * (boys_arg**3 + boys_arg**2 + boys_arg)
    return ssss

def eri_psss(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Returns all 3 integrals of psss class'''
    oot_aa = 1 / (2 * aa)
    return oot_aa * jax.jacfwd(eri_ssss, 0)(A, B, C, D, aa, bb, cc, dd, coeff)
    #primals_out, px = jax.jvp(eri_ssss, (A, B, C, D, aa, bb, cc, dd, coeff), (np.array([1.,0.,0.]), np.zeros(3), np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0, 0.0,0.0))
    #primals_out, py = jax.jvp(eri_ssss, (A, B, C, D, aa, bb, cc, dd, coeff), (np.array([0.,1.,0.]), np.zeros(3), np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0, 0.0,0.0))
    #primals_out, pz = jax.jvp(eri_ssss, (A, B, C, D, aa, bb, cc, dd, coeff), (np.array([0.,0.,1.]), np.zeros(3), np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0, 0.0,0.0))
    #psss = np.array([px, py, pz])
    #return oot_aa * psss

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
A = np.array([-0.4939594255,-0.2251760374, 0.3240754142])
B = np.array([ 0.4211401526, 1.8106751596,-0.1734137286])
C = np.array([-0.5304044183, 1.5987236612, 2.0935583523])
D = np.array([ 1.9190079941, 0.0838367286, 1.4064021040])

alpha = 0.2
beta = 0.3
gamma = 0.4
delta = 0.5
coeff = 1.0
##

args = (A, B, C, D, alpha, beta, gamma, delta, coeff)
#eri_ssss(*args)
#eri_psss(*args)
#eri_psps(*args)
#eri_ppss(*args)
#eri_ppps(*args)
#eri_pppp(*args)


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
