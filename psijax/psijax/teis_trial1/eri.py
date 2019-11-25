import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from integrals_utils import lower_take_mask, boys0

def eri_ssss(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4):
    coeff = c1 * c2 * c3 * c4
    zeta = aa + bb
    eta = cc + dd
   
    factor = np.sqrt(2)*np.pi**(5/4)
    K_ab = (factor / zeta) * np.exp((-aa * bb / zeta) * np.dot(A-B,A-B))
    K_cd = (factor / eta) * np.exp((-cc * dd / eta) * np.dot(C-D,C-D))

    P = (aa * A + bb * B) / zeta
    Q = (cc * C + dd * D) / eta

    boys_arg = (zeta * eta / (zeta + eta)) * np.dot(P-Q,P-Q)
    ssss = coeff * (zeta + eta)**(-1/2) * K_ab * K_cd * boys0(boys_arg)
    return ssss

def eri_psss(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4):
    '''Returns all 3 integrals of psss class'''
    oot_aa = 1 / (2 * aa)
    return oot_aa * jax.jacrev(eri_ssss, 0)(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4)

def eri_psps(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4):
    '''Returns all 9 integrals (shape=(3,3)) of psps class'''
    oot_cc = 1 / (2 * cc)
    return oot_cc * jax.jacrev(eri_psss, 2)(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4)

def eri_ppss(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4):
    '''Returns all 9 integrals (shape=(3,3)) of ppss class'''
    oot_bb = 1 / (2 * bb)
    return oot_bb * jax.jacrev(eri_psss, 1)(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4)

def eri_ppps(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4):
    '''Returns all 27 integrals (shape=(3,3,3)) of ppps class'''
    oot_cc = 1 / (2 * cc)
    return oot_cc * jax.jacfwd(eri_ppss, 2)(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4)

def eri_pppp(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4):
    '''Returns all 81 integrals (shape=(3,3,3,3)) of pppp class'''
    oot_dd = 1 / (2 * dd)
    return oot_dd * jax.jacfwd(eri_ppps, 3)(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4)

#A = np.array([0.0,0.0,-0.849220457955])
#B = A
#C = np.array([0.0,0.0,0.849220457955])
#D = C
#alpha = 0.5
#beta = 0.5
#gamma = 0.4
#delta = 0.4
#c1,c2,c3,c4 = 1.0,1.0,1.0,1.0
#
#args = (A, B, C, D, alpha, beta, gamma, delta, c1, c2, c3, c4)
#print('psss', eri_psss(*args).shape)
#print('psps', eri_psps(*args).shape)
#print('ppss', eri_ppss(*args).shape)
#print('ppps', eri_ppps(*args).shape)
#print('pppp', eri_pppp(*args).shape)
#print("Running a bunch")
#
#def body(i):
#    w = eri_ssss(*args)
#    a = eri_psss(*args)
#    b = eri_psps(*args)
#    c = eri_ppss(*args)
#    d = eri_ppps(*args)
#    e = eri_pppp(*args)
#    return e
#
#res = jax.lax.map(body, np.arange(10000))
#
#
#print("done")
