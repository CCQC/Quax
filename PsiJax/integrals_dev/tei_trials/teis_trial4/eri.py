import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from integrals_utils import boys0, boys1, boys2, boys3, boys_old, taylor

#def boys1(arg):
#    return boys0(arg)
#def boys2(arg):
#    return boys0(arg)

def base(A, B, C, D, aa, bb, cc, dd, coeff):
    zeta = jax.lax.add(aa,bb) 
    eta = jax.lax.add(cc,dd)
    K_ab = jax.lax.mul(jax.lax.div(1.,zeta),jax.lax.exp(jax.lax.mul((jax.lax.mul(jax.lax.mul(-aa,bb),jax.lax.div(1.,zeta))),jax.lax.dot(jax.lax.sub(A,B),jax.lax.sub(A,B)))))
    K_cd = jax.lax.mul(jax.lax.div(1.,eta),jax.lax.exp(jax.lax.mul((jax.lax.mul(jax.lax.mul(-cc,dd),jax.lax.div(1.,eta))),jax.lax.dot(jax.lax.sub(C,D),jax.lax.sub(C,D)))))
    P = (jax.lax.add(jax.lax.mul(aa,A),jax.lax.mul(bb,B))) * (jax.lax.div(1.,zeta))
    Q = (jax.lax.add(jax.lax.mul(cc,C),jax.lax.mul(dd,D))) * (jax.lax.div(1.,eta))
    W = (zeta * P + eta * Q) / (zeta + eta)
    boys_arg = (zeta * eta / (zeta + eta)) * jax.lax.dot(P-Q,P-Q)
    ssss_0 = 2 * np.pi**(10/4) * coeff * (zeta + eta)**(-1/2) * K_ab * K_cd 
    return A,B,C,D, zeta, eta, P, Q, W, boys_arg, ssss_0

# Call structure:
# args = base(A, B, C, D, aa, bb, cc, dd, coeff)
# ijkl = eri_ijkl(*args)
# NOTE some arguments are not used most of the time

def eri_ssss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0):# checked!
    #TODO TEMP
    return ssss_0 * boys0(boys_arg)
    # Simulate Obara-Saika-esque terms
    #return ssss_0 * (P-W)[0] *  boys0(boys_arg) + (P-W)[0] * ssss_0 * boys1(boys_arg) + ssss_0 * (P-W)[0] *  boys2(boys_arg) + ssss_0 *(P-W)[0] *  boys3(boys_arg)
    
def eri_psss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0):# checked!
    psss = (P-A) * ssss_0 * boys0(boys_arg) + (W-P) * ssss_0 * boys1(boys_arg)
    return psss

def eri_psps(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0):# checked!
    psss_0 = eri_psss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0)
    psss_1 = (P-A) * ssss_0 * boys1(boys_arg) + (W-P) * ssss_0 * boys2(boys_arg)
    ssss_1 = ssss_0 * boys1(boys_arg)

    first = np.einsum('k,i->ik', Q-C, psss_0)
    second = np.einsum('k,i->ik', W-Q, psss_1)
    third = (1 / (2*(zeta + eta))) * np.eye(3) * ssss_1 
    return first + second + third

# rho = zeta * eta / eta + zeta
# rho/zeta = eta / eta + zeta

def eri_ppss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0):# checked!
    psss_0 = eri_psss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0)
    psss_1 = (P-A) * ssss_0 * boys1(boys_arg) + (W-P) * ssss_0 * boys2(boys_arg)
    
    first = np.einsum('j,i->ij', P-B, psss_0)
    second = np.einsum('j,i->ij', W-P, psss_1)
    third = (1 / (2*zeta)) * np.eye(3) * (ssss_0 * boys0(boys_arg) - eta / (eta + zeta) * ssss_0 * boys1(boys_arg))
    return first + second + third

def eri_ppps(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0): # checked!
    spss_1 = (P-B) * ssss_0 * boys1(boys_arg) + (W-P) * ssss_0 * boys2(boys_arg)
    psss_1 = (P-A) * ssss_0 * boys1(boys_arg) + (W-P) * ssss_0 * boys2(boys_arg)
    psss_2 = (P-A) * ssss_0 * boys2(boys_arg) + (W-P) * ssss_0 * boys3(boys_arg)
    ppss_0 = eri_ppss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0)
    ppss_1 = np.einsum('j,i->ij',P-B,psss_1) + np.einsum('j,i->ij', W-P, psss_2) + (1 / (2*zeta)) * np.eye(3) * (ssss_0 * boys1(boys_arg) - eta / (eta + zeta) * ssss_0 * boys2(boys_arg))

    first = np.einsum('k,ij->ijk',Q-C,ppss_0)
    second = np.einsum('k,ij->ijk',W-Q,ppss_1)
    third = (1 / (2*(zeta + eta))) * (np.einsum('ik,j->ijk', np.eye(3), spss_1) + np.einsum('jk,i->ijk', np.eye(3), psss_1))
    return first + second + third

def eri_pppp(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0):
    return np.ones((3,3,3,3))
#    ppps_0 = eri_ppps(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0)
#    # making ppps_1 is a lot of work, this could be alleviated using a general boys function, and an 'm' argument to each function above!!
#    spss_2 = (P-B) * ssss_0 * boys2(boys_arg) + (W-P) * ssss_0 * boys3(boys_arg)
#    psss_2 = (P-A) * ssss_0 * boys2(boys_arg) + (W-P) * ssss_0 * boys3(boys_arg)
#    psss_3 = (P-A) * ssss_0 * boys3(boys_arg) + (W-P) * ssss_0 * boys4(boys_arg)
#    ppss_1 = np.einsum('j,i->ij',P-B,psss_1) + np.einsum('j,i->ij', W-P, psss_2) + (1 / (2*zeta)) * np.eye(3) * (ssss_0 * boys1(boys_arg) - eta / (eta + zeta) * ssss_0 * boys2(boys_arg))
#    ppss_2 = np.einsum('j,i->ij',P-B,psss_1) + np.einsum('j,i->ij', W-P, psss_2) + (1 / (2*zeta)) * np.eye(3) * (ssss_0 * boys2(boys_arg) - eta / (eta + zeta) * ssss_0 * boys3(boys_arg))
#    first = np.einsum('k,ij->ijk',Q-C,ppss_1)
#    second = np.einsum('k,ij->ijk',W-Q,ppss_2)
#    third = (1 / (2*(zeta + eta))) * (np.einsum('ik,j->ijk', np.eye(3), spss_2) + np.einsum('jk,i->ijk', np.eye(3), psss_2))
#    ppps_1 = first + second + third
#    # ppps_1 done
#
#    # spps_1
#
#
#    first = np.einsum('l,ijk->ijkl', Q-D, ppps_0) 
#    second = np.einsum('l,ijk->ijkl', W-Q, ppps_1)
#
#
#    return first + second + third 

def eri_ssss_old(A, B, C, D, aa, bb, cc, dd, coeff):
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
    ssss = 2 * np.pi**(10/4) * coeff * (zeta + eta)**(-1/2) * K_ab * K_cd * boys0(boys_arg)
    return ssss

def eri_psss_old(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Returns all 3 integrals of psss class'''
    oot_aa = 1 / (2 * aa)
    return oot_aa * jax.jacfwd(eri_ssss_old, 0)(A, B, C, D, aa, bb, cc, dd, coeff)

def eri_psps_old(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Returns all 9 integrals (shape=(3,3)) of psps class'''
    oot_cc = 1 / (2 * cc)
    return oot_cc * jax.jacfwd(eri_psss_old, 2)(A, B, C, D, aa, bb, cc, dd, coeff)

def eri_ppss_old(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Returns all 9 integrals (shape=(3,3)) of ppss class'''
    oot_bb = 1 / (2 * bb)
    return oot_bb * jax.jacfwd(eri_psss_old, 1)(A, B, C, D, aa, bb, cc, dd, coeff)

def eri_ppps_old(A, B, C, D, aa, bb, cc, dd, coeff):
    '''Returns all 27 integrals (shape=(3,3,3)) of ppps class'''
    oot_cc = 1 / (2 * cc)
    return oot_cc * jax.jacfwd(eri_ppss_old, 2)(A, B, C, D, aa, bb, cc, dd, coeff)
#
#def eri_pppp(A, B, C, D, aa, bb, cc, dd, coeff):
#    '''Returns all 81 integrals (shape=(3,3,3,3)) of pppp class'''
#    oot_dd = 1 / (2 * dd)
#    return oot_dd * jax.jacfwd(eri_ppps, 3)(A, B, C, D, aa, bb, cc, dd, coeff)

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
tmpargs = (A, B, C, D, alpha, beta, gamma, delta, coeff)

args = base(*tmpargs)
#print(eri_ssss(*args))
#print(eri_psss(*args))
#print(eri_psps(*args))
#for i in range(100):
#    eri_psps(*args)

#print(np.allclose(eri_ssss_old(*tmpargs), eri_ssss(*args)))
#print(np.allclose(eri_psss_old(*tmpargs), eri_psss(*args)))
#print(np.allclose(eri_psps_old(*tmpargs), eri_psps(*args)))
#print(np.allclose(eri_ppss_old(*tmpargs), eri_ppss(*args)))
#print(np.allclose(eri_ppps_old(*tmpargs), eri_ppps(*args)))


###jaxpr = jax.make_jaxpr(eri_ssss)(*args)
#jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(eri_pppp,0),1),2),3))(*args)
#print(len(str(jaxpr).splitlines()))

#jaxpr = jax.make_jaxpr(eri_pppp)(*args)
#print(jaxpr)
##jaxpr = jax.make_jaxpr(jax.jacfwd(eri_pppp,0))(*args)
##jaxpr = jax.make_jaxpr(eri_pppp)(*args)
#print(jaxpr)
#
#print('psss', eri_psss(*args))
#print('psps', eri_psps(*args))


# Psi4 input
