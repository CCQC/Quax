#NOTE Jitting is better in this case
import jax
import jax.numpy as np
from jax.scipy import special
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)

# trial 1: try using just exact, no where, using fudge epislon
def boysn(x,n):
    #return 0.5 * (x + 1e-10)**(-(n + 0.5)) * special.gammainc(n + 0.5, x + 1e-10) * np.exp(special.gammaln(n + 0.5))
    #return 0.5 * (x + 1e-10)**(-(n + 0.5)) * jax.lax.igamma(n + 0.5, x + 1e-10) * np.exp(special.gammaln(n + 0.5))
    return 0.5 * (x + 1e-10)**(-(n + 0.5)) * jax.lax.igamma(n + 0.5, x + 1e-10) * np.exp(jax.lax.lgamma(n + 0.5))


    #return 0.5 * (x)**(-(n + 0.5)) * special.gammainc(n + 0.5, x) * np.exp(special.gammaln(n + 0.5))
    #return 0.5 * (x + 1e-9)**(-(n + 0.5)) #* special.gammainc(n + 0.5, x + 1e-9) * np.exp(special.lgamma(n + 0.5))
    #return 0.5 * (x + 1e-9)**(-(n + 0.5)) * special.gammainc(n + 0.5, x + 1e-9) #* np.exp(special.lgamma(n + 0.5))
    #return 0.5 * (x)**(-(n + 0.5)) * special.gammainc(n + 0.5, x) #* np.exp(special.lgamma(n + 0.5))
    #return 0.5 * (x)**(-(n + 0.5)) * np.exp(special.gammaln(n + 0.5))
    #print(x)
    #a = special.gammainc(n + 0.5, x)
    #return 0.5 * (x)**(-(n + 0.5)) * np.exp(special.gammaln(x + 0.5))

print(boysn(0.5, 0))
print(boysn(0.5, 1))
print(boysn(0.5, 2))

print(boysn(0.0, 0))
print(boysn(0.0, 1))
print(boysn(0.0, 2))

#def boysn(x,n):
#    return np.where(x<1e-8, boys_exact(x,n), boys_taylor(x,n))
#
#
#
#def boys_taylor(x,n):

#def boysn(x,n):
#    '''This function fails for x>30, which occurs when atoms are far apart and/or when orbital expnoents are large'''
#    interval = 1e-5 
#    i = jax.lax.convert_element_type(np.round(x / interval), np.int64) # index of gridpoint nearest to x
#    xgrid = xgrid_array[i] # grid x-value
#    xx = x - xgrid
#    tmp = boys[:,i]
#    f = tmp[np.arange(6) + n]
#    F = f[0] - xx * f[1] + 0.5 * xx**2 * f[2] - (1/6) * xx**3 * f[3] + (1/24) * xx**4 * f[4] - (1/120) * xx**5 * f[5]
#    return F

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
# ijkl = eri_ijkl(*args, m)
# NOTE some arguments are not used most of the time
# rho = zeta * eta / eta + zeta
# rho/zeta = eta / eta + zeta

# All equations obtained from Obara, Saika 1986, Table I
def eri_ssss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m):# checked
    return ssss_0 * boysn(boys_arg, m)
    
def eri_psss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m):# checked
    psss = (P-A) * ssss_0 * boysn(boys_arg, m) + (W-P) * ssss_0 * boysn(boys_arg, m+1)
    return psss

def eri_psps(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m):# checked
    psss_0 = eri_psss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m+0)
    psss_1 = eri_psss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m+1)
    ssss_1 = ssss_0 * boysn(boys_arg, m+1)
    first = np.einsum('k,i->ik', Q-C, psss_0)
    second = np.einsum('k,i->ik', W-Q, psss_1)
    third = (1 / (2*(zeta + eta))) * np.eye(3) * ssss_1 
    return first + second + third

def eri_ppss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m):# checked
    psss_0 = eri_psss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m+0)
    psss_1 = eri_psss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m+1)
    first = np.einsum('j,i->ij', P-B, psss_0)
    second = np.einsum('j,i->ij', W-P, psss_1)
    third = (1 / (2*zeta)) * np.eye(3) * (ssss_0 * boysn(boys_arg,m+0) - eta / (eta + zeta) * ssss_0 * boysn(boys_arg, m+1))
    return first + second + third

def eri_ppps(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m): # checked
    ppss_0 = eri_ppss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m)
    ppss_1 = eri_ppss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m+1)
    # This works since zeta/eta/P/Q/W are invariant to swaps within bra or swaps within ket.
    # If you need to swap bra/ket, you would need to swap AB/CD, zeta/eta, and P/Q, but W is invariant
    spss_1 = eri_psss(B,A,C,D,zeta,eta,P,Q,W, boys_arg, ssss_0, m+1) 
    psss_1 = eri_psss(A,B,C,D,zeta,eta,P,Q,W, boys_arg, ssss_0, m+1)
    first = np.einsum('k,ij->ijk',Q-C,ppss_0)
    second = np.einsum('k,ij->ijk',W-Q,ppss_1)
    third = (1 / (2*(zeta + eta))) * (np.einsum('ik,j->ijk', np.eye(3), spss_1) + np.einsum('jk,i->ijk', np.eye(3), psss_1))
    return first + second + third

def eri_pppp(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m): #checked
    ppps_0 = eri_ppps(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m+0)
    ppps_1 = eri_ppps(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m+1)
    spps_1 = eri_psps(B, A, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m+1)
    psps_1 = eri_psps(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m+1)
    ppss_0 = eri_ppss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m+0)
    ppss_1 = eri_ppss(A, B, C, D, zeta, eta, P, Q, W, boys_arg, ssss_0, m+1)
    first = np.einsum('l,ijk->ijkl',Q-D,ppps_0)
    second = np.einsum('l,ijk->ijkl',W-Q,ppps_1)
    third = (1 / (2 * (zeta + eta))) * (np.einsum('il,jk->ijkl', np.eye(3), spps_1) + np.einsum('jl,ik->ijkl', np.eye(3), psps_1))
    rho_o_eta = zeta / (eta + zeta)
    fourth = (1/(2*eta)) * np.einsum('kl,ij->ijkl' ,np.eye(3), ppss_0 - rho_o_eta * ppss_1)
    return first + second + third + fourth

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
###
#tmpargs = (A, B, C, D, alpha, beta, gamma, delta, coeff)
#
#args = base(*tmpargs)
#jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(eri_pppp,0),1),2),3)(*args,0)

