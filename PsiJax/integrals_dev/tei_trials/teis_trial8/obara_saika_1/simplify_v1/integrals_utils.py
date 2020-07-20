import numpy as np
from scipy import special

#import numpy as np

#@jax.jit
def gaussian_product(alpha_bra,A,alpha_ket, C):
    '''Gaussian product theorem. Returns center and coefficient of product'''
    R = (alpha_bra * A + alpha_ket * C) / (alpha_bra + alpha_ket)
    return R

def create_primitive(ax,ay,az, R, alpha):
    '''A primitive is completely defined by integer vector of angluar momenta, center coordinates, orbital exponent (TODO contraction coeff normalization ig?)'''
    #L = np.array([ax,ay,az])
    #L = [ax,ay,az]
    L = (ax,ay,az) #tuple best?
    return (L, R, alpha)

def delta(i):
    if i == 0: 
        return 1
    else: 
        return i

def boys(n,x):
    result = np.where(x < 1e-7, 1 / (2 * n + 1) - x *  (1 / (2 * n + 3)), 
                      0.5 * (x)**(-(n + 0.5)) * special.gammainc(n + 0.5,x) * special.gamma(n + 0.5))
    return result
#

#@jax.jit
#def boys(n,x):
#    result = np.where(x < 1e-7, 1 / (2 * n + 1) - x *  (1 / (2 * n + 3)), 
#                      0.5 * (x)**(-(n + 0.5)) * jax.lax.igamma(n + 0.5,x) * np.exp(jax.lax.lgamma(n + 0.5)))
#    return result
#
    #return 0.5 * (x + 1e-8)**(-(n + 0.5)) * jax.lax.igamma(n + 0.5, x + 1e-8) * np.exp(jax.lax.lgamma(n + 0.5))
    #return 0.5 * (x)**(-(n + 0.5)) * jax.lax.igamma(n + 0.5, x) * np.exp(jax.lax.lgamma(n + 0.5))

#def boys(nu, x):
#  if np.all(x) < 1e-7:
#    return (2*nu+1)**(-1) - x*(2*nu+3)**(-1) # (Handout 4, Eq. 17)
#  else:
#    return (1/2) * x**(-(nu+0.5)) * special.gamma(nu+0.5) * special.gammainc(nu+0.5,x) # (Handout 4, Eq. 16)

def ssss_0(a,b,c,d,gP,gQ,ABsq,CDsq):
  f = ( 2 * np.pi**2 ) / ( gP * gQ ) * np.sqrt( np.pi / ( gP + gQ ) ) * np.exp( -(a*b*ABsq)/gP ) * np.exp(-(c*d*CDsq)/gQ)
  return f
