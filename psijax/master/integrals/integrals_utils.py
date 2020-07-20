import jax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
from jax.experimental import loops

def boys(m,x,eps=1e-12):
    return 0.5 * (x + eps)**(-(m + 0.5)) * jax.lax.igamma(m + 0.5, x + eps) \
           * np.exp(jax.lax.lgamma(m + 0.5))

#def boys(n,x):
#    result = np.where(x < 1e-8, 1 / (2 * n + 1) - x *  (1 / (2 * n + 3)), 
#       0.5 * (x)**(-(n + 0.5)) * jax.lax.igamma(n + 0.5,x) * np.exp(jax.lax.lgamma(n + 0.5)))
#    return result

def binom(n,k):
    '''Binomial coefficient'''
    C = factorial(n) // (factorial(k) * factorial(n-k))
    return C

def binomial_prefactor(s,ia,ib,xpa,xpb):
    with loops.Scope() as L:
        L.total = 0.
        L.t = 0
        for _ in L.while_range(lambda: L.t < s + 1):
          for _ in L.cond_range(s-ia <= L.t):
            for _ in L.cond_range(L.t <= ib):
              L.total += binom(ia,s-L.t) * binom(ib,L.t) * xpa**(ia-s + L.t) * xpb**(ib - L.t)
          L.t += 1
        return L.total

def factorial(n):
  '''Note: switch to float for high values (n>20) for stability'''
  with loops.Scope() as s:
    s.result = 1
    s.k = 1
    for _ in s.while_range(lambda: s.k < n + 1):
      s.result *= s.k
      s.k += 1
    return s.result

def gaussian_product(alpha1,A,alpha2,B):
    '''Gaussian product theorem. Returns center.'''
    return (alpha1*A+alpha2*B)/(alpha1+alpha2)
 
#def gaussian_product(alpha_bra,alpha_ket,A,C):
#    R = (alpha_bra * A + alpha_ket * C) / (alpha_bra + alpha_ket)
#    c = np.exp(np.dot(A-C,A-C) * (-alpha_bra * alpha_ket / (alpha_bra + alpha_ket)))
#    return R,c

def find_unique_shells(nshells):
    '''Find shell quartets which correspond to corresponding to unique two-electron integrals, i>=j, k>=l, IJ>=KL'''
    v = onp.arange(nshells,dtype=np.int16) 
    indices = old_cartesian_product(v,v,v,v)
    cond1 = (indices[:,0] >= indices[:,1]) & (indices[:,2] >= indices[:,3]) 
    cond2 = indices[:,0] * (indices[:,0] + 1)/2 + indices[:,1] >= indices[:,2] * (indices[:,2] + 1)/2 + indices[:,3]
    mask = cond1 & cond2 
    return np.asarray(indices[mask,:])

#@jax.jit
def cartesian_product(*arrays):
    '''JAX-friendly version of cartesian product. Same order as other function, more memory requirements though.'''
    tmp = np.asarray(np.meshgrid(*arrays, indexing='ij')).reshape(len(arrays),-1).T
    #tmp = np.meshgrid(*arrays, indexing='ij')
    return np.asarray(tmp)

def am_vectors(am, length=3):
    '''
    Builds up all possible angular momentum component vectors of with total angular momentum 'am'
    am = 2 ---> [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    Returns a generator which must be converted to an iterable,
    for example, call the following: [list(i) for i in am_vectors(2)]

    Works by building up each possibility :
    For a given value in reversed(range(am+1)), find all other possible values for other entries in length 3 vector
     value     am_vectors(am-value,length-1)    (value,) + permutation
       2 --->         [0,0]                 ---> [2,0,0] ---> dxx
       1 --->         [1,0]                 ---> [1,1,0] ---> dxy
         --->         [0,1]                 ---> [1,0,1] ---> dxz
       0 --->         [2,0]                 ---> [0,2,0] ---> dyy
         --->         [1,1]                 ---> [0,1,1] ---> dyz
         --->         [0,2]                 ---> [0,0,2] ---> dzz
    '''
    if length == 1:
        yield (am,)
    else:
        # reverse so angular momentum order is canonical, e.g., dxx dxy dxz dyy dyz dzz
        for value in reversed(range(am + 1)):
            for permutation in am_vectors(am - value,length - 1):
                yield (value,) + permutation

angular_momentum_combinations = np.array([[0,0,0], 
                                          [1,0,0],
                                          [0,1,0],
                                          [0,0,1],
                                          [2,0,0],
                                          [1,1,0],
                                          [1,0,1],
                                          [0,2,0],
                                          [0,1,1],
                                          [0,0,2], 
                                          [3,0,0],
                                          [2,1,0],
                                          [2,0,1],
                                          [1,2,0],
                                          [1,1,1],
                                          [1,0,2],
                                          [0,3,0],
                                          [0,2,1],
                                          [0,1,2],
                                          [0,0,3]])
am_leading_indices = np.array([0,1,4,10])

