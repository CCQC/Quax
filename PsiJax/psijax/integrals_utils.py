import jax
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.experimental import loops

def boys(m,x,eps=1e-12):
    return 0.5 * (x + eps)**(-(m + 0.5)) * jax.lax.igamma(m + 0.5, x + eps) \
           * np.exp(jax.lax.lgamma(m + 0.5))

#def boys(n,x):
#    result = np.where(x < 1e-8, 1 / (2 * n + 1) - x *  (1 / (2 * n + 3)), 
#       0.5 * (x)**(-(n + 0.5)) * jax.lax.igamma(n + 0.5,x) * np.exp(jax.lax.lgamma(n + 0.5)))
#    return result

#def binom(n,k):
#    '''Binomial coefficient'''
#    C = factorial(n) // (factorial(k) * factorial(n-k))
#    return C

def binomial_prefactor(s,ia,ib,xpa,xpb):
    with loops.Scope() as L:
        L.total = 0.
        L.t = 0
        for _ in L.while_range(lambda: L.t < s + 1):
          for _ in L.cond_range(s-ia <= L.t):
            for _ in L.cond_range(L.t <= ib):
              #L.total += binom(ia,s-L.t) * binom(ib,L.t) * xpa**(ia-s + L.t) * xpb**(ib - L.t)
              L.total += binomials[ia,s-L.t] * binomials[ib,L.t] * xpa**(ia-s + L.t) * xpb**(ib - L.t)
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

def fact_ratio2(a,b):
    #return factorial(a) / factorial(b) / factorial(a-2*b)
    #TODO this may go out of range? is a or b ever negative?
    return factorials[a] / factorials[b] / factorials[a-2*b]

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

factorials = np.array([1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600,
                       6227020800,87178291200,1307674368000,20922789888000,355687428096000,
                       6402373705728000,121645100408832000,2432902008176640000])

binomials = np.array([[1, 1,  0,  0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0,0,0], 
                      [1, 1,  0,  0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0,0,0],
                      [1, 2,  1,  0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0,0,0],
                      [1, 3,  3,  1,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0,0,0],
                      [1, 4,  6,  4,   1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0,0,0],
                      [1, 5, 10, 10,   5,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0,0,0],
                      [1, 6, 15, 20,  15,    6,    1,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0,0,0],
                      [1, 7, 21, 35,  35,   21,    7,    1,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0,0,0],
                      [1, 8, 28, 56,  70,   56,   28,    8,    1,    0,    0,    0,    0,    0,    0,   0,  0,  0,0,0],
                      [1, 9, 36, 84, 126,  126,   84,   36,    9,    1,    0,    0,    0,    0,    0,   0,  0,  0,0,0],
                      [1,10, 45,120, 210,  252,  210,  120,   45,   10,    1,    0,    0,    0,    0,   0,  0,  0,0,0],
                      [1,11, 55,165, 330,  462,  462,  330,  165,   55,   11,    1,    0,    0,    0,   0,  0,  0,0,0],
                      [1,12, 66,220, 495,  792,  924,  792,  495,  220,   66,   12,    1,    0,    0,   0,  0,  0,0,0],
                      [1,13, 78,286, 715, 1287, 1716, 1716, 1287,  715,  286,   78,   13,    1,    0,   0,  0,  0,0,0],
                      [1,14, 91,364,1001, 2002, 3003, 3432, 3003, 2002, 1001,  364,   91,   14,    1,   0,  0,  0,0,0],
                      [1,15,105,455,1365, 3003, 5005, 6435, 6435, 5005, 3003, 1365,  455,  105,   15,   1,  0,  0,0,0],
                      [1,16,120,560,1820, 4368, 8008,11440,12870,11440, 8008, 4368, 1820,  560,  120,  16,  1,  0,0,0],
                      [1,17,136,680,2380, 6188,12376,19448,24310,24310,19448,12376, 6188, 2380,  680, 136, 17,  1,0,0],
                      [1,18,153,816,3060, 8568,18564,31824,43758,48620,43758,31824,18564, 8568, 3060, 816,153, 18,1,0],
                      [1,19,171,969,3876,11628,27132,50388,75582,92378,92378,75582,50388,27132,11628,3876,969,171,19,1]])

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

