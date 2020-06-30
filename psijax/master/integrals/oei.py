import jax 
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
from jax.experimental import loops
from integrals_utils import gaussian_product, boys, binomial_prefactor, factorial, cartesian_product

def double_factorial(n):
    with loops.Scope() as s:
      s.k = 1
      for i in s.range(n, -1, -2):
        for _ in s.cond_range(i > 1):
          s.k *= i
      return s.k

def overlap(aa,La,A,bb,Lb,B):
    """
    Computes a single overlap integral. Taketa, Hunzinaga, Oohata 2.12
    """
    la,ma,na = La 
    lb,mb,nb = Lb 
    rab2 = np.dot(A-B,A-B)
    gamma = aa + bb
    P = gaussian_product(aa,A,bb,B)

    prefactor = (np.pi / gamma)**1.5 * np.exp(-aa * bb * rab2 / gamma)

    wx = overlap_component(la,lb,P[0]-A[0],P[0]-B[0],gamma)
    wy = overlap_component(ma,mb,P[1]-A[1],P[1]-B[1],gamma)
    wz = overlap_component(na,nb,P[2]-A[2],P[2]-B[2],gamma)

    return prefactor * wx * wy * wz

def overlap_component(l1,l2,PAx,PBx,gamma):
    """
    The 1d overlap integral component. Taketa, Hunzinaga, Oohata 2.12
    """
    total = 0
    K = 1 + (l1 + l2) // 2  
    #for i in range(1+int(floor(0.5*(l1+l2)))):
        #total += binomial_prefactor(2*i,l1,l2,PAx,PBx)* \
        #         double_factorial(2*i-1)/pow(2*gamma,i)
    with loops.Scope() as s:
      s.total = 0
      for i in s.range(K):
        s.total += binomial_prefactor(2*i,l1,l2,PAx,PBx) * double_factorial(2*i-1) \
                 / (2*gamma)**i
      return s.total

def kinetic(aa,La,A,bb,Lb,B):
    """
    Computes a single kinetic energy integral.
    """
    la,ma,na = La
    lb,mb,nb = Lb
    term0 = bb*(2*(lb+mb+nb)+3) * overlap(aa,La,A,bb,Lb,B)
    term1 = -2 * bb*2 * (overlap(aa,(l1,m1,n1),A, bb,(l2+2,m2,n2),B) \
             + overlap(aa,(la,ma,na),A,bb,(lb,mb+2,nb),B) \
             + overlap(aa,(la,ma,na),A,bb,(lb,mb,nb+2),B))

    term2 = -0.5*(l2*(l2-1)*overlap(aa,(la,ma,na),A,bb,(lb-2,mb,nb),B) \
                  + m2*(m2-1)*overlap(aa,(la,ma,na),A,bb,(lb,mb-2,nb),B) +
                  + n2*(n2-1)*overlap(aa,(la,ma,na),A,bb,(lb,mb,nb-2),B))
    return term0+term1+term2



def oei_arrays(geom, basis):
    #TODO mimic tei structure
    S = np.zeros((nbf,nbf))
    T = np.zeros((nbf,nbf))
    V = np.zeros((nbf,nbf))
    
    return S,T,V
