import jax 
from jax import lax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from jax.experimental import loops

#@jax.jit
def binom(x,y):
    C = factorial(x) / (factorial(x-y) * factorial(y))
    return C

#@jax.jit
def factorial(n):
  n = n.astype(float)
  return jax.lax.exp(jax.lax.lgamma(n + 1))

#@jax.jit
def theta(l,lA,lB,PA,PB,r,g):
  """
  Calculate the theta factor of the gi term.
  (Handout 4, Eq. 23)
  """
  theta = ck(l,lA,lB,PA,PB) * factorial(l) * g**(r-l) / (factorial(r) * factorial(l-2*r)) # real
  return theta

#@jax.jit
def ck(j,l,m,a,b):
  '''
  Proves you can jit-compile a function which takes integer arguments and modifies them in while loops
  This means you can in principle convert to while loops for whatever you need
  This is probably the best way to do recursion but its clunky
  '''
  with loops.Scope() as s:
    s.coefficient = 0.0
    s.k = l 
    s.i = m 
    s.j = j
    s.l = l
    s.m = m
    s.a = a
    s.b = b
    for _ in s.while_range(lambda: s.k > -1):
      s.i = s.m
      for _ in s.while_range(lambda: s.i > -1):
        for _ in s.cond_range(s.k + s.i == s.j): 
          s.coefficient += binom(s.l,s.k) * binom(s.m,s.i) * s.a**(s.l-s.k) * s.b**(s.m-s.i)
        s.i -= 1
      s.k -= 1
    return s.coefficient

#@jax.jit
def gi(l,lp,r,rp,i, lA,lB,Ai,Bi,Pi,gP, lC,lD,Ci,Di,Qi,gQ):
  """
  Calculate the i-th coordinate component of the integral over primitives.
  (Handout 4, Eq. 22)
  """
  delta = 1/(4*gP) + 1/(4*gQ)
  gi  = (-1)**l 
  gi *= theta(l,lA,lB,Pi-Ai,Pi-Bi,r,gP) * theta(lp,lC,lD,Qi-Ci,Qi-Di,rp,gQ)
  gi *= (-1)**i * (2 * delta)**(2 * (r + rp))
  gi *= factorial(l - 2 * r + lp - 2 * rp) * delta**i
  gi *= (Pi - Qi)**(l - 2 * r + lp - 2 * (rp + i))
  gi /= (4 * delta)**(l) * (4 * delta)**(lp) * factorial(i) # this guy
  gi /= factorial(l - 2 * r + lp - 2 * (rp + i))
  return gi

#@jax.jit
def gaussian_product(alpha_bra,alpha_ket,A,C):
    '''Gaussian product theorem. Returns center and coefficient of product'''
    R = (alpha_bra * A + alpha_ket * C) / (alpha_bra + alpha_ket)
    return R

#@jax.jit
def boys(n,x):
    #TODO vmap with boys function not working, just use s function form
    #return 0.88622692545275798 * jax.lax.rsqrt(x + 1e-10) * jax.lax.erf(jax.lax.sqrt(x + 1e-10))
    return x + 1.0
    #return 0.5 * (x + 1e-11)**(-(n + 0.5)) * jax.lax.igamma(n + 0.5, x + 1e-11) * np.exp(jax.lax.lgamma(n + 0.5))

#@jax.jit
def cartesian_product(*arrays):
    '''JAX-friendly version of cartesian product. Same order as other function, more memory requirements though.'''
    tmp = np.asarray(np.meshgrid(*arrays, indexing='ij')).reshape(len(arrays),-1).T
    return np.asarray(tmp)

def np_cartesian_product(*arrays):                                   
    '''Generalized cartesian product of any number of arrays'''       
    la = len(arrays)                                                  
    dtype = onp.find_common_type([a.dtype for a in arrays], [])       
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)     
    for i, a in enumerate(onp.ix_(*arrays)):                          
        arr[...,i] = a                                                
    return arr.reshape(-1, la)                                        

#@jax.jit
def prefactor(a,b,c,d,gP,gQ,ABsq,CDsq):
  f = ( 2 * np.pi**2 ) / ( gP * gQ ) * np.sqrt( np.pi / ( gP + gQ ) ) * np.exp( -(a*b*ABsq)/gP ) * np.exp(-(c*d*CDsq)/gQ)
  return f


vmapped_gaussian_product = jax.vmap(gaussian_product, (0,0,None,None))
vmapped_prefactor = jax.vmap(prefactor, (0,0,0,0,0,0,None,None))
#vmapped_gi = jax.vmap(gi, (None,None,None,None,None,None,None,None,None,None,0,None,None,None,None,None,0))
vmapped_gi = jax.vmap(gi, (None,None,None,None,None,None,None,None,None,0,0,None,None,None,None,0,0))
                         #(s.n,s.np,s.t,s.tp,s.k, s.nA,s.nB,s.RA[2],s.RB[2],s.RP[2],s.gP, s.nC,s.nD,s.RC[2],s.RD[2],s.RQ[2],s.gQ)

#(l,lp,r,rp,i, lA,lB,Ai,Bi,Pi,gP, lC,lD,Ci,Di,Qi,gQ)


def contracted_eri(L,a,b,c,d,RA,RB,RC,RD,contraction):
#def primitive_eri(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD,contraction):
  """
  Computes a single ERI primitive using highly-inefficient scheme of 
  Taketa, O-ohata, and Hunzinaga 1968, or Kevin Murphy, H.F. Schaefer 2018.
  Parameters
  ----------
  L : Vector of size 12, (x,y,z)-components of angular momentum on center A,B,C,D
  a,b,c,d     : All Gaussian exponents on center (A,B,C,D) for the contraction
  RA,RB,RC,RD : Cartesian-coordinate vector of center (A,B,C,D)
  contraction : The fused normalization constant/contraction coefficients for this primitive  N_contract*Na*Nb*Nc*Nd*coeffa*coeffb*coeffc*coeffd for all primitives in contraction

  Returns
  -------
  A single contracted ERI.
  """
  with loops.Scope() as s:
    #s.lA,s.mA,s.nA,s.lB,s.mB,s.nB,s.lC,s.mC,s.nC,s.lD,s.mD,s.nD,s.a,s.b,s.c,s.d,s.RA,s.RB,s.RC,s.RD = lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD
    # Unpack arguments into scope variable
    s.lA,s.mA,s.nA,s.lB,s.mB,s.nB,s.lC,s.mC,s.nC,s.lD,s.mD,s.nD = L
    s.a,s.b,s.c,s.d = a, b, c, d 
    s.RA,s.RB,s.RC,s.RD = RA, RB, RC, RD 
    s.gP = s.a + s.b
    s.gQ = s.c + s.d
    s.delta = 1/(4*s.gP) + 1/(4*s.gQ)
    s.RP = vmapped_gaussian_product(s.a,s.b,s.RA,s.RB) # (K,3)
    s.RQ = vmapped_gaussian_product(s.c,s.d,s.RC,s.RD) # (K,3)
    s.ABsq = np.dot(s.RA-s.RB,s.RA-s.RB)
    s.CDsq = np.dot(s.RC-s.RD,s.RC-s.RD)
    #s.PQsq = np.dot(s.RP-s.RQ,s.RP-s.RQ)
    s.PQsq = np.einsum('pq,pq->p', s.RP-s.RQ,s.RP-s.RQ)                 # shape (K,)  is this einsum right?
    s.ssss = vmapped_prefactor(s.a,s.b,s.c,s.d,s.gP,s.gQ,s.ABsq,s.CDsq) # shape (K,)
    s.boysarg = s.PQsq / (4 * s.delta)                                  # shape (K,)

    # Initialize loop variables
    #s.Gxyz = 0.0
    s.Gxyz = 0.0
    #s.Gxyz = np.zeros_like(
    s.l = s.lA + s.lB 
    s.r = np.floor(s.l/2) 
    s.lp = s.lC + s.lD 
    s.rp = np.floor(s.lp/2) 
    s.i = np.floor((s.l  - 2 * s.r + s.lp - 2 * s.rp) / 2)

    s.m = s.mA + s.mB 
    s.s = np.floor(s.m/2) 
    s.mp = s.mC + s.mD 
    s.sp = np.floor(s.mp/2) 
    s.j = np.floor((s.m  - 2 * s.s + s.mp - 2 * s.sp) / 2)

    s.n = s.nA + s.nB 
    s.t = np.floor(s.n/2) 
    s.np = s.nC + s.nD 
    s.tp = np.floor(s.np/2) 
    s.k = np.floor((s.n  - 2 * s.t + s.np - 2 * s.tp) / 2)
    
    # Loop over angular momentum and accumulate contributions to primitive. See Taketa, O-ohata, and Hunzinaga 1968, or Kevin Murphy, H.F. Schaefer 2018
    for _ in s.while_range(lambda: s.l > -1): # X
      s.r = np.floor(s.l/2)
      for _ in s.while_range(lambda: s.r > -1):
        s.lp = s.lC + s.lD 
        for _ in s.while_range(lambda: s.lp > -1):
          s.rp = np.floor(s.lp/2)
          for _ in s.while_range(lambda: s.rp > -1):
            s.i = np.floor((s.l  - 2 * s.r + s.lp - 2 * s.rp) / 2) # This works, compiles fine since loop variables are added in order
            for _ in s.while_range(lambda: s.i > -1):
              gx = vmapped_gi(s.l,s.lp,s.r,s.rp,s.i, s.lA,s.lB,s.RA[0],s.RB[0],s.RP[:,0],s.gP, s.lC,s.lD,s.RC[0],s.RD[0],s.RQ[:,0],s.gQ)
              s.m = s.mA + s.mB 
              for _ in s.while_range(lambda: s.m > -1): # Y
                s.s = np.floor(s.m/2)
                for _ in s.while_range(lambda: s.s > -1):
                  s.mp = s.mC + s.mD 
                  for _ in s.while_range(lambda: s.mp > -1):
                    s.sp = np.floor(s.mp/2)
                    for _ in s.while_range(lambda: s.sp > -1):
                      s.j = np.floor((s.m  - 2 * s.s + s.mp - 2 * s.sp) / 2)
                      for _ in s.while_range(lambda: s.j > -1):
                        gy = vmapped_gi(s.m,s.mp,s.s,s.sp,s.j, s.mA,s.mB,s.RA[1],s.RB[1],s.RP[:,1],s.gP, s.mC,s.mD,s.RC[1],s.RD[1],s.RQ[:,1],s.gQ)
                        s.n = s.nA + s.nB 
                        for _ in s.while_range(lambda: s.n > -1): # Z
                          s.t = np.floor(s.n/2)
                          for _ in s.while_range(lambda: s.t > -1):
                            s.np = s.mC + s.mD 
                            for _ in s.while_range(lambda: s.np > -1):
                              s.tp = np.floor(s.mp/2)
                              for _ in s.while_range(lambda: s.tp > -1):
                                s.k = np.floor((s.n  - 2 * s.t + s.np - 2 * s.tp) / 2)
                                for _ in s.while_range(lambda: s.k > -1):
                                  gz = vmapped_gi(s.n,s.np,s.t,s.tp,s.k, s.nA,s.nB,s.RA[2],s.RB[2],s.RP[:,2],s.gP, s.nC,s.nD,s.RC[2],s.RD[2],s.RQ[:,2],s.gQ)
                                  nu = s.l - 2 * s.r + s.lp - 2 * s.rp - s.i + \
                                       s.m - 2 * s.s + s.mp - 2 * s.sp - s.j + \
                                       s.n - 2 * s.t + s.np - 2 * s.tp - s.k #THIS ORDER MATTERS (UGH)
                                  F = boys(nu, s.boysarg) #TODO fix this when JAX fixes the vmap issue with igamma
                                  #s.Gxyz += F * gx * gy * gz # this is working
                                  s.Gxyz += np.sum(F * gx * gy * gz * contraction * s.ssss)  #TODO something is definitely wrong here
                                  #s.Gxyz += np.sum(F * gx * gy * gz) 

                                  s.k -= 1  
                                s.tp -= 1  
                              s.np -= 1  
                            s.t -= 1  
                          s.n -= 1  
                        s.j -= 1  
                      s.sp -= 1  
                    s.mp -= 1   # Decrement all loop variables until 0
                  s.s -= 1  
                s.m -= 1  
              s.i -= 1
            s.rp -= 1
          s.lp -= 1
        s.r -= 1
      s.l -= 1

    #s.Gxyz *= s.ssss
    #s.Gxyz *= contraction 
    #TODO this contraction order may be wrong, you are summing along the contraction dimension in the loop then multiplying by all con
    return s.Gxyz

# TEST
#primitive_eri(L,a,b,c,d,RA,RB,RC,RD,contraction):
#L = [1,0,0,1,0,0,1,0,0,1,0,0]
#a,b,c,d = np.array([0.5,0.4]),np.array([0.5,0.4]), np.array([0.5,0.4]), np.array([0.5,0.4])
#RA,RB,RC,RD = np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9]), np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9])
#contraction = np.array([0.75,0.25])
#
#print(contracted_eri(L,a,b,c,d,RA,RB,RC,RD,contraction))







