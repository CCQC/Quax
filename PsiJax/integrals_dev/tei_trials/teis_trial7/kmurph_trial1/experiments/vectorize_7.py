import jax 
from jax import lax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
from jax.experimental import loops
np.set_printoptions(linewidth=300)

@jax.jit
def binom(x,y):
    C = factorial(x) / (factorial(x-y) * factorial(y))
    return C

@jax.jit
def factorial(n):
  n = n.astype(float)
  return jax.lax.exp(jax.lax.lgamma(n + 1))

@jax.jit
def theta(l,lA,lB,PA,PB,r,g):
  """
  Calculate the theta factor of the gi term.
  (Handout 4, Eq. 23)
  """
  theta = jax_ck(l,lA,lB,PA,PB) * factorial(l) * g**(r-l) / (factorial(r) * factorial(l-2*r)) # real
  return theta

@jax.jit
def jax_ck(j,l,m,a,b):
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

@jax.jit
def gi(l,lp,r,rp,i, lA,lB,Ai,Bi,Pi,gP, lC,lD,Ci,Di,Qi,gQ):
  """
  Calculate the i-th coordinate component of the integral over primitives.
  (Handout 4, Eq. 22)

  note to self:
  gP = aa + bb
  gQ = cc + dd  
  """
  # TEMP TODO fixed ordering, this is working
  delta = 1/(4*gP) + 1/(4*gQ)
  gi  = (-1)**l 
  gi *= theta(l,lA,lB,Pi-Ai,Pi-Bi,r,gP) * theta(lp,lC,lD,Qi-Ci,Qi-Di,rp,gQ)
  gi *= (-1)**i * (2 * delta)**(2 * (r + rp))
  gi *= factorial(l - 2 * r + lp - 2 * rp) * delta**i
  gi *= (Pi - Qi)**(l - 2 * r + lp - 2 * (rp + i))
  gi /= (4 * delta)**(l) * (4 * delta)**(lp) * factorial(i) # this guy
  gi /= factorial(l - 2 * r + lp - 2 * (rp + i))
  return gi

@jax.jit
def gaussian_product(alpha_bra,alpha_ket,A,C):
    '''Gaussian product theorem. Returns center and coefficient of product'''
    R = (alpha_bra * A + alpha_ket * C) / (alpha_bra + alpha_ket)
    return R

@jax.jit
def boys(n,x):
    return 0.5 * (x + 1e-11)**(-(n + 0.5)) * jax.lax.igamma(n + 0.5, x + 1e-11) * np.exp(jax.lax.lgamma(n + 0.5))

@jax.jit
def cartesian_product(*arrays):
    '''JAX-friendly version of cartesian product. Same order as other function, more memory requirements though.'''
    tmp = np.asarray(np.meshgrid(*arrays, indexing='ij')).reshape(len(arrays),-1).T
    return np.asarray(tmp)

@jax.jit
def prefactor(a,b,c,d,gP,gQ,ABsq,CDsq):
  f = ( 2 * np.pi**2 ) / ( gP * gQ ) * np.sqrt( np.pi / ( gP + gQ ) ) * np.exp( -(a*b*ABsq)/gP ) * np.exp(-(c*d*CDsq)/gQ)
  return f

@jax.jit
def Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD):
  """
  Calling intermediate function gi to calculate individual x,y,z components,
  calculate integrals over primitives.
  (bracketed part of Handout 4, Eq. 18)
  """
  with loops.Scope() as s:
    s.lA,s.mA,s.nA,s.lB,s.mB,s.nB,s.lC,s.mC,s.nC,s.lD,s.mD,s.nD,s.a,s.b,s.c,s.d,s.RA,s.RB,s.RC,s.RD = lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD
    s.gP = s.a + s.b
    s.gQ = s.c + s.d
    s.delta = 1/(4*s.gP) + 1/(4*s.gQ)
    s.RP = gaussian_product(s.a,s.b,s.RA,s.RB)
    s.RQ = gaussian_product(s.c,s.d,s.RC,s.RD)
    s.ABsq = np.dot(s.RA-s.RB,s.RA-s.RB)
    s.CDsq = np.dot(s.RC-s.RD,s.RC-s.RD)
    s.PQsq = np.dot(s.RP-s.RQ,s.RP-s.RQ)
    s.ssss = prefactor(s.a,s.b,s.c,s.d,s.gP,s.gQ,s.ABsq,s.CDsq)
    s.boysarg = s.PQsq / (4 * s.delta)

    # Initialize loop variables
    s.Gxyz = 0.0
    s.l = lA + lB 
    s.r = np.floor(s.l/2) 
    s.lp = lC + lD 
    s.rp = np.floor(s.lp/2) 
    s.i = np.floor((s.l  - 2 * s.r + s.lp - 2 * s.rp) / 2)

    s.m = mA + mB 
    s.s = np.floor(s.m/2) 
    s.mp = mC + mD 
    s.sp = np.floor(s.mp/2) 
    s.j = np.floor((s.m  - 2 * s.s + s.mp - 2 * s.sp) / 2)

    s.n = nA + nB 
    s.t = np.floor(s.n/2) 
    s.np = nC + nD 
    s.tp = np.floor(s.np/2) 
    s.k = np.floor((s.n  - 2 * s.t + s.np - 2 * s.tp) / 2)
    
    # Loop over angular momentum and accumulate contributions to primitive. See Taketa, O-ohata, and Hunzinaga 1968, or Kevin Murphy, H.F. Schaefer 2018
    for _ in s.while_range(lambda: s.l > -1): # X
      s.r = np.floor(s.l/2)
      for _ in s.while_range(lambda: s.r > -1):
        s.lp = lC + lD 
        for _ in s.while_range(lambda: s.lp > -1):
          s.rp = np.floor(s.lp/2)
          for _ in s.while_range(lambda: s.rp > -1):
            s.i = np.floor((s.l  - 2 * s.r + s.lp - 2 * s.rp) / 2) # This works, compiles fine since loop variables are added in order
            for _ in s.while_range(lambda: s.i > -1):
              gx = gi(s.l,s.lp,s.r,s.rp,s.i, s.lA,s.lB,s.RA[0],s.RB[0],s.RP[0],s.gP, s.lC,s.lD,s.RC[0],s.RD[0],s.RQ[0],s.gQ)
              s.m = mA + mB 
              for _ in s.while_range(lambda: s.m > -1): # Y
                s.s = np.floor(s.m/2)
                for _ in s.while_range(lambda: s.s > -1):
                  s.mp = mC + mD 
                  for _ in s.while_range(lambda: s.mp > -1):
                    s.sp = np.floor(s.mp/2)
                    for _ in s.while_range(lambda: s.sp > -1):
                      s.j = np.floor((s.m  - 2 * s.s + s.mp - 2 * s.sp) / 2)
                      for _ in s.while_range(lambda: s.j > -1):
                        gy = gi(s.m,s.mp,s.s,s.sp,s.j, s.mA,s.mB,s.RA[1],s.RB[1],s.RP[1],s.gP, s.mC,s.mD,s.RC[1],s.RD[1],s.RQ[1],s.gQ)
                        s.n = nA + nB 
                        for _ in s.while_range(lambda: s.n > -1): # Z
                          s.t = np.floor(s.n/2)
                          for _ in s.while_range(lambda: s.t > -1):
                            s.np = mC + mD 
                            for _ in s.while_range(lambda: s.np > -1):
                              s.tp = np.floor(s.mp/2)
                              for _ in s.while_range(lambda: s.tp > -1):
                                s.k = np.floor((s.n  - 2 * s.t + s.np - 2 * s.tp) / 2)
                                for _ in s.while_range(lambda: s.k > -1):
                                  gz = gi(s.n,s.np,s.t,s.tp,s.k, s.nA,s.nB,s.RA[2],s.RB[2],s.RP[2],s.gP, s.nC,s.nD,s.RC[2],s.RD[2],s.RQ[2],s.gQ)
                                  nu = s.l - 2 * s.r + s.lp - 2 * s.rp - s.i + \
                                       s.m - 2 * s.s + s.mp - 2 * s.sp - s.j + \
                                       s.n - 2 * s.t + s.np - 2 * s.tp - s.k #THIS ORDER MATTERS (UGH)
                                  F = boys(nu, s.boysarg)
                                  s.Gxyz += F * gx * gy * gz # this is working

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

    s.Gxyz *= s.ssss
    #Na = N(a,lA,mA,nA)
    #Nb = N(b,lB,mB,nB)
    #Nc = N(c,lC,mC,nC)
    #Nd = N(d,lD,mD,nD)
    # normally normalization here
    return s.Gxyz


# This is an (fxyz fxyz | fxyz fxyz) integral. Under this implementation requires 15k boys function evaluations, maximum 
lA,mA,nA = 1,0,0
lB,mB,nB = 1,0,0
lC,mC,nC = 1,0,0
lD,mD,nD = 1,0,0
a,b,c,d = 0.5, 0.5, 0.5, 0.5
RA,RB,RC,RD = np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9]), np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9])
#print(Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD))

# Vmap the function
vmap_primitive = jax.vmap(Gxyz, in_axes=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))

# Create array inputs to function
lA = np.broadcast_to(lA, (1000000,))
mA = np.broadcast_to(mA, (1000000,))
nA = np.broadcast_to(nA, (1000000,))
lB = np.broadcast_to(lB, (1000000,))
mB = np.broadcast_to(mB, (1000000,))
nB = np.broadcast_to(nB, (1000000,))
lC = np.broadcast_to(lC, (1000000,))
mC = np.broadcast_to(mC, (1000000,))
nC = np.broadcast_to(nC, (1000000,))
lD = np.broadcast_to(lD, (1000000,))
mD = np.broadcast_to(mD, (1000000,))
nD = np.broadcast_to(nD, (1000000,))
a  = np.broadcast_to(a,  (1000000,))
b  = np.broadcast_to(b,  (1000000,))
c  = np.broadcast_to(c,  (1000000,))
d  = np.broadcast_to(d,  (1000000,))
RA = np.broadcast_to(RA, (1000000,3))
RB = np.broadcast_to(RB, (1000000,3))
RC = np.broadcast_to(RC, (1000000,3))
RD = np.broadcast_to(RD, (1000000,3))

vmap_primitive(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD)


#print(Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD))
#for i in range(1000):
#    print(Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD))


