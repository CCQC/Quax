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
  #delta = 1/(4*gP) + 1/(4*gQ)
  #gi  = (-1)**l 
  #gi *= theta(l,lA,lB,Pi-Ai,Pi-Bi,r,gP) * theta(lp,lC,lD,Qi-Ci,Qi-Di,rp,gQ)
  #gi *= (-1)**i * (2 * delta)**(2 * (r + rp))
  #gi *= factorial(l + lp - 2 * r - 2 * rp) * delta**i
  #gi *= (Pi - Qi)**(l + lp - 2 * (r + rp + i))
  #gi /= (4 * delta)**(l + lp) * factorial(i)
  #gi /= factorial(l + lp - 2 * (r + rp + i))

  # TEMP TODO
  delta = 1/(4*gP) + 1/(4*gQ)
  gi  = (-1)**l 
  gi *= theta(l,lA,lB,Pi-Ai,Pi-Bi,r,gP) * theta(lp,lC,lD,Qi-Ci,Qi-Di,rp,gQ)
  gi *= (-1)**i * (2 * delta)**(2 * (r + rp))
  #gi *= factorial(l + lp - 2 * r - 2 * rp) * delta**i
  gi *= factorial(l - 2 * r + lp - 2 * rp) * delta**i
  #gi *= (Pi - Qi)**(l + lp - 2 * (r + rp + i))
  gi *= (Pi - Qi)**(l - 2 * r + lp - 2 * (rp + i))
  #gi /= (4 * delta)**(l + lp) * factorial(i) # this guy
  gi /= (4 * delta)**(l) * (4 * delta)**(lp) * factorial(i) # this guy
  #gi /= factorial(l + lp - 2 * (r + rp + i))
  gi /= factorial(l - 2 * r + lp - 2 * (rp + i))
  return gi

#gi_vmap = jax.jit(jax.vmap(gi, in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None)))
gi_vmap = jax.jit(jax.vmap(gi, in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None)))

#print(gi(3,3,1,1,1, 3,3,7.,8.,9.,2.,3,3,2.,3.,4.,5.))
#print(gi(3,3,1,1,1, 3,3,7.,8.,9.,2.,3,3,2.,3.,4.,5.))
#for i in range(10000):
#    gi(3,3,1,1,1, 3,3,7.,8.,9.,2.,3,3,2.,3.,4.,5.)

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
    s.boysarg = s.PQsq / (4 * s.delta)

    s.Gxyz = 0.0
    s.l = lA + lB 
    s.r = np.floor(s.l/2) 
    s.lp = lC + lD 
    s.rp = np.floor(s.lp/2) 
    s.i = np.floor((s.l + s.lp - 2 * s.r - 2 * s.rp) / 2) 

    for _ in s.while_range(lambda: s.l > -1):
      s.r = np.floor(s.l/2)
      for _ in s.while_range(lambda: s.r > -1):
        s.lp = lC + lD 
        for _ in s.while_range(lambda: s.lp > -1):
          s.rp = np.floor(s.lp/2)
          for _ in s.while_range(lambda: s.rp > -1):
            #s.i = np.floor((s.l + s.lp - 2 * s.r - 2 * s.rp) / 2) # Compiles forever until OOM
            s.i = np.floor((s.l  - 2 * s.r + s.lp - 2 * s.rp) / 2) # This works, compiles fine since loop variables are added in order
            for _ in s.while_range(lambda: s.i > -1):
              gx = gi(s.l,s.lp,s.r,s.rp,s.i, s.lA,s.lB,s.RA[0],s.RB[0],s.RP[0],s.gP, s.lC,s.lD,s.RC[0],s.RD[0],s.RQ[0],s.gQ)
              nu = s.l - 2 * s.r + s.lp - 2 * s.rp - s.i  #THIS ORDER MATTERS (UGH)
              F = boys(nu, s.boysarg)
              s.Gxyz += F * gx


              #s.nu = s.l + s.lp - 2 * (s.r+s.rp)-s.i
              #nu = s.l - 2 * s.r + s.lp - 2 * s.rp - s.i
              #F = boys(s.nu, s.boysarg)
              #F = 0.5 * (s.boysarg + 1e-11)**(-(s.nu + 0.5)) * jax.lax.igamma(s.nu + 0.5, s.boysarg + 1e-11) * np.exp(jax.lax.lgamma(s.nu + 0.5))

              # WORKS: fake boys function acucmulation
              #nu = s.l - 2 * s.r + s.lp - 2 * s.rp - s.i  #THIS ORDER MATTERS (UGH)
              #F = 0.5 * (s.boysarg + 1e-11)**(-(nu + 0.5)) * jax.lax.igamma(nu + 0.5, s.boysarg + 1e-11) * np.exp(jax.lax.lgamma(nu + 0.5))
              #s.Gxyz += F


              # This works ( thou shalt not add s.l + s.lp
              #nu = s.l - 2 * s.r + s.lp - 2 * s.rp - s.i  #THIS ORDER MATTERS (UGH)
              #F = boys(nu, s.boysarg)
              #s.Gxyz += F

              s.i -= 1
            s.rp -= 1
          s.lp -= 1
        s.r -= 1
      s.l -= 1

    return s.Gxyz



  #Na = N(a,lA,mA,nA)
  #Nb = N(b,lB,mB,nB)
  #Nc = N(c,lC,mC,nC)
  #Nd = N(d,lD,mD,nD)

  #Gxyz *= Na * Nb * Nc * Nd
#  return Gxyz

@jax.jit
def prefactor(a,b,c,d,gP,gQ,ABsq,CDsq):
  f = ( 2 * np.pi**2 ) / ( gP * gQ ) * np.sqrt( np.pi / ( gP + gQ ) ) * np.exp( -(a*b*ABsq)/gP ) * np.exp(-(c*d*CDsq)/gQ)
  return f

# This is an (fxyz fxyz | fxyz fxyz) integral. Under this implementation requires 15k boys function evaluations, maximum 
lA,mA,nA = 1,1,1
lB,mB,nB = 1,1,1
lC,mC,nC = 1,1,1
lD,mD,nD = 1,1,1
a,b,c,d = 0.5, 0.5, 0.5, 0.5
RA,RB,RC,RD = np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9]), np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9])

print(Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD))
#for i in range(1000):
#    print(Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD))

#for i in range(1000):
#    Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD, gx_vec, gy_vec, gz_vec)
#    #Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD)
#
#
#lA,mA,nA = 0,1,1
#lB,mB,nB = 0,1,1
#lC,mC,nC = 0,1,1
#lD,mD,nD = 0,1,1
#a,b,c,d = 0.5, 0.5, 0.5, 0.5
#RA,RB,RC,RD = np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9]), np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9])
#gx_vec, gy_vec, gz_vec = get_gi_vecs(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD)
#
#for i in range(1000):
#    Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD, gx_vec, gy_vec, gz_vec)


