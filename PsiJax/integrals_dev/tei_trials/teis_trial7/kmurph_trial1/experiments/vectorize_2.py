import jax 
from jax import lax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
from jax.experimental import loops

@jax.jit
def binom(x,y):
    C = factorial(x) / (factorial(x-y) * factorial(y))
    return C

#TODO these two factorials give basically the same time, so...
#@jax.jit
#def factorial(n):
#  with loops.Scope() as s:
#    s.num = 1
#    s.n = n
#    for _ in s.while_range(lambda: s.n >=1):
#      s.num = s.num * s.n
#      s.n -= 1
#    return s.num

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
  #theta = factorial(l) * g**(r-l) / (factorial(r) * factorial(l-2*r)) # fake TODO
  theta = jax_ck(l,lA,lB,PA,PB) * factorial(l) * g**(r-l) / (factorial(r) * factorial(l-2*r)) # real


  #theta = ck(l,lA,lB,PA,PB) * factorial(l) * g**(r-l) / (factorial(r) * factorial(l-2*r))
  #theta = factorial(l) * g**(r-l) / (factorial(r) * factorial(l-2*r))
  return theta

@jax.jit
def jax_ck(j,l,m,a,b):
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
  delta = 1/(4*gP) + 1/(4*gQ)
  gi  = (-1)**l 
  gi *= theta(l,lA,lB,Pi-Ai,Pi-Bi,r,gP) * theta(lp,lC,lD,Qi-Ci,Qi-Di,rp,gQ)
  gi *= (-1)**i * (2 * delta)**(2 * (r + rp))
  gi *= factorial(l + lp - 2 * r - 2 * rp) * delta**i
  gi *= (Pi - Qi)**(l + lp - 2 * (r + rp + i))
  gi /= (4 * delta)**(l + lp) * factorial(i)
  gi /= factorial(l + lp - 2 * (r + rp + i))
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

def Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD):
  """
  Calling intermediate function gi to calculate individual x,y,z components,
  calculate integrals over primitives.
  (bracketed part of Handout 4, Eq. 18)
  """
  gP = a + b
  gQ = c + d
  delta = 1/(4*gP) + 1/(4*gQ)
  RP = gaussian_product(a,b,RA,RB)
  RQ = gaussian_product(c,d,RC,RD)
  ABsq = np.dot(RA-RB,RA-RB)
  CDsq = np.dot(RC-RD,RC-RD)
  PQsq = np.dot(RP-RQ,RP-RQ)
  ssss = prefactor(a,b,c,d,gP,gQ,ABsq,CDsq)
  boysarg = PQsq / (4 * delta)

  # Collect all angular momentum indices. We dont do any computations in the loops, so they are fast
  # every loop variable is ultimately dependent on the 
  # 3 angular momentum components l,m,n for the 4 centers: lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD
  gx_indices = []
  gy_indices = []
  gz_indices = []
  nu_indices = []
  for l in range(0,lA+lB+1):
    for r in range(0,int(l/2)+1):
      for lp in range(0,lC+lD+1):
        for rp in range(0,int(lp/2)+1):
          for i in range(0,int((l+lp-2*r-2*rp)/2)+1):
            gx_indices.append([l,lp,r,rp,i])
  for m in range(0,mA+mB+1):
    for s in range(0,int(m/2)+1):
      for mp in range(0,mC+mD+1):
        for sp in range(0,int(mp/2)+1):
          for j in range(0,int((m+mp-2*s-2*sp)/2)+1):
            gy_indices.append([m,mp,s,sp,j])
  for n in range(0,nA+nB+1):
    for t in range(0,int(n/2)+1):
      for NP in range(0,nC+nD+1):
        for tp in range(0,int(NP/2)+1):
          for k in range(0,int((n+NP-2*t-2*tp)/2)+1):
            gz_indices.append([n,NP,t,tp,k])
  for l in range(0,lA+lB+1):
    for r in range(0,int(l/2)+1):
      for lp in range(0,lC+lD+1):
        for rp in range(0,int(lp/2)+1):
          for i in range(0,int((l+lp-2*r-2*rp)/2)+1):
            for m in range(0,mA+mB+1):
              for s in range(0,int(m/2)+1):
                for mp in range(0,mC+mD+1):
                  for sp in range(0,int(mp/2)+1):
                    for j in range(0,int((m+mp-2*s-2*sp)/2)+1):
                      for n in range(0,nA+nB+1):
                        for t in range(0,int(n/2)+1):
                          for NP in range(0,nC+nD+1):
                            for tp in range(0,int(NP/2)+1):
                              for k in range(0,int((n+NP-2*t-2*tp)/2)+1):
                                nu_indices.append(l+lp+m+mp+n+NP-2*(r+rp+s+sp+t+tp)-(i+j+k))
  

  gx_vec = np.asarray(onp.asarray(gx_indices))
  print('gx',gx_vec.shape)
  gy_vec = np.asarray(onp.asarray(gy_indices))
  print('gy',gy_vec.shape)
  gz_vec = np.asarray(onp.asarray(gz_indices))
  print('gz',gz_vec.shape)
  boys_nu = np.asarray(onp.asarray(nu_indices))
  print('nu',boys_nu.shape)
 
  #gi_vmap function defined above
  gx = gi_vmap(gx_vec[:,0],gx_vec[:,1],gx_vec[:,2],gx_vec[:,3],gx_vec[:,4],lA,lB,RA[0],RB[0],RP[0],gP, lC,lD,RC[0],RD[0],RQ[0],gQ)
  gy = gi_vmap(gy_vec[:,0],gy_vec[:,1],gy_vec[:,2],gy_vec[:,3],gy_vec[:,4],mA,mB,RA[1],RB[1],RP[1],gP, mC,mD,RC[1],RD[1],RQ[1],gQ)
  gz = gi_vmap(gz_vec[:,0],gz_vec[:,1],gz_vec[:,2],gz_vec[:,3],gz_vec[:,4],nA,nB,RA[2],RB[2],RP[2],gP, nC,nD,RC[2],RD[2],RQ[2],gQ)

  F = boys(boys_nu, np.broadcast_to(boysarg, boys_nu.shape))
  #F = np.broadcast_to(1, boys_nu.shape)

  #Gxyz = np.sum(np.einsum('x,y,z->xyz', gx, gy, gz).flatten() * F) #THIS WORKS, but probably smarter contraction
  Gxyz = np.dot(np.einsum('x,y,z->xyz', gx, gy, gz).flatten(),F)  # slightly better


  Gxyz *= ssss 

  # NOTE replaced with prefactor
  #Gxyz *= ( 2 * np.pi**2 ) / ( gP * gQ ) 
  #Gxyz *= np.sqrt( np.pi / ( gP + gQ ) )
  #Gxyz *= np.exp( -(a*b*ABsq)/gP ) 
  #Gxyz *= np.exp( -(c*d*CDsq)/gQ )

  #Na = N(a,lA,mA,nA)
  #Nb = N(b,lB,mB,nB)
  #Nc = N(c,lC,mC,nC)
  #Nd = N(d,lD,mD,nD)

  #Gxyz *= Na * Nb * Nc * Nd
  return Gxyz

@jax.jit
def prefactor(a,b,c,d,gP,gQ,ABsq,CDsq):
  f = ( 2 * np.pi**2 ) / ( gP * gQ ) * np.sqrt( np.pi / ( gP + gQ ) ) * np.exp( -(a*b*ABsq)/gP ) * np.exp(-(c*d*CDsq)/gQ)
  return f
 
#lA,mA,nA = 0,0,1
#lB,mB,nB = 0,0,1
#lC,mC,nC = 0,0,1
#lD,mD,nD = 0,0,1

# This is an (fxyz fxyz | fxyz fxyz) integral. Under this implementation requires 
lA,mA,nA = 2,0,0
lB,mB,nB = 2,0,0
lC,mC,nC = 2,0,0
lD,mD,nD = 2,0,0
a,b,c,d = 0.5, 0.5, 0.5, 0.5
RA,RB,RC,RD = np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9]), np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9])

print(Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD))
#for i in range(1000):
#    print(Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD))
    #Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD)




