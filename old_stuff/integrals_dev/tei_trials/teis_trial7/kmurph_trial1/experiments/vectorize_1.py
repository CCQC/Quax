import jax 
from jax import lax
import jax.numpy as np
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
from jax.experimental import loops

#@jax.jit
def binom(x,y):
    C = factorial(x) / (factorial(x-y) * factorial(y))
    return C

#@jax.jit
def factorial(n):
  with loops.Scope() as s:
    s.num = 1
    s.n = n
    for _ in s.while_range(lambda: s.n >=1):
      s.num = s.num * s.n
      s.n -= 1
    return s.num

#@jax.jit
def theta(l,lA,lB,PA,PB,r,g):
  """
  Calculate the theta factor of the gi term.
  (Handout 4, Eq. 23)
  """
  theta = jax_ck(l,lA,lB,PA,PB) * factorial(l) * g**(r-l) / (factorial(r) * factorial(l-2*r))
  #theta = ck(l,lA,lB,PA,PB) * factorial(l) * g**(r-l) / (factorial(r) * factorial(l-2*r))
  #theta = factorial(l) * g**(r-l) / (factorial(r) * factorial(l-2*r))
  return theta

#@jax.jit
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

gi_vmap = jax.jit(jax.vmap(gi, in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None)))

#print(gi(3,3,1,1,1, 3,3,7.,8.,9.,2.,3,3,2.,3.,4.,5.))
#print(gi(3,3,1,1,1, 3,3,7.,8.,9.,2.,3,3,2.,3.,4.,5.))

#for i in range(10000):
#    gi(3,3,1,1,1, 3,3,7.,8.,9.,2.,3,3,2.,3.,4.,5.)

def gaussian_product(alpha_bra,alpha_ket,A,C):
    '''Gaussian product theorem. Returns center and coefficient of product'''
    R = (alpha_bra * A + alpha_ket * C) / (alpha_bra + alpha_ket)
    return R

@jax.jit
def boys(n,x):
    return 0.5 * (x + 1e-11)**(-(n + 0.5)) * jax.lax.igamma(n + 0.5, x + 1e-11) * np.exp(jax.lax.lgamma(n + 0.5))

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
  
  boysarg = PQsq / (4 * delta)

  # Simplification: pretend the integral is just the sum of all gx. Try to vectorize
  Gxyz = 0
  gx_indices = []
  gy_indices = []
  gz_indices = []
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

  gx_vec = np.asarray(gx_indices)
  gy_vec = np.asarray(gy_indices)
  gz_vec = np.asarray(gz_indices)

  # The Boys function is evaluated for every combination of gx, gy, gz.==> 25 x 25 x 25 = 
  # As a temporary thing to be sure of understanding, just collect indices in nested loops
  nu_s = []
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
                                nu_s.append(l+lp+m+mp+n+NP-2*(r+rp+s+sp+t+tp)-(i+j+k))
 
  #gi_vmap function defined above
  gx = gi_vmap(gx_vec[:,0],gx_vec[:,1],gx_vec[:,2],gx_vec[:,3],gx_vec[:,4],lA,lB,RA[0],RB[0],RP[0],gP, lC,lD,RC[0],RD[0],RQ[0],gQ)
  gy = gi_vmap(gy_vec[:,0],gy_vec[:,1],gy_vec[:,2],gy_vec[:,3],gy_vec[:,4],mA,mB,RA[1],RB[1],RP[1],gP, mC,mD,RC[1],RD[1],RQ[1],gQ)
  gz = gi_vmap(gz_vec[:,0],gz_vec[:,1],gz_vec[:,2],gz_vec[:,3],gz_vec[:,4],nA,nB,RA[2],RB[2],RP[2],gP, nC,nD,RC[2],RD[2],RQ[2],gQ)

  boys_nu = np.asarray(nu_s)
  F = boys(boys_nu, np.broadcast_to(boysarg, boys_nu.shape))

  Gxyz = np.sum(np.einsum('x,y,z->xyz', gx, gy, gz).flatten() * F) #THIS WORKS, but probably smarter contraction
  print(Gxyz)

  #Gxyz = np.einsum(' ', np.einsum('x,y,z->xyz', gx, gy, gz), F)  #does this work?
  #print(Gxyz)

  #components = np.sum(gx) * np.sum(gy) * np.sum(gz) # this works! matches kmurph product gx * gy * gz summing over all loops
  #print(components)


  #print(np.sum(F) * components)

  # TODOkkkkkkkkkkk
  # not sure if this is the way.. seems wrong since each nu is over 15 loops
  # so it cant be constructed from 3 arrays which represent 5 loops each... hmm
  # perhaps a cartesian product can be used
  # TODO use a similar trick to how you compute every possible combinatino of aa + bb in vectorized one electron integrals! #TODO
  # except its more analgous to (aa + bb + cc) where here we do (l + lp) + (m + mp) + (n + np), which that alone will give proper shape when flattened, 
  # then add in the other terms with r,s,t,i,j,k
  # TODO

  #nu    = l+lp+m+mp+n+np-2*(r+rp+s+sp+t+tp)-(i+j+k)
  #boys_nu = gx_vec[:,0] + gx_vec[:,1] + gy_vec[:,0] + gy_vec[:,1] + gz_vec[:,0] + gz_vec[:,1] \
  #        - 2 * (gx_vec[:,2] + gx_vec[:,3] + gy_vec[:,2] + gy_vec[:,3] + gz_vec[:,2] + gz_vec[:,3]) \
  #        - (gx_vec[:,4] +  gy_vec[:,4] + gz_vec[:,4])

  #test = boys(boys_nu, np.broadcast_to(boysarg, boys_nu.shape))
  #print(test)

  #print(np.sum(test) * components)
  #print(boys_nu.shape)
  #print(gx.shape)

  #print(Gxyz)

  #test = np.asarray(indices)
  #print(test)
  #result = np.sum(gi_vmap(test[:,0],test[:,1],test[:,2],test[:,3],test[:,4],lA,lB,RA[0],RB[0],RP[0],gP, lC,lD,RC[0],RD[0],RQ[0],gQ))
  #print(result)


  #test = gi_vmap(np.arange(3),np.arange(3),np.arange(3),np.arange(3),np.arange(3),lA,lB,RA[0],RB[0],RP[0],gP, lC,lD,RC[0],RD[0],RQ[0],gQ )
  #finish = np.sum(test)

  # Need to mimic loop variables now
  # Challenge is the loop variables are coupled together... cant get around it with a cartesian product.. right?
  # Does the same issue occur without rounding?
  # yeah idk, the l and lp play nice but the others are interdependent. could simplify the loop but that's it
  #l = np.arange(lA + lB + 1)
  #r = np.floor(l / 2)
  #lp = np.arange(lC + lD + 1)
  #rp = np.floor(lp / 2)
  #i = np.floor((l + lp - 2 * r - 2 * rp) / 2)

  #r = np.arange(int(l.shape[0]/2) + 1)
  #print(r)
  #rp = np.arange(int(lp.shape[0]/2) + 1)
  #print(rp)
  #i = np.arange(int((l.shape[0]+lp.shape[0]-2*r.shape[0]-2*rp.shape[0])/2) + 1)
  #print(i)
  #print(l, r, lp, rp, i)

  #what = cartesian_product(l,lp)
  #print(what.shape)
  #print(what)



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
#lA,mA,nA = 0,0,1
#lB,mB,nB = 0,0,1
#lC,mC,nC = 0,0,1
#lD,mD,nD = 0,0,1

lA,mA,nA = 1,1,1
lB,mB,nB = 1,1,1
lC,mC,nC = 1,1,1
lD,mD,nD = 1,1,1
a,b,c,d = 0.5, 0.5, 0.5, 0.5
RA,RB,RC,RD = np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9]), np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9])

Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD)




