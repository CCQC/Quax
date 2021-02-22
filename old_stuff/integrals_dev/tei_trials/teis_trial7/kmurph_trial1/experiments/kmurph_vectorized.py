import math
import numpy as np
from scipy import special, special 

import jax
import jax.numpy as jnp

def buildG(basis, G, K):
  """
  Calling the intermediate function Gxyz to calculate integrals over primitives,
  compute all elements of the G = (AB|CD) 4-dimensional matrix.
  (part of Handout 4, Eq. 18)
  """
  Ntei = 0 # create a two-electron integral counter, as TEIs can take a long time
  for A, bA in enumerate(basis): # retrieve atomic orbital A from basis
    for B, bB in enumerate(basis):
      for C, bC in enumerate(basis):
        for D, bD in enumerate(basis):
  
          Ntei +=1
          if Ntei % 250 == 0:
            print ('Computed '+ str(Ntei) + ' of ' + str(K**4) + ' integrals.')
  
          for a, dA in zip(bA['a'],bA['d']): # retrieve alpha and contract coefficients for atomic orbital A
            for b, dB in zip(bB['a'],bB['d']):
              for c, dC in zip(bC['a'],bC['d']):
                for d, dD in zip(bD['a'],bD['d']):
   
                  RA, RB, RC, RD = bA['R'], bB['R'], bC['R'], bD['R'] # hold coordinates of AOs A, B, C, and D
  
                  ## variables for angular momenta, in terms of x,y,z, for each orbital
                  lA, mA, nA = bA['l'], bA['m'], bA['n']
                  lB, mB, nB = bB['l'], bB['m'], bB['n']
                  lC, mC, nC = bC['l'], bC['m'], bC['n']
                  lD, mD, nD = bD['l'], bD['m'], bD['n']
  
                  tei  = dA * dB * dC * dD # multiply together the contraction coefficients
                  tei *= Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD) # multiply by integral over primitives
   
                  G[A,B,C,D] += tei

  return G

def Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD):
  """
  Calling intermediate function gi to calculate individual x,y,z components,
  calculate integrals over primitives.
  (bracketed part of Handout 4, Eq. 18)
  """
  gP = a + b
  gQ = c + d

  delta = 1/(4*gP) + 1/(4*gQ)

  RP = gaussianProduct(a,RA,b,RB,gP)
  RQ = gaussianProduct(c,RC,d,RD,gQ)

  ABsq = IJsq(RA,RB)
  CDsq = IJsq(RC,RD)
  PQsq = IJsq(RP,RQ)

  boysarg = PQsq / (4 * delta)

  # Collect all angular-momentum-related indices 
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
                          for nq in range(0,nC+nD+1):
                            for tp in range(0,int(nq/2)+1):
                              for k in range(0,int((n+nq-2*t-2*tp)/2)+1):
                                gz_indices.append([n,nq,t,tp,k])
                                nu_indices.append(l+lp+m+mp+n+nq-2*(r+rp+s+sp+t+tp)-(i+j+k))

  boys_nu = np.asarray(nu_indices)
  # Compute all boys function evaluations at once on vector quantities
  F = boys(boys_nu, np.broadcast_to(boysarg, boys_nu.shape))

  # All combinations of indices [l,lp,r,rp,i], etc in loops
  gx = np.asarray(gx_indices)
  gy = np.asarray(gy_indices)
  gz = np.asarray(gz_indices)

  shp = gx.shape
  # Compute all gi components simultaneously; we must broadcast constant values (lA, Ai, etc)
  # to the same size as the varying angular momentum indices l, lp, r, rp, i
  #  l       lp      r       rp      i        lA, lB, Ai, Bi, Pi, gP, lC, lD, Ci, Di, Qi, gQ
  gx_all = gi(gx[:,0],gx[:,1],gx[:,2],gx[:,3],gx[:,4], 
              np.broadcast_to(lA,shp),np.broadcast_to(lB,shp),
              np.broadcast_to(RA[0],shp),np.broadcast_to(RB[0],shp),
              np.broadcast_to(RP[0],shp),np.broadcast_to(gP,shp),
              np.broadcast_to(lC,shp),np.broadcast_to(lD,shp),
              np.broadcast_to(RC[0],shp),np.broadcast_to(RD[0],shp),
              np.broadcast_to(RQ[0],shp),np.broadcast_to(gQ,shp))

  #gy_all = gi(gy[:,0],gy[:,1],gy[:,2],gy[:,3],gy[:,4], 
  #            np.broadcast_to(mA,shp),np.broadcast_to(mB,shp),
  #            np.broadcast_to(Ai,shp),np.broadcast_to(Bi,shp),
  #            np.broadcast_to(Pi,shp),np.broadcast_to(gP,shp),
  #            np.broadcast_to(lC,shp),np.broadcast_to(lD,shp),
  #            np.broadcast_to(Ci,shp),np.broadcast_to(Di,shp),
  #            np.broadcast_to(Qi,shp),np.broadcast_to(gQ,shp))

  #gi_vmap = jax.vmap(gi, in_axes=(0,0,0,0,0,None,None,None,None,None,None,None,None,None,None,None,None))


  #Gxyz = np.sum(np.einsum('x,y,z->xyz', gx, gy, gz).flatten() * F) #THIS WORKS, but probably smarter contraction

  #Gxyz *= ( 2 * math.pi**2 ) / ( gP * gQ ) 
  #Gxyz *= math.sqrt( math.pi / ( gP + gQ ) )
  #Gxyz *= math.exp( -(a*b*ABsq)/gP ) 
  #Gxyz *= math.exp( -(c*d*CDsq)/gQ )

  #Na = N(a,lA,mA,nA)
  #Nb = N(b,lB,mB,nB)
  #Nc = N(c,lC,mC,nC)
  #Nd = N(d,lD,mD,nD)

  #Gxyz *= Na * Nb * Nc * Nd

  return Gxyz

def gi(l,lp,r,rp,i, lA,lB,Ai,Bi,Pi,gP, lC,lD,Ci,Di,Qi,gQ):
  """
  Calculate the i-th coordinate component of the integral over primitives.
  (Handout 4, Eq. 22)
  """
  delta = 1/(4*gP) + 1/(4*gQ)
  gi  = (-1)**l 
  gi *= theta(l,lA,lB,Pi-Ai,Pi-Bi,r,gP) * theta(lp,lC,lD,Qi-Ci,Qi-Di,rp,gQ)
  gi *= (-1)**i * (2*delta)**(2*(r+rp))
  gi *= special.factorial(l+lp-2*r-2*rp,exact=True) * delta**i
  gi *= (Pi-Qi)**(l+lp-2*(r+rp+i))
  gi /= (4*delta)**(l+lp) * special.factorial(i,exact=True)
  gi /= special.factorial(l+lp-2*(r+rp+i),exact=True)
  return gi

def theta(l,lA,lB,PA,PB,r,g):
  """
  Calculate the theta factor of the gi term.
  (Handout 4, Eq. 23)
  """
  theta  = ck(l,lA,lB,PA,PB) 
  theta *= special.factorial(l,exact=True) * g**(r-l) 
  theta /= special.factorial(r,exact=True) * special.factorial(l-2*r,exact=True) 
  return theta

def ck(j,l,m,a,b):
  """
  Calculate the coefficient 'ck' factor within the theta expression,
  associated with a third center between position vectors
  of the nuclei A and B.
  (Handout 4, Eq. 8)
  """
  coefficient = 0.0
 
  for k in range(0,l+1):
    for i in range(0,m+1):
      if i + k == j:
        coefficient += special.binom(l,k) * special.binom(m,i) * a**(l-k) * b**(m-i)

  return coefficient

def N(a,l,m,n):
  """
  Calculate the normalization factors.
  (Handout 4, Eq. 9)
  """
  N  = (4*a)**(l+m+n)
  N /= special.factorial2(2*l-1,exact=True) * special.factorial2(2*m-1,exact=True) * special.factorial2(2*n-1,exact=True)
  N *= ((2*a)/math.pi)**(1.5)
  N  = N**(0.5)

  return N

def BoysFunction(nu, x):
  """
  The analytical function coded herein was suggested by CCL forum; similar to
  result when evaluating the Boys Function integral in Mathematica. 
  Depends on gamma functions, which are easily computed with SciPy's 
  library of special functions.
  """
  if x < 1e-7:
    return (2*nu+1)**(-1) - x*(2*nu+3)**(-1) # (Handout 4, Eq. 17)
  else:
    return (1/2) * x**(-(nu+0.5)) * special.gamma(nu+0.5) * special.gammainc(nu+0.5,x) # (Handout 4, Eq. 16)

def boys(nu,x):
    return (1/2) * (x + 1e-11)**(-(nu+0.5)) * special.gamma(nu+0.5) * special.gammainc(nu+0.5,x + 1e-11) # (Handout 4, Eq. 16)

def gaussianProduct(a,RA,b,RB,g):
  """
  The product of two Gaussians is a third Gaussian.
  (Handout 4, Eq. 5)
  """
  P = []
  for i in range(3):
    P.append( (a*RA[i]+b*RB[i])/g )

  return P

def IJsq(RI,RJ):
  """
  Calculate the square of the distance between two points.
  (Handout 4, Eq. 6)
  """
  return sum( (RI[i]-RJ[i])**2 for i in (0,1,2) )


lA,mA,nA = 1,1,1
lB,mB,nB = 1,1,1
lC,mC,nC = 1,1,1
lD,mD,nD = 1,1,1
a,b,c,d = 0.5, 0.5, 0.5, 0.5
RA,RB,RC,RD = np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9]), np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9])

for i in range(10):
    print(Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD))
