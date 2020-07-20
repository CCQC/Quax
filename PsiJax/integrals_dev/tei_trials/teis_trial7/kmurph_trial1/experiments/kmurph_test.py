# Original Kmurph code, but just for needed gi() function
import math
import numpy as np
from scipy import special 

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


print(gi(3,3,1,1,1, 3,3,7.,8.,9.,2.,3,3,2.,3.,4.,5.))
#for i in range(10000):
#    gi(3,3,1,1,1, 3,3,7.,8.,9.,2.,3,3,2.,3.,4.,5.)

