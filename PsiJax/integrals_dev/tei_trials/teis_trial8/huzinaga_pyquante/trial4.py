import jax 
from jax import lax
import numpy as onp
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
from jax.experimental import loops
#https://github.com/rpmuller/pyquante2/blob/master/pyquante2/ints/two.py

@jax.jit
def factorial(n):
  n = n.astype(float)
  return jax.lax.exp(jax.lax.lgamma(n + 1))

@jax.jit
def binom(x,y):
    C = factorial(x) / (factorial(x-y) * factorial(y))
    return C

#val = fB(i1,l1,l2,Px,Ax,Bx,r1,gamma1) \
def fB(i,l1,l2,P,A,B,r,g): 
    return binomial_prefactor(i,l1,l2,P-A,P-B) * B0(i,r,g)

def B0(i,r,g): 
    return fact_ratio2(i,r) * (4*g)**(r-i)

def fact_ratio2(a,b):
    return factorial(a)/factorial(b)/factorial(a-2*b)

@jax.jit
def binomial_prefactor(s,ia,ib,xpa,xpb):
    with loops.Scope() as L:
        L.total = 0.0
        L.s = s 
        L.ia = ia
        L.ib = ib
        L.xpa = xpa
        L.xpb = xpb

        L.t = s + 1
    
        for _ in L.while_range(lambda: L.t > -1):
            for _ in L.cond_range( (L.s - L.ia) <= L.t):
                for _ in L.cond_range(L.t <= L.ib):
                    L.total += binom(L.ia,L.s - L.t) * binom(L.ib,L.t) * \
                               L.xpa**(L.ia-L.s+L.t) * L.xpb**(L.ib-L.t)
            L.t -= 1
        return L.total

def gaussian_product(alpha1,A,alpha2,B):
    return (alpha1*A+alpha2*B)/(alpha1+alpha2)

#@jax.jit
#def boys(m,x):
#    return 0.5 * (x + 1e-11)**(-(m + 0.5)) * jax.lax.igamma(m + 0.5, x + 1e-11) * np.exp(jax.lax.lgamma(m + 0.5))

@jax.jit
def boys(n,x):
    result = np.where(x < 1e-7, 1 / (2 * n + 1) - x *  (1 / (2 * n + 3)), 
                      0.5 * (x)**(-(n + 0.5)) * jax.lax.igamma(n + 0.5,x) * np.exp(jax.lax.lgamma(n + 0.5)))
    return result

def coulomb_repulsion(xyza,norma,lmna,alphaa,xyzb,normb,lmnb,alphab,xyzc,normc,lmnc,alphac,xyzd,normd,lmnd,alphad):
    """
    Return the coulomb repulsion between four primitive gaussians a,b,c,d with the given origin
,    x,y,z, normalization constants norm, angular momena l,m,n, and exponent alpha.
    >>> p1 = array((0.,0.,0.),'d')
    >>> p2 = array((0.,0.,1.),'d')
    >>> lmn = (0,0,0)
    >>> isclose(coulomb_repulsion(p1,1.,lmn,1.,p1,1.,lmn,1.,p1,1.,lmn,1.,p1,1.,lmn,1.),4.373355)
    True
    >>> isclose(coulomb_repulsion(p1,1.,lmn,1.,p1,1.,lmn,1.,p2,1.,lmn,1.,p2,1.,lmn,1.),3.266127)
    True
    >>> isclose(coulomb_repulsion(p1,1.,lmn,1.,p2,1.,lmn,1.,p1,1.,lmn,1.,p2,1.,lmn,1.),1.6088672)
    True
    """
    la,ma,na = lmna
    lb,mb,nb = lmnb
    lc,mc,nc = lmnc
    ld,md,nd = lmnd
    xa,ya,za = xyza
    xb,yb,zb = xyzb
    xc,yc,zc = xyzc
    xd,yd,zd = xyzd

    rab2 = np.dot(xyza-xyzb,xyza-xyzb)
    rcd2 = np.dot(xyzc-xyzd,xyzc-xyzd)
    xyzp = gaussian_product(alphaa,xyza,alphab,xyzb)
    xp,yp,zp = xyzp
    xyzq = gaussian_product(alphac,xyzc,alphad,xyzd)
    xq,yq,zq = xyzq
    rpq2 = np.dot(xyzp-xyzq,xyzp-xyzq)
    gamma1 = alphaa+alphab
    gamma2 = alphac+alphad
    delta = 0.25*(1/gamma1+1/gamma2)

    Bx = B_array(la,lb,lc,ld,xp,xa,xb,xq,xc,xd,gamma1,gamma2,delta)
    By = B_array(ma,mb,mc,md,yp,ya,yb,yq,yc,yd,gamma1,gamma2,delta)
    Bz = B_array(na,nb,nc,nd,zp,za,zb,zq,zc,zd,gamma1,gamma2,delta)
    
    boys_arg = 0.25*rpq2/delta

    sum = 0.
    for I in range(la+lb+lc+ld+1):
        for J in range(ma+mb+mc+md+1):
            for K in range(na+nb+nc+nd+1):
                sum = sum + Bx[I] * By[J] * Bz[K] * boys(I+J+K,boys_arg)

    return 2*jax.lax.pow(np.pi,2.5)/(gamma1*gamma2*np.sqrt(gamma1+gamma2)) \
           *np.exp(-alphaa*alphab*rab2/gamma1) \
           *np.exp(-alphac*alphad*rcd2/gamma2)*sum*norma*normb*normc*normd

@jax.jit
def B_term(i1,i2,r1,r2,u,l1,l2,l3,l4,Px,Ax,Bx,Qx,Cx,Dx,gamma1,gamma2,delta):
    val = fB(i1,l1,l2,Px,Ax,Bx,r1,gamma1) \
           * (-1)**i2 * fB(i2,l3,l4,Qx,Cx,Dx,r2,gamma2) \
           * (-1)**u * fact_ratio2(i1+i2-2*(r1+r2),u) \
           * (Qx-Px)**(i1+i2-2*(r1+r2)-2*u) \
           / delta**(i1+i2-2*(r1+r2)-u)
    return val

@jax.jit
def B_array(l1,l2,l3,l4,p,a,b,q,c,d,g1,g2,delta):
    # This makes arrays with argument-dependent shapes. need fix size for jit compiling
    # Hard code only up to f functions (fxxx, fxxx | fxxx, fxxx) = l1 + l2 + l3 + l4
    #Imax = l1+l2+l3+l4+1
    with loops.Scope() as s:
      s.B = np.zeros(12)

      s.i2 = 0
      s.r1 = 0
      s.r2 = 0
      s.u = 0 

      s.i1 = l1 + l2  
      for _ in s.while_range(lambda: s.i1 > -1):   
        s.r1 = s.i1 // 2
        for _ in s.while_range(lambda: s.r1 > -1):
          s.i2 = l3 + l4 
          for _ in s.while_range(lambda: s.i2 > -1):
            s.r2 = s.i2 // 2
            for _ in s.while_range(lambda: s.r2 > -1):
              s.u = (s.i1 + s.i2) // 2 - s.r1 - s.r2 
              for _ in s.while_range(lambda: s.u > -1):
                I = s.i1 + s.i2 - 2 * (s.r1 + s.r2) - s.u 
                term = B_term(s.i1,s.i2,s.r1,s.r2,s.u,l1,l2,l3,l4,p,a,b,q,c,d,g1,g2,delta) 
                s.B = jax.ops.index_add(s.B, I, term)
     
                s.u -= 1
              s.r2 -= 1
            s.i2 -= 1
          s.r1 -= 1
        s.i1 -= 1
      return s.B



xyza = np.array([0.0,0.1,0.9])
xyzb = np.array([0.0,-0.1,-0.9])
xyzc = np.array([0.0,-0.1, 0.9])
xyzd = np.array([0.0,-0.1,-0.9])
norma = normb = normc = normd = 1.0
lmna = (1,1,1)
lmnb = (1,1,1)
lmnc = (1,1,1)
lmnd = (1,1,1)
alphaa,alphab,alphac,alphad = 0.5, 0.4, 0.3, 0.2
#M = 0

result = coulomb_repulsion(xyza,norma,lmna,alphaa,xyzb,normb,lmnb,alphab,xyzc,normc,lmnc,alphac,xyzd,normd,lmnd,alphad)
print(result)

