import numpy as np
from math import factorial
from scipy import special
#https://github.com/rpmuller/pyquante2/blob/master/pyquante2/ints/two.py

def binomial(n,k):
    """
    Binomial coefficient
    >>> binomial(5,2)
    10
    >>> binomial(10,5)
    252
    """
    if n==k: return 1
    assert n>k, "Attempting to call binomial(%d,%d)" % (n,k)
    return factorial(n)//(factorial(k)*factorial(n-k))

def binomial_prefactor(s,ia,ib,xpa,xpb):
    """
    The integral prefactor containing the binomial coefficients from Augspurger and Dykstra.
    >>> binomial_prefactor(0,0,0,0,0)
    1
    """
    total= 0
    for t in range(s+1):
        if s-ia <= t <= ib:
            total +=  binomial(ia,s-t)*binomial(ib,t)* \
                     pow(xpa,ia-s+t)*pow(xpb,ib-t)
    return total


def gaussian_product_center(alpha1,A,alpha2,B):
    return (alpha1*A+alpha2*B)/(alpha1+alpha2)

def Fgamma(m,x):
    ''' This is literally the same as your boys function, despite it's weridness '''
    SMALL=1e-12
    x = max(x,SMALL)
    return 0.5*pow(x,-m-0.5)*gamm_inc_scipy(m+0.5,x)

def gamm_inc_scipy(a,x):
    return special.gamma(a)*special.gammainc(a,x)

def norm2(a): return np.dot(a,a)

def coulomb_repulsion(xyza,norma,lmna,alphaa,
                      xyzb,normb,lmnb,alphab,
                      xyzc,normc,lmnc,alphac,
                      xyzd,normd,lmnd,alphad):
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

    rab2 = norm2(xyza-xyzb)
    rcd2 = norm2(xyzc-xyzd)
    xyzp = gaussian_product_center(alphaa,xyza,alphab,xyzb)
    xp,yp,zp = xyzp
    xyzq = gaussian_product_center(alphac,xyzc,alphad,xyzd)
    xq,yq,zq = xyzq
    rpq2 = norm2(xyzp-xyzq)
    gamma1 = alphaa+alphab
    gamma2 = alphac+alphad
    delta = 0.25*(1/gamma1+1/gamma2)

    Bx = B_array(la,lb,lc,ld,xp,xa,xb,xq,xc,xd,gamma1,gamma2,delta)
    By = B_array(ma,mb,mc,md,yp,ya,yb,yq,yc,yd,gamma1,gamma2,delta)
    Bz = B_array(na,nb,nc,nd,zp,za,zb,zq,zc,zd,gamma1,gamma2,delta)

    sum = 0.
    for I in range(la+lb+lc+ld+1):
        for J in range(ma+mb+mc+md+1):
            for K in range(na+nb+nc+nd+1):
                sum = sum + Bx[I]*By[J]*Bz[K]*Fgamma(I+J+K,0.25*rpq2/delta)

    return 2*pow(np.pi,2.5)/(gamma1*gamma2*np.sqrt(gamma1+gamma2)) \
           *np.exp(-alphaa*alphab*rab2/gamma1) \
           *np.exp(-alphac*alphad*rcd2/gamma2)*sum*norma*normb*normc*normd

def B_term(i1,i2,r1,r2,u,l1,l2,l3,l4,Px,Ax,Bx,Qx,Cx,Dx,gamma1,gamma2,delta):
    "THO eq. 2.22"
    val= fB(i1,l1,l2,Px,Ax,Bx,r1,gamma1) \
           *pow(-1,i2)*fB(i2,l3,l4,Qx,Cx,Dx,r2,gamma2) \
           *pow(-1,u)*fact_ratio2(i1+i2-2*(r1+r2),u) \
           *pow(Qx-Px,i1+i2-2*(r1+r2)-2*u) \
           /pow(delta,i1+i2-2*(r1+r2)-u)
    return val

def B_array(l1,l2,l3,l4,p,a,b,q,c,d,g1,g2,delta):
    Imax = l1+l2+l3+l4+1
    B = [0]*Imax
    for i1 in range(l1+l2+1):
        for i2 in range(l3+l4+1):
            for r1 in range(i1//2+1):
                for r2 in range(i2//2+1):
                    for u in range((i1+i2)//2-r1-r2+1):
                        I = i1+i2-2*(r1+r2)-u
                        B[I] = B[I] + B_term(i1,i2,r1,r2,u,l1,l2,l3,l4,p,a,b,q,c,d,g1,g2,delta)
    return B

def fB(i,l1,l2,P,A,B,r,g): 
    return binomial_prefactor(i,l1,l2,P-A,P-B)*B0(i,r,g)
def B0(i,r,g): 
    return fact_ratio2(i,r)*pow(4*g,r-i)
def fact_ratio2(a,b): 
    return factorial(a)/factorial(b)/factorial(a-2*b)


#xyza = np.array([0.0,0.1,0.9])
#xyzb = np.array([0.0,-0.1,-0.9])
#xyzc = np.array([0.0,-0.1, 0.9])
#xyzd = np.array([0.0,-0.1,-0.9])
#norma = normb = normc = normd = 1.0
#lmna = (1,0,0)
#lmnb = (0,1,0)
#lmnc = (0,0,1)
#lmnd = (1,0,0)
#alphaa,alphab,alphac,alphad = 0.5, 0.4, 0.3, 0.2
#M = 0
#
#result = coulomb_repulsion(xyza,norma,lmna,alphaa,xyzb,normb,lmnb,alphab,xyzc,normc,lmnc,alphac,xyzd,normd,lmnd,alphad)


#print("Fact ratio 2 test")
#print(fact_ratio2(3,1))
#print(fact_ratio2(4,2))
#print(fact_ratio2(3,1))
#print(fact_ratio2(5,1))
#print(fact_ratio2(0,0))

