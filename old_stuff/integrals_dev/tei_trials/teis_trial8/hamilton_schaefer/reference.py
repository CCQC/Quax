import numpy as np
from scipy.special import gamma,gammainc

def gaussian_product_center(alpha1,A,alpha2,B):
    return (alpha1*A+alpha2*B)/(alpha1+alpha2)

# He's doing some weird shiz here for the boys function, I asssume that 
# the below expression is an ssss[0] integral
#            norma*normb*normc*normd*Kab*Kcd/np.sqrt(zeta+eta)*Fgterms[im]

def Fgamma(m,x):
    ''' This is literally the same as your boys function, despite it's weridness '''
    SMALL=1e-12
    x = max(x,SMALL)
    return 0.5*pow(x,-m-0.5)*gamm_inc_scipy(m+0.5,x)

def gamm_inc_scipy(a,x):
    return gamma(a)*gammainc(a,x)

def boys(n,x):
    result = np.where(x < 1e-7, 1 / (2 * n + 1) - x *  (1 / (2 * n + 3)), 
                      0.5 * (x)**(-(n + 0.5)) * gammainc(n + 0.5,x) * gamma(n + 0.5))
    return result

def vrr(xyza,norma,lmna,alphaa, xyzb,normb,alphab, xyzc,normc,lmnc,alphac, xyzd,normd,alphad,M):

    la,ma,na = lmna
    lc,mc,nc = lmnc
    xa,ya,za = xyza
    xb,yb,zb = xyzb
    xc,yc,zc = xyzc
    xd,yd,zd = xyzd

    px,py,pz = xyzp = gaussian_product_center(alphaa,xyza,alphab,xyzb)
    qx,qy,qz = xyzq = gaussian_product_center(alphac,xyzc,alphad,xyzd)
    zeta,eta = float(alphaa+alphab),float(alphac+alphad)
    wx,wy,wz = xyzw = gaussian_product_center(zeta,xyzp,eta,xyzq)

    rab2 = pow(xa-xb,2) + pow(ya-yb,2) + pow(za-zb,2)
    Kab = np.sqrt(2)*pow(np.pi,1.25)/(alphaa+alphab)\
          *np.exp(-alphaa*alphab/(alphaa+alphab)*rab2)
    rcd2 = pow(xc-xd,2) + pow(yc-yd,2) + pow(zc-zd,2)
    Kcd = np.sqrt(2)*pow(np.pi,1.25)/(alphac+alphad)\
          *np.exp(-alphac*alphad/(alphac+alphad)*rcd2)
    rpq2 = pow(px-qx,2) + pow(py-qy,2) + pow(pz-qz,2)
    T = zeta*eta/(zeta+eta)*rpq2

    mtot = la+ma+na+lc+mc+nc+M

    Fgterms = [0]*(mtot+1)
    #print(Fgterms)
    Fgterms[mtot] = Fgamma(mtot,T)
    for im in range(mtot-1,-1,-1):
        Fgterms[im]=(2.*T*Fgterms[im+1]+np.exp(-T))/(2.*im+1)

    # Todo: setup this as a regular array

    # Store the vrr values as a 7 dimensional array
    # vrr_terms[la,ma,na,lc,mc,nc,m]
    vrr_terms = {}
    for im in range(mtot+1):
        vrr_terms[0,0,0,0,0,0,im] = (
            norma*normb*normc*normd*Kab*Kcd/np.sqrt(zeta+eta)*Fgterms[im]
            )

    # Todo: use itertools.product() for the nested for loops
    for i in range(la):
        for im in range(mtot-i):
            vrr_terms[i+1,0,0, 0,0,0, im] = (
                (px-xa)*vrr_terms[i,0,0, 0,0,0, im]
                + (wx-px)*vrr_terms[i,0,0, 0,0,0, im+1]
                )
            if i: # theres an if i so the array doesnt go out of range and to make sure redundant computation doesnt occur
                vrr_terms[i+1,0,0, 0,0,0, im] += (
                    i/2./zeta*( vrr_terms[i-1,0,0, 0,0,0, im]
                               - eta/(zeta+eta)*vrr_terms[i-1,0,0, 0,0,0, im+1]
                               ))

    for j in range(ma):
        for i in range(la+1):
            for im in range(mtot-i-j):
                vrr_terms[i,j+1,0, 0,0,0, im] = (
                    (py-ya)*vrr_terms[i,j,0, 0,0,0, im]
                    + (wy-py)*vrr_terms[i,j,0, 0,0,0, im+1]
                    )
                if j:
                    vrr_terms[i,j+1,0, 0,0,0, im] += (
                        j/2./zeta*(vrr_terms[i,j-1,0, 0,0,0, im]
                                  - eta/(zeta+eta)
                                  *vrr_terms[i,j-1,0, 0,0,0, im+1]
                                  ))

    for k in range(na):
        for j in range(ma+1):
            for i in range(la+1):
                for im in range(mtot-i-j-k):
                    vrr_terms[i,j,k+1, 0,0,0, im] = (
                        (pz-za)*vrr_terms[i,j,k, 0,0,0, im]
                        + (wz-pz)*vrr_terms[i,j,k, 0,0,0, im+1]
                        )
                    if k:
                        vrr_terms[i,j,k+1, 0,0,0, im] += (
                            k/2./zeta*(vrr_terms[i,j,k-1, 0,0,0, im]
                                      - eta/(zeta+eta)
                                      *vrr_terms[i,j,k-1, 0,0,0, im+1]
                                      ))

    for q in range(lc):
        for k in range(na+1):
            for j in range(ma+1):
                for i in range(la+1):
                    for im in range(mtot-i-j-k-q):
                        vrr_terms[i,j,k, q+1,0,0, im] = (
                            (qx-xc)*vrr_terms[i,j,k, q,0,0, im]
                            + (wx-qx)*vrr_terms[i,j,k, q,0,0, im+1]
                            )
                        if q:
                            vrr_terms[i,j,k, q+1,0,0, im] += (
                                q/2./eta*(vrr_terms[i,j,k, q-1,0,0, im]
                                         - zeta/(zeta+eta)
                                         *vrr_terms[i,j,k, q-1,0,0, im+1]
                                         ))
                        if i:
                            vrr_terms[i,j,k, q+1,0,0, im] += (
                                i/2./(zeta+eta)*vrr_terms[i-1,j,k, q,0,0, im+1]
                                )

    for r in range(mc):
        for q in range(lc+1):
            for k in range(na+1):
                for j in range(ma+1):
                    for i in range(la+1):
                        for im in range(mtot-i-j-k-q-r):
                            vrr_terms[i,j,k, q,r+1,0, im] = (
                                (qy-yc)*vrr_terms[i,j,k, q,r,0, im]
                                + (wy-qy)*vrr_terms[i,j,k, q,r,0, im+1]
                                )
                            if r:
                                vrr_terms[i,j,k, q,r+1,0, im] += (
                                    r/2./eta*(vrr_terms[i,j,k, q,r-1,0, im]
                                             - zeta/(zeta+eta)
                                             *vrr_terms[i,j,k, q,r-1,0, im+1]
                                             ))
                            if j:
                                vrr_terms[i,j,k, q,r+1,0, im] += (
                                    j/2./(zeta+eta)*vrr_terms[i,j-1,k,q,r,0,im+1]
                                    )

    for s in range(nc):
        for r in range(mc+1):
            for q in range(lc+1):
                for k in range(na+1):
                    for j in range(ma+1):
                        for i in range(la+1):
                            for im in range(mtot-i-j-k-q-r-s):
                                vrr_terms[i,j,k,q,r,s+1,im] = (
                                    (qz-zc)*vrr_terms[i,j,k,q,r,s,im]
                                    + (wz-qz)*vrr_terms[i,j,k,q,r,s,im+1]
                                    )
                                if s:
                                    vrr_terms[i,j,k,q,r,s+1,im] += (
                                        s/2./eta*(vrr_terms[i,j,k,q,r,s-1,im]
                                                 - zeta/(zeta+eta)
                                                 *vrr_terms[i,j,k,q,r,s-1,im+1]
                                                 ))
                                if k:
                                    vrr_terms[i,j,k,q,r,s+1,im] += (
                                        k/2./(zeta+eta)*vrr_terms[i,j,k-1,q,r,s,im+1]
                                        )
    # To do full 8 term OS, would have to add 8-loop, 9-loop, ..., 13-loop. This probably bad idea.
    # TODO consider if you can get mulitple integrals in a block at once by saving the array
    return vrr_terms[la,ma,na,lc,mc,nc,M]


xyza = np.array([0.0,0.1,0.9])
xyzb = np.array([0.0,-0.1,-0.9])
xyzc = np.array([0.0,-0.1, 0.9])
xyzd = np.array([0.0,-0.1,-0.9])
norma = normb = normc = normd = 1.0
lmna = (1,2,1)
lmnb = (0,0,0)
lmnc = (1,0,0)
lmnd = (0,0,0)
alphaa,alphab,alphac,alphad = 0.5, 0.4, 0.3, 0.2
M = 0

# note faster to do 6 units of angular momentum on one compojnent for A and C than split the angular momentum evenly
result = vrr(xyza,norma,lmna,alphaa, xyzb,normb,alphab, xyzc,normc,lmnc,alphac, xyzd,normd,alphad,M)
print(result)
#for i in range(10000):
#    result = vrr(xyza,norma,lmna,alphaa, xyzb,normb,alphab, xyzc,normc,lmnc,alphac, xyzd,normd,alphad,M)
# need to find one-to-one correspondance with (a0|b0) special case of full OS scheme 






