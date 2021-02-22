import numpy as np 
from scipy import special

def boys(m,x):
    #return 0.5 * (x + 1e-11)**(-(m + 0.5)) * jax.lax.igamma(m + 0.5, x + 1e-11) * np.exp(jax.lax.lgamma(m + 0.5))
    return 0.5 * (x + 1e-11)**(-(m + 0.5)) * special.gammainc(m + 0.5, x + 1e-11) * special.gamma(m + 0.5)

def gaussian_product_center(alpha1,A,alpha2,B):
    return (alpha1 * A + alpha2 * B) / (alpha1 + alpha2)

# ME: Removing normalization constants since they do nothing
def vrr(xyza,lmna,alphaa, xyzb,alphab, xyzc,lmnc,alphac, xyzd,alphad,M):
    la,ma,na = lmna
    lc,mc,nc = lmnc
    xa,ya,za = xyza
    xb,yb,zb = xyzb
    xc,yc,zc = xyzc
    xd,yd,zd = xyzd

    px,py,pz = xyzp = gaussian_product_center(alphaa,xyza,alphab,xyzb)
    qx,qy,qz = xyzq = gaussian_product_center(alphac,xyzc,alphad,xyzd)
    zeta = alphaa + alphab
    eta = alphac + alphad
    wx,wy,wz = xyzw = gaussian_product_center(zeta,xyzp,eta,xyzq)

    rab2 = (xa-xb)**2 + (ya-yb)**2 + (za-zb)**2
    Kab = np.sqrt(2) * np.pi**1.25 / (alphaa+alphab)  \
          * np.exp(-alphaa * alphab / (alphaa + alphab) * rab2)
    rcd2 = (xc-xd)**2 + (yc-yd)**2 + (zc-zd)**2
    Kcd = np.sqrt(2) * np.pi**1.25 / (alphac + alphad)\
          * np.exp(-alphac * alphad / (alphac + alphad) * rcd2)
    rpq2 = (px-qx)**2 + (py-qy)**2 + (pz-qz)**2
    boys_arg = zeta * eta / (zeta + eta) * rpq2
    mtot = la + ma + na + lc + mc + nc + M

    boys_indices = np.arange(mtot + 1)
    boys_arg = np.repeat(boys_arg,mtot + 1)
    boys_vals = boys(boys_indices, boys_arg)

    vrr_terms = np.zeros((la+1,ma+1,na+1,lc+1,mc+1,nc+1,mtot+1)) 
    for im in range(mtot+1):
        vrr_terms[0,0,0,0,0,0,im] = (
            Kab*Kcd/np.sqrt(zeta+eta) * boys_vals[im]
            )

    for i in range(la):
        for im in range(mtot-i):


            vrr_terms[i+1,0,0, 0,0,0, im] = (
                (px-xa)*vrr_terms[i,0,0, 0,0,0, im]
                + (wx-px)*vrr_terms[i,0,0, 0,0,0, im+1]
                )

            if i:
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

    print(vrr_terms)
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
    #print(vrr_terms.shape)
    return vrr_terms[la,ma,na,lc,mc,nc,M]


xyza = np.array([0.0,0.1,0.9])
xyzb = np.array([0.0,-0.1,-0.9])
xyzc = np.array([0.0,-0.1, 0.9])
xyzd = np.array([0.0,-0.1,-0.9])
norma = normb = normc = normd = 1.0
lmna = (6,0,0)
lmnb = (0,0,0)
lmnc = (6,0,0)
lmnd = (0,0,0)
alphaa,alphab,alphac,alphad = 0.5, 0.4, 0.3, 0.2
M = 0

result = vrr(xyza,lmna,alphaa, xyzb,alphab, xyzc,lmnc,alphac, xyzd,alphad,M)
print(result)

