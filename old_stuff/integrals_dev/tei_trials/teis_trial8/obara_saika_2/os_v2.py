import jax
from jax.experimental import loops
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np

#import numpy as np 
#from scipy import special

def boys(m,x):
    #return 0.5 * (x + 1e-11)**(-(m + 0.5)) * jax.lax.igamma(m + 0.5, x + 1e-11) * np.exp(jax.lax.lgamma(m + 0.5))
    return 0.5 * (x + 1e-11)**(-(m + 0.5)) * jax.scipy.special.gammainc(m + 0.5, x + 1e-11) * np.exp(jax.lax.lgamma(m + 0.5))

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
        tmp = Kab*Kcd/np.sqrt(zeta+eta) * boys_vals[im]
        vrr_terms = jax.ops.index_update(vrr_terms, jax.ops.index[0,0,0,0,0,0,im], tmp)

    for i in range(la):
        for im in range(mtot-i):
            tmp = (px-xa) * vrr_terms[i,0,0, 0,0,0, im] + (wx - px) * vrr_terms[i,0,0, 0,0,0, im+1]
            vrr_terms = jax.ops.index_update(vrr_terms, jax.ops.index[i+1,0,0,0,0,0,im], tmp)

            if i:
                tmp = i/2./zeta* (vrr_terms[i-1,0,0, 0,0,0, im] - eta/(zeta+eta)*vrr_terms[i-1,0,0, 0,0,0, im+1])
                vrr_terms = jax.ops.index_add(vrr_terms, jax.ops.index[i+1,0,0, 0,0,0, im], tmp)

    for j in range(ma):
        for i in range(la+1):
            for im in range(mtot-i-j):
                tmp =  (py - ya) * vrr_terms[i,j,0, 0,0,0, im] + (wy - py) * vrr_terms[i,j,0, 0,0,0, im+1]
                vrr_terms = jax.ops.index_update(vrr_terms, jax.ops.index[i,j+1,0,0,0,0,im], tmp)
                if j:
                    tmp = j / 2. / zeta*(vrr_terms[i,j-1,0, 0,0,0, im] - eta / (zeta+ eta) * vrr_terms[i,j-1,0, 0,0,0, im+1])
                    vrr_terms = jax.ops.index_add(vrr_terms, jax.ops.index[i,j+1,0, 0,0,0, im], tmp)


    for k in range(na):
        for j in range(ma+1):
            for i in range(la+1):
                for im in range(mtot-i-j-k):
                    tmp = (pz-za)*vrr_terms[i,j,k, 0,0,0, im] + (wz-pz)*vrr_terms[i,j,k, 0,0,0, im+1]
                    vrr_terms = jax.ops.index_update(vrr_terms, jax.ops.index[i,j,k+1,0,0,0,im], tmp)
                    if k:
                        tmp = k/2./zeta*(vrr_terms[i,j,k-1, 0,0,0, im] - eta/(zeta+eta) *vrr_terms[i,j,k-1, 0,0,0, im+1])
                        vrr_terms = jax.ops.index_add(vrr_terms, jax.ops.index[i,j,k+1,0,0,0,im], tmp)

    for q in range(lc):
        for k in range(na+1):
            for j in range(ma+1):
                for i in range(la+1):
                    for im in range(mtot-i-j-k-q):
                        tmp = (qx-xc)*vrr_terms[i,j,k, q,0,0, im] + (wx-qx)*vrr_terms[i,j,k, q,0,0, im+1]
                        vrr_terms = jax.ops.index_update(vrr_terms, jax.ops.index[i,j,k,q+1,0,0,im], tmp)
                        if q:
                            tmp = q/2./eta*(vrr_terms[i,j,k, q-1,0,0, im] - zeta/(zeta+eta)*vrr_terms[i,j,k, q-1,0,0, im+1])
                            vrr_terms = jax.ops.index_add(vrr_terms, jax.ops.index[i,j,k,q+1,0,0,im], tmp)
                        if i:
                            tmp = i/2./(zeta+eta)*vrr_terms[i-1,j,k, q,0,0, im+1]
                            vrr_terms = jax.ops.index_add(vrr_terms, jax.ops.index[i,j,k,q+1,0,0,im], tmp)

    for r in range(mc):
        for q in range(lc+1):
            for k in range(na+1):
                for j in range(ma+1):
                    for i in range(la+1):
                        for im in range(mtot-i-j-k-q-r):
                            tmp = (qy-yc)*vrr_terms[i,j,k, q,r,0, im] + (wy-qy)*vrr_terms[i,j,k, q,r,0, im+1]
                            vrr_terms = jax.ops.index_update(vrr_terms, jax.ops.index[i,j,k,q,r+1,0,im], tmp)
                            if r:
                                tmp = r/2./eta*(vrr_terms[i,j,k, q,r-1,0, im] - zeta/(zeta+eta) * vrr_terms[i,j,k, q,r-1,0, im+1])
                                vrr_terms = jax.ops.index_add(vrr_terms, jax.ops.index[i,j,k,q,r+1,0,im], tmp)
                            if j:
                                tmp = j/2./(zeta+eta)*vrr_terms[i,j-1,k,q,r,0,im+1]
                                vrr_terms = jax.ops.index_add(vrr_terms, jax.ops.index[i,j,k,q,r+1,0,im], tmp)

    for s in range(nc):
        for r in range(mc+1):
            for q in range(lc+1):
                for k in range(na+1):
                    for j in range(ma+1):
                        for i in range(la+1):
                            for im in range(mtot-i-j-k-q-r-s):
                                tmp = (qz-zc)*vrr_terms[i,j,k,q,r,s,im] + (wz-qz)*vrr_terms[i,j,k,q,r,s,im+1]
                                vrr_terms = jax.ops.index_update(vrr_terms, jax.ops.index[i,j,k,q,r,s+1,im], tmp)
                                if s:
                                    tmp = s/2./eta*(vrr_terms[i,j,k,q,r,s-1,im]- zeta/(zeta+eta) *vrr_terms[i,j,k,q,r,s-1,im+1])
                                    vrr_terms = jax.ops.index_add(vrr_terms, jax.ops.index[i,j,k,q,r,s+1,im], tmp)
                
                                if k:
                                    tmp = k/2./(zeta+eta)*vrr_terms[i,j,k-1,q,r,s,im+1]
                                    vrr_terms = jax.ops.index_add(vrr_terms, jax.ops.index[i,j,k,q,r,s+1,im], tmp)

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

