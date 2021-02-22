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


@jax.jit
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

    #mtot = la + ma + na + lc + mc + nc + M
    mtot = 8

    boys_indices = np.arange(mtot + 1)
    boys_arg = np.repeat(boys_arg,mtot + 1)
    boys_vals = boys(boys_indices, boys_arg)


    with loops.Scope() as S:
        S.vrr_terms = np.zeros((3,3,3,3,3,3,mtot+1))

        S.im = 0
        for _ in S.while_range(lambda: S.im < mtot + 1):
            tmp = Kab * Kcd / np.sqrt(zeta + eta) * boys_vals[S.im]
            S.vrr_terms = jax.ops.index_update(S.vrr_terms, jax.ops.index[0,0,0,0,0,0,S.im], tmp)
            jax.ops.index_update(S.vrr_terms, jax.ops.index[0,0,0,0,0,0,S.im], tmp)
            S.im += 1
    
        S.i = 0
        for _ in S.while_range(lambda: S.i < la):
            S.im = 0
            for _ in S.while_range(lambda: S.im < mtot - S.i):
                tmp = (px - xa) * S.vrr_terms[S.i,0,0, 0,0,0, S.im] + (wx - px) * S.vrr_terms[S.i,0,0, 0,0,0, S.im+1]
                #S.vrr_terms = jax.ops.index_update(S.vrr_terms, jax.ops.index[S.i+1,0,0,0,0,0,S.im], tmp)
                tmp = S.i/2./zeta * (S.vrr_terms[S.i-1,0,0, 0,0,0, S.im] - eta / (zeta + eta) * S.vrr_terms[S.i-1,0,0, 0,0,0, S.im+1])
                #S.vrr_terms = jax.ops.index_add(S.vrr_terms, jax.ops.index[S.i+1,0,0, 0,0,0, S.im], tmp)
                S.im += 1
            S.i += 1

        S.j = 0
        for _ in S.while_range(lambda: S.j < ma):
            S.i = 0
            for _ in S.while_range(lambda: S.i < (la + 1)):
                S.im = 0 
                for _ in S.while_range(lambda: S.im < (mtot - S.i - S.j)):
                    tmp =  (py - ya) * S.vrr_terms[S.i,S.j,0, 0,0,0, S.im] + (wy - py) * S.vrr_terms[S.i,S.j,0, 0,0,0, S.im+1]
                    #S.vrr_terms = jax.ops.index_update(S.vrr_terms, jax.ops.index[S.i,S.j+1,0,0,0,0,S.im], tmp)
                    tmp = S.j / 2. / zeta*(S.vrr_terms[S.i,S.j-1,0, 0,0,0, S.im] - eta / (zeta+ eta) * S.vrr_terms[S.i,S.j-1,0, 0,0,0, S.im+1])
                    #S.vrr_terms = jax.ops.index_add(S.vrr_terms, jax.ops.index[S.i,S.j+1,0, 0,0,0, S.im], tmp)
                    S.im += 1 
                S.i += 1 
            S.j += 1 


        S.k = 0
        for _ in S.while_range(lambda: S.k < na):
            S.j = 0
            for _ in S.while_range(lambda: S.j < (ma + 1)):
                S.i = 0
                for _ in S.while_range(lambda: S.i < (la + 1)):
                    S.im = 0 
                    for _ in S.while_range(lambda: S.im < (mtot - S.i - S.j - S.k)):
                        tmp = (pz - za) * S.vrr_terms[S.i,S.j,S.k, 0,0,0, S.im] + (wz - pz) * S.vrr_terms[S.i,S.j,S.k, 0,0,0, S.im+1]
                        #S.vrr_terms = jax.ops.index_update(S.vrr_terms, jax.ops.index[S.i,S.j,S.k+1,0,0,0,S.im], tmp)
                        tmp = S.k /2. / zeta * (S.vrr_terms[S.i,S.j,S.k-1, 0,0,0, S.im] - eta / (zeta + eta) * S.vrr_terms[S.i,S.j,S.k-1, 0,0,0, S.im+1])
                        #S.vrr_terms = jax.ops.index_add(S.vrr_terms, jax.ops.index[S.i,S.j,S.k+1,0,0,0,S.im], tmp)
                        S.im += 1
                    S.i += 1
                S.j += 1
            S.k += 1



        S.q = 0        
        for _ in S.while_range(lambda: S.q < lc):
            S.k = 0
            for _ in S.while_range(lambda: S.k < (na + 1)):
                S.j = 0
                for _ in S.while_range(lambda: S.j < (ma + 1)):
                    S.i = 0
                    for _ in S.while_range(lambda: S.i < (la + 1)):
                        S.im = 0 
                        for _ in S.while_range(lambda: S.im < (mtot - S.i - S.j - S.k - S.q)):
                            tmp = (qx - xc) * S.vrr_terms[S.i,S.j,S.k, S.q,0,0, S.im] + (wx - qx) * S.vrr_terms[S.i,S.j,S.k, S.q,0,0, S.im+1]
                            #S.vrr_terms = jax.ops.index_update(S.vrr_terms, jax.ops.index[S.i,S.j,S.k,S.q+1,0,0,S.im], tmp)
                            tmp = S.q / 2. / eta * (S.vrr_terms[S.i,S.j,S.k, S.q-1,0,0, S.im] - zeta / (zeta + eta) * S.vrr_terms[S.i,S.j,S.k, S.q-1,0,0, S.im+1])
                            #S.vrr_terms = jax.ops.index_add(S.vrr_terms, jax.ops.index[S.i,S.j,S.k,S.q+1,0,0,S.im], tmp)
                            tmp = S.i / 2. / (zeta + eta) * S.vrr_terms[S.i-1,S.j,S.k, S.q,0,0, S.im+1]
                            #S.vrr_terms = jax.ops.index_add(S.vrr_terms, jax.ops.index[S.i,S.j,S.k,S.q+1,0,0,S.im], tmp)
                            S.im += 1
                        S.i += 1
                    S.j += 1
                S.k += 1
            S.q += 1



        S.r = 0
        for _ in S.while_range(lambda: S.r < mc):
            S.q = 0        
            for _ in S.while_range(lambda: S.q < (lc + 1)):
                S.k = 0
                for _ in S.while_range(lambda: S.k < (na + 1)):
                    S.j = 0
                    for _ in S.while_range(lambda: S.j < (ma + 1)):
                        S.i = 0
                        for _ in S.while_range(lambda: S.i < (la + 1)):
                            S.im = 0 
                            for _ in S.while_range(lambda: S.im < (mtot - S.i - S.j - S.k - S.q - S.r)):
                                tmp = (qy - yc) * S.vrr_terms[S.i,S.j,S.k, S.q,S.r,0, S.im] + (wy - qy) * S.vrr_terms[S.i,S.j,S.k, S.q,S.r,0, S.im+1]
                                #S.vrr_terms = jax.ops.index_update(S.vrr_terms, jax.ops.index[S.i,S.j,S.k,S.q,S.r+1,0,S.im], tmp)
                                tmp = S.r / 2. / eta * (S.vrr_terms[S.i,S.j,S.k, S.q,S.r-1,0, S.im] - zeta / (zeta + eta) * S.vrr_terms[S.i,S.j,S.k, S.q,S.r-1,0, S.im+1])
                                #S.vrr_terms = jax.ops.index_add(S.vrr_terms, jax.ops.index[S.i,S.j,S.k,S.q,S.r+1,0,S.im], tmp)
                                tmp = S.j /2. / (zeta + eta) * S.vrr_terms[S.i,S.j-1,S.k,S.q,S.r,0,S.im+1]
                                #S.vrr_terms = jax.ops.index_add(S.vrr_terms, jax.ops.index[S.i,S.j,S.k,S.q,S.r+1,0,S.im], tmp)
                                S.im += 1
                            S.i += 1
                        S.j += 1
                    S.k += 1
                S.q += 1
            S.r += 1



        S.s = 0
        for _ in S.while_range(lambda: S.s < nc):
            S.r = 0
            for _ in S.while_range(lambda: S.r < (mc + 1)):
                S.q = 0        
                for _ in S.while_range(lambda: S.q < (lc + 1)):
                    S.k = 0
                    for _ in S.while_range(lambda: S.k < (na + 1)):
                        S.j = 0
                        for _ in S.while_range(lambda: S.j < (ma + 1)):
                            S.i = 0
                            for _ in S.while_range(lambda: S.i < (la + 1)):
                                S.im = 0 
                                for _ in S.while_range(lambda: S.im < (mtot - S.i - S.j - S.k - S.q - S.r - S.s)):
                                    tmp = (qz - zc) * S.vrr_terms[S.i,S.j,S.k,S.q,S.r,S.s,S.im] + (wz - qz) * S.vrr_terms[S.i,S.j,S.k,S.q,S.r,S.s,S.im+1]
                                    #S.vrr_terms = jax.ops.index_update(S.vrr_terms, jax.ops.index[S.i,S.j,S.k,S.q,S.r,S.s+1,S.im], tmp)
                                    tmp = S.s/ 2. / eta * (S.vrr_terms[S.i,S.j,S.k,S.q,S.r,S.s-1,S.im]- zeta / (zeta + eta) * S.vrr_terms[S.i,S.j,S.k,S.q,S.r,S.s-1,S.im+1])
                                    #S.vrr_terms = jax.ops.index_add(S.vrr_terms, jax.ops.index[S.i,S.j,S.k,S.q,S.r,S.s+1,S.im], tmp)
                                    tmp = S.k / 2. / (zeta + eta) * S.vrr_terms[S.i,S.j,S.k-1,S.q,S.r,S.s,S.im+1]
                                    #S.vrr_terms = jax.ops.index_add(S.vrr_terms, jax.ops.index[S.i,S.j,S.k,S.q,S.r,S.s+1,S.im], tmp)
                                    S.im += 1
                                S.i += 1
                            S.j += 1
                        S.k += 1
                    S.q += 1
                S.r += 1
            S.s += 1

    print(S.vrr_terms)
    return S.vrr_terms[la,ma,na,lc,mc,nc,M]


xyza = np.array([0.0,0.1,0.9])
xyzb = np.array([0.0,-0.1,-0.9])
xyzc = np.array([0.0,-0.1, 0.9])
xyzd = np.array([0.0,-0.1,-0.9])
norma = normb = normc = normd = 1.0
lmna = (1,2,1)
lmnb = (0,0,0)
lmnc = (1,0,2)
lmnd = (0,0,0)
alphaa,alphab,alphac,alphad = 0.5, 0.4, 0.3, 0.2
M = 0

result = vrr(xyza,lmna,alphaa, xyzb,alphab, xyzc,lmnc,alphac, xyzd,alphad,M)
print(result)

#for i in range(10000):
#    result = vrr(xyza,lmna,alphaa, xyzb,alphab, xyzc,lmnc,alphac, xyzd,alphad,M)


#vrr_vmap = jax.vmap(vrr, (0,0,0,0,0,0,0,0,0,0,0),0)
#
#xyza = np.tile(xyza,(10000,1))
#xyzb = np.tile(xyzb,(10000,1))
#xyzc = np.tile(xyzc,(10000,1))
#xyzd = np.tile(xyzd,(10000,1))
#print(xyza.shape)
#lmna = np.tile(np.array([2,0,0]),(10000,1))
#lmnb = np.tile(np.array([0,0,0]),(10000,1))
#lmnc = np.tile(np.array([0,2,0]),(10000,1))
#lmnd = np.tile(np.array([0,0,0]),(10000,1))
#print(lmnd.shape)
#
#alphaa = np.repeat(0.5,10000)
#alphab = np.repeat(0.4,10000)
#alphac = np.repeat(0.3,10000)
#alphad = np.repeat(0.2,10000)
#M = np.repeat(0, 10000)
#
#results = vrr_vmap(xyza,lmna,alphaa, xyzb,alphab, xyzc,lmnc,alphac, xyzd,alphad,M)



