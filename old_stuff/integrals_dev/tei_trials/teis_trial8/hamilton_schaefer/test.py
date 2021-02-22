from reference import vrr as ref
from trial1 import vrr as trial
import numpy as np

xyza = np.array([0.0,0.1,0.9])
xyzb = np.array([0.0,-0.1,-0.9])
xyzc = np.array([0.0,-0.1, 0.9])
xyzd = np.array([0.0,-0.1,-0.9])
norma = normb = normc = normd = 1.0
lmna = (0,1,1)
lmnb = (0,0,0)
lmnc = (0,0,0)
lmnd = (0,0,0)
alphaa,alphab,alphac,alphad = 0.5, 0.4, 0.3, 0.2
M = 0

xa,ya,za = 0.0,0.1,0.9
xb,yb,zb = 0.0,-0.1,-0.9
xc,yc,zc = 0.0,-0.1, 0.9
xd,yd,zd = 0.0,-0.1,-0.9
la,ma,na = 0,0,1

am = [(0,0,0), (0,0,1), (0,1,1), (1,1,1), (1,0,1), (1,0,0), (0,1,0), (2,0,0), (2,1,0) ,(2,1,1), (2,2,2), (3,3,3), (4,4,4)]

for a in am:
    la, ma, na = a
    lmna = (la,ma,na)
    refresult = ref(xyza,norma,lmna,alphaa, xyzb,normb,alphab, xyzc,normc,lmnc,alphac, xyzd,normd,alphad,M)
    result = trial((la,ma,na,xa,ya,za,xb,yb,zb,xc,yc,zc,xd,yd,zd, alphaa,alphab,alphac,alphad))
    print(refresult)
    print(result)
