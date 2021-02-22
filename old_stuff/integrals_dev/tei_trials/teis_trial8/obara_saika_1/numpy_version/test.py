#import jax
#from jax.config import config; config.update("jax_enable_x64", True)
#import jax.numpy as np 
import numpy as np

#from reference_os import PrimitiveBasis, Basis, ObaraSaika
#from control import Gxyz
#from obara_saika_v1 import os_begin
#from obara_saika_v2 import primitive_eri
from obara_saika_v3 import primitive_eri
from integrals_utils import create_primitive

lA,mA,nA = 1,0,0
lB,mB,nB = 0,1,0
lC,mC,nC = 0,0,1
lD,mD,nD = 1,0,0
a,b,c,d = 0.5, 0.4, 0.3, 0.2
#RA,RB,RC,RD = np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9]), np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9])
RA,RB,RC,RD = np.array([0.0,0.1,0.9]), np.array([0.0,-0.1,-0.9]), np.array([0.0,-0.1,0.9]), np.array([0.0,0.1,-0.9])

g1 = create_primitive(lA,mA,nA, RA, a) 
g2 = create_primitive(lB,mB,nB, RB, b) 
g3 = create_primitive(lC,mC,nC, RC, c) 
g4 = create_primitive(lD,mD,nD, RD, d) 

os_result = primitive_eri(g1,g2,g3,g4)
print(os_result)
#
#basis1 = Basis([PrimitiveBasis(1.0, a,  tuple(RA), (lA,mA,nA))],  tuple(RA), (lA,mA,nA))
#basis2 = Basis([PrimitiveBasis(1.0, b,  tuple(RB), (lB,mB,nB))],  tuple(RB), (lB,mB,nB))
#basis3 = Basis([PrimitiveBasis(1.0, c,  tuple(RC), (lC,mC,nC))],  tuple(RC), (lC,mC,nC))
#basis4 = Basis([PrimitiveBasis(1.0, d,  tuple(RD), (lD,mD,nD))],  tuple(RD), (lD,mD,nD))
#
#obj = ObaraSaika()
#result = obj.integrate(basis1, basis2, basis3, basis4)
#print(result)

#kmurph_result = Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD)
#print(kmurph_result)


