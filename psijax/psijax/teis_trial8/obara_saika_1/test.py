import jax.numpy as np 
#from kmurph_full import Gxyz
from control import Gxyz
from obara_saika_v1 import primitive_eri
from integrals_utils import create_primitive


lA,mA,nA = 1,0,0
lB,mB,nB = 0,0,0
lC,mC,nC = 0,0,0
lD,mD,nD = 0,0,0
a,b,c,d = 0.5, 0.5, 0.5, 0.5
RA,RB,RC,RD = np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9]), np.array([0.0,0.0,0.9]), np.array([0.0,0.0,-0.9])

g1 = create_primitive(lA,mA,nA,RA, a) 
g2 = create_primitive(lB,mB,nB,RB, b) 
g3 = create_primitive(lC,mC,nC,RC, c) 
g4 = create_primitive(lD,mD,nD,RD, d) 

os_result = primitive_eri(g1, g2, g3, g4)
print(os_result)

kmurph_result = Gxyz(lA,mA,nA,lB,mB,nB,lC,mC,nC,lD,mD,nD,a,b,c,d,RA,RB,RC,RD)
print(kmurph_result)

#print(np.allclose(os_result, kmurph_result))
