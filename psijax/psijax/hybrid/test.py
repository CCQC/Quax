import jax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)

from oei_s import *
from oei_p import *
from oei_d import *
from oei_f import *

Ax = 0.0
Ay = 0.0
Az = -0.849220457955
Cx = 0.0
Cy = 0.0
Cz = 0.849220457955
alpha_bra = 0.5 
alpha_ket = 0.5 

c_S = 0.4237772081237576
c_P = 0.5993114751532237
c_D = 0.489335770373359
c_F = 0.3094831149945914
c_G = 0.1654256833287603
c_H = 0.07798241497612321
c_I = 0.0332518134720999

# Change coefficients depending on which function youre testing
c1 = c_S
c2 = c_S
print(overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))

c1 = c_D
c2 = c_D
print(overlap_dd(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))

c1 = c_F
c2 = c_D
print(overlap_fd(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))

c1 = c_F
c2 = c_F
print(overlap_ff(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))

c1 = c_F
c2 = c_S
print(overlap_fs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))


