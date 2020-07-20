import jax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)

from oei import *

Ax = 0.0
Ay = 0.0
Az = -0.849220457955
Cx = 0.0
Cy = 0.0
Cz = 0.849220457955
alpha_bra = 0.5 
alpha_ket = 0.5 

# normalizations constants times coefficients (I set coeffs to 1.0 in psi4 inputs)
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
print("(s|s)")
print(overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))

c1 = c_P
c2 = c_S
print("(p|s)")
print(overlap_ps(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))

c1 = c_P
c2 = c_P
print("(p|p)")
print(overlap_pp(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
print(new_overlap_pp(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))

c1 = c_D
c2 = c_S
print("(d|s)")
print(overlap_ds(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))

#c1 = c_D
#c2 = c_P
#print("(d|p)")
#print(overlap_dp(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
#
#c1 = c_D
#c2 = c_D
#print("(d|d)")
#print(overlap_dd(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
#
#c1 = c_F
#c2 = c_S
#print("(f|s)")
#print(overlap_fs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
#
#c1 = c_F
#c2 = c_P
#print("(f|p)")
#print(overlap_fp(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
#
#c1 = c_F
#c2 = c_D
#print("(f|d)")
#print(overlap_fd(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
#
#c1 = c_F
#c2 = c_F
#print("(f|f)")
#print(overlap_ff(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
#
#
