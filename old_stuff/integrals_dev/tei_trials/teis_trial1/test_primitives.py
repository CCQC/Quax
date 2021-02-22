import jax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)

from eri import eri_ssss 

A = np.array([0.0,0.0,-0.849220457955])
B = A
C = np.array([0.0,0.0,0.849220457955])
D = C

alpha = 0.5
beta = 0.5
gamma = 0.4
delta = 0.4

# normalizations constants times coefficients (I set coeffs to 1.0 in psi4 inputs)
c_S = 0.4237772081237576
c_other = 0.35847187357690596
c_P = 0.5993114751532237
c_D = 0.489335770373359
c_F = 0.3094831149945914
c_G = 0.1654256833287603
c_H = 0.07798241497612321
c_I = 0.0332518134720999

# Change coefficients depending on which function youre testing
c1 = c_S
c2 = c_S
c3 = c_other
c4 = c_other
print("(s|s)")
print(eri_ssss(A,B,C,D, alpha, beta, gamma, delta, c1, c2, c3, c4))

