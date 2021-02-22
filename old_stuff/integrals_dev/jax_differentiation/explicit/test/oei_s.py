"""
All one-electron integrals over s functions (s|s)
These are used to construct higher angular momentum integral functions using
(a + 1i | b) = 1/2alpha * (d/dAi (a|b) + ai (a - 1i | b))
(a | b + 1i) = 1/2beta  * (d/dBi (a|b) + bi (a | b - 1i))
where i is a cartesian component of the gaussian
"""
import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)

def overlap_ss(A,C, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns a (s|s) overlap integral
    """
    ss = ((np.pi / (alpha_bra + alpha_ket))**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(A-C, A-C)) / (alpha_bra + alpha_ket)))
    return ss * c1 * c2

X = np.array([1.0,1.0,1.0])
Y = np.array([1.0,1.0,1.0])
args = (X,Y,0.5,0.5,0.5,0.5)
overlap_ps = jax.jacfwd(overlap_ss, 0)
overlap_pp = jax.jacfwd(overlap_ps, 1)
overlap_ds = jax.jacfwd(overlap_ps, 0)
overlap_dp = jax.jacfwd(overlap_ds, 1)
overlap_dd = jax.jacfwd(overlap_dp, 1)

print('ps',overlap_ps(*args).shape)
print('pp',overlap_pp(*args).shape)
print('ds',overlap_ds(*args).shape)
print('dp',overlap_dp(*args).shape)
print('dd',overlap_dd(*args).shape)

dx, dy, dz = np.split(overlap_ds(*args), 3, axis=0)
print(dx)


#def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
#    """
#    Computes and returns a (s|s) overlap integral
#    """
#    A = np.array([Ax, Ay, Az])
#    C = np.array([Cx, Cy, Cz])
#    alpha_sum = np.add(alpha_bra,alpha_ket)
#    AC = np.subtract(A,C)
#    ss = ((np.pi / (alpha_sum))**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(AC, AC)) * np.reciprocal(alpha_sum)))
#    return ss * c1 * c2



# Using this adds a massive memory/time overhead. Takeaway: always use lax primitives instead of python ops wherever possible!
#def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
#    """
#    Computes and returns a (s|s) overlap integral
#    """
#    ss = ((np.pi / (alpha_bra + alpha_ket))**(3/2) * np.exp((-alpha_bra * alpha_ket * ((Ax - Cx)**2 + (Ay - Cy)**2 + (Az - Cz)**2)) / (alpha_bra + alpha_ket)))
#    return ss * c1 * c2


