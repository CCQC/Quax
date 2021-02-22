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
np.set_printoptions(linewidth=500)
from integrals_utils import boys, gaussian_product

#@jax.jit
def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns a (s|s) overlap integral
    """
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    ss = ((np.pi / (alpha_bra + alpha_ket))**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(A-C, A-C)) / (alpha_bra + alpha_ket)))
    return ss * c1 * c2


#@jax.jit
#def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
#    """
#    Computes and returns a (s|s) overlap integral
#    """
#    ss = ((np.pi / (alpha_bra + alpha_ket))**(3/2) * np.exp((-alpha_bra * alpha_ket * ((Ax - Cx)**2 + (Ay - Cy)**2 + (Az - Cz)**2)) / (alpha_bra + alpha_ket)))
#    return ss * c1 * c2


@jax.jit
def kinetic_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns a (s|s) kinetic integral
    """
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    P = (alpha_bra * alpha_ket) / (alpha_bra + alpha_ket)
    ab = -1.0 * np.dot(A-C, A-C)
    K = overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2) * (3 * P + 2 * P * P * ab)
    return K

@jax.jit
def potential_ss(Ax, Ay, Az, Cx, Cy, Cz, geom, charge, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns a (s|s) potential integral
    """
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    g = alpha_bra + alpha_ket
    P, c = gaussian_product(alpha_bra,alpha_ket,A,C)
    V = 0
    # For every atom
    for i in range(geom.shape[0]):
        arg = g * np.dot(P - geom[i], P - geom[i])
        F = boys(arg)
        V += -charge[i] * F * c * 2 * np.pi / g
    return V * c1 * c2


