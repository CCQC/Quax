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

def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns a (s|s) overlap integral
    """
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    #ss = ((np.pi / (alpha_bra + alpha_ket))**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(A-C, A-C)) / (alpha_bra + alpha_ket)))
    #return ss * c1 * c2
    alpha_sum = alpha_bra + alpha_ket
    return c1 * c2 * (np.pi / alpha_sum)**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(A-C, A-C)) / alpha_sum)

    #AC = np.array([Ax-Cx, Ay-Cy, Az-Cz])
    #alpha_sum = alpha_bra + alpha_ket
    #return c1 * c2 * (np.pi / alpha_sum)**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(AC, AC)) / alpha_sum)


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

