"""
All one-electron integrals over p functions
(p|s) (s|p) (p|p)
"""
import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from oei_s import overlap_ss, kinetic_ss, potential_ss
np.set_printoptions(linewidth=500)

def overlap_ps(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (px|s) (py|s) (pz|s)
    """
    oot_alpha_bra = 1 / (2 * alpha_bra)
    first_term = np.asarray(jax.jacrev(overlap_ss, (0,1,2))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
    return oot_alpha_bra * first_term

def overlap_pp(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (px|s) (py|s) (pz|s)
    """
    oot_alpha_ket = 1 / (2 * alpha_ket)
    first_term = np.asarray(jax.jacfwd(overlap_ps, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))   
    return oot_alpha_ket * first_term

def kinetic_ps(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (px|s) (py|s) (pz|s)
    """
    oot_alpha_bra = 1 / (2 * alpha_bra)
    first_term = np.asarray(jax.jacrev(kinetic_ss, (0,1,2))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
    return oot_alpha_bra * first_term

def kinetic_pp(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (px|s) (py|s) (pz|s)
    """
    oot_alpha_ket = 1 / (2 * alpha_ket)
    first_term = np.asarray(jax.jacfwd(kinetic_ps, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
    return oot_alpha_ket * first_term

def potential_ps(Ax, Ay, Az, Cx, Cy, Cz, geom, charge, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (px|s) (py|s) (pz|s)
    """
    oot_alpha_bra = 1 / (2 * alpha_bra)
    first_term = np.asarray(jax.jacrev(potential_ss, (0,1,2))(Ax, Ay, Az, Cx, Cy, Cz, geom, charge, alpha_bra, alpha_ket, c1, c2))
    return oot_alpha_bra * first_term

def potential_pp(Ax, Ay, Az, Cx, Cy, Cz, geom, charge, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (px|s) (py|s) (pz|s)
    """
    oot_alpha_ket = 1 / (2 * alpha_ket)
    first_term = np.asarray(jax.jacfwd(potential_ps, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, geom, charge, alpha_bra, alpha_ket, c1, c2))
    return oot_alpha_ket * first_term





