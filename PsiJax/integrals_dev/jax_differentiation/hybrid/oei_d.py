"""
All one-electron integrals over d functions
(d|s) (d|p) (d|d)
Uses the following equations for promoting angular momentum:
(a + 1i | b) = 1/2alpha * (d/dAi (a|b) + ai (a - 1i | b))
(a | b + 1i) = 1/2beta  * (d/dBi (a|b) + bi (a | b - 1i))
where i is a cartesian component of the gaussian x,y,z
"""
import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from integrals_utils import lower_take_mask, boys
from oei_s import * 
from oei_p import * 

def overlap_ds(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (dxx|s) (dxy|s) (dxz|s) (dyy|s) (dyz|s) (dzz|s) 
    """
    oot_alpha_bra = 1 / (2 * alpha_bra)
    #dx_first_term, dy_first_term, dz_first_term = jax.jacfwd(overlap_ps, (0,1,2))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    dx_first_term, dy_first_term, dz_first_term = jax.jacrev(overlap_ps, (0,1,2))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    #dx_first_term, dy_first_term, dz_first_term = jacfwd_overlap_ps(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    lower = overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)

    dx_second_term = np.array([1,0,0]) * lower
    dx = (dx_first_term + dx_second_term)

    dy_second_term = np.array([0,1,0]) * lower
    dy = (dy_first_term + dy_second_term)

    dz_second_term = np.array([0,0,1]) * lower
    dz = (dz_first_term + dz_second_term)
    return  oot_alpha_bra * np.hstack((dx, dy[1:], dz[-1]))

def overlap_dp(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a matrix: 
    (dxx|px) (dxy|px) (dxz|px) (dyy|px) (dyz|px) (dzz|px) 
    (dxx|py) (dxy|py) (dxz|py) (dyy|py) (dyz|py) (dzz|py) 
    (dxx|pz) (dxy|pz) (dxz|pz) (dyy|pz) (dyz|pz) (dzz|pz) 
    """
    oot_alpha_ket = 1 / (2 * alpha_ket)
    #first_term = np.asarray(jax.jacfwd(overlap_ds, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
    first_term = np.asarray(jax.jacrev(overlap_ds, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
    #first_term = np.asarray(jacfwd_overlap_ds(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
    return oot_alpha_ket * first_term

def overlap_dd(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    # NOTE this function returns redundant integrals NOTE #
    """
    Computes and returns the following integrals as a matrix: 
    (dxx|dxx) (dxy|dxx) (dxz|dxx) (dyy|dxx) (dyz|dxx) (dzz|dxx) 
    (dxx|dxy) (dxy|dxy) (dxz|dxy) (dyy|dxy) (dyz|dxy) (dzz|dxy) 
    (dxx|dxz) (dxy|dxz) (dxz|dxz) (dyy|dxz) (dyz|dxz) (dzz|dxz) 
    (dxx|dyy) (dxy|dyy) (dxz|dyy) (dyy|dyy) (dyz|dyy) (dzz|dyy) 
    (dxx|dyz) (dxy|dyz) (dxz|dyz) (dyy|dyz) (dyz|dyz) (dzz|dyz) 
    (dxx|dzz) (dxy|dzz) (dxz|dzz) (dyy|dzz) (dyz|dzz) (dzz|dzz) 
    """
    oot_alpha_ket = 1 / (2 * alpha_ket)
    lower = overlap_ds(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    lower_padded = np.pad(lower.reshape(-1), (1,0))
    # Compute first term of integral derivative equation.
    #dx_1, dy_1, dz_1 = jax.jacfwd(overlap_dp, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    dx_1, dy_1, dz_1 = jax.jacrev(overlap_dp, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    #dx_1, dy_1, dz_1 = jacfwd_overlap_dp(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)

    # coefficient array for second term. mimics number of 'x's in ket in the (d|p) function's returned matrix
    bx = np.tile(np.array([1,0,0]), 6).reshape(6,3).T
    # mimics structure of previous matrix, where nonzero values are replaced with number in numerical order 1,2,3
    take_x = lower_take_mask(bx)

    # coefficient array for second term. mimics number of 'y's in ket in the (d|p) function's returned matrix
    by = np.tile(np.array([0,1,0]), 6).reshape(6,3).T
    # mimics structure of previous matrix, where nonzero values are replaced with number in numerical order 1,2,3
    take_y = lower_take_mask(by)

    # coefficient array for second term. mimics number of 'z's in ket in the (d|p) function's returned matrix
    bz = np.tile(np.array([0,0,1]), 6).reshape(6,3).T
    # mimics structure of previous matrix, where nonzero values are replaced with number in numerical order 1,2,3
    take_z = lower_take_mask(bz)

    # Match proper primitive integral with correct coefficient for creating second term
    dx_2 = bx * np.take(lower_padded, take_x)
    dy_2 = by * np.take(lower_padded, take_y)
    dz_2 = bz * np.take(lower_padded, take_z)

    dx = oot_alpha_ket * (dx_1 + dx_2)
    dy = oot_alpha_ket * (dy_1 + dy_2)
    dz = oot_alpha_ket * (dz_1 + dz_2)
    
    # slice of 'y' part is equal to the promoted angular momentum component of the result. d=2, f=3, g=4 etc..
    return np.vstack((dx, dy[-2:], dz[-1]))


