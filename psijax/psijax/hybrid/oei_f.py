"""
All one-electron integrals over f functions
(f|s) (f|p) (f|d) (f|f)
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
from oei_s import overlap_ss, kinetic_ss, potential_ss
from oei_p import * 
from oei_d import * 

@jax.jit
def overlap_fs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (fxxx|s) (fxxy|s) (fxxz|s) (fxyy|s) (fxyz|s) (fxzz|s) (fyyy|s) (fyyz|s) (fyzz|s) (fzzz|s)
    """
    oot_alpha_bra = 1 / (2 * alpha_bra)
    # pad the lower angluar momentum integral (a-1i|b) with 1 zero on each side.
    lower = np.pad(overlap_ps(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2), 1)
    fx_first_term, fy_first_term, fz_first_term = jax.jacfwd(overlap_ds, (0,1,2))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)

    # take either a 0 (index 0) the integral (px|s) (py|s) (pz|s) (indices 1,2,3) to make second term
    fx_second_term = np.array([2,1,1,0,0,0]) * np.take(lower, [1,2,3,0,0,0])
    fx = oot_alpha_bra * (fx_first_term + fx_second_term)

    # take either a 0 (index 0) the integral (px|s) (py|s) (pz|s) (indices 1,2,3) to make second term
    fy_second_term = np.array([0,1,0,2,1,0]) * np.take(lower, [0,1,0,2,3,0])
    fy = oot_alpha_bra * (fy_first_term + fy_second_term)

    # take either a 0 (index 0) the integral (px|s) (py|s) (pz|s) (indices 1,2,3) to make second term
    fz_second_term = np.array([0,0,1,0,1,2]) * np.take(lower, [0,0,1,0,2,3])
    fz = oot_alpha_bra * (fz_first_term + fz_second_term)
    return np.hstack((fx, fy[-3:], fz[-1]))

@jax.jit
def overlap_fp(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a matrix: 
    (fxxx|px) (fxxy|px) (fxxz|px) (fxyy|px) (fxyz|px) (fxzz|px) (fyyy|px) (fyyz|px) (fyzz|px) (fzzz|px)
    (fxxx|py) (fxxy|py) (fxxz|py) (fxyy|py) (fxyz|py) (fxzz|py) (fyyy|py) (fyyz|py) (fyzz|py) (fzzz|py)
    (fxxx|pz) (fxxy|pz) (fxxz|pz) (fxyy|pz) (fxyz|pz) (fxzz|pz) (fyyy|pz) (fyyz|pz) (fyzz|pz) (fzzz|pz)
    """
    oot_alpha_ket = 1 / (2 * alpha_ket)
    lower = overlap_fs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    first_term = np.asarray(jax.jacfwd(overlap_fs, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
    return oot_alpha_ket * first_term

@jax.jit
def overlap_fd(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a matrix: 
    (fxxx|dxx) (fxxy|dxx) (fxxz|dxx) (fxyy|dxx) (fxyz|dxx) (fxzz|dxx) (fyyy|dxx) (fyyz|dxx) (fyzz|dxx) (fzzz|dxx)
    (fxxx|dxy) (fxxy|dxy) (fxxz|dxy) (fxyy|dxy) (fxyz|dxy) (fxzz|dxy) (fyyy|dxy) (fyyz|dxy) (fyzz|dxy) (fzzz|dxy)
    (fxxx|dxz) (fxxy|dxz) (fxxz|dxz) (fxyy|dxz) (fxyz|dxz) (fxzz|dxz) (fyyy|dxz) (fyyz|dxz) (fyzz|dxz) (fzzz|dxz)
    (fxxx|dyy) (fxxy|dyy) (fxxz|dyy) (fxyy|dyy) (fxyz|dyy) (fxzz|dyy) (fyyy|dyy) (fyyz|dyy) (fyzz|dyy) (fzzz|dyy)
    (fxxx|dyz) (fxxy|dyz) (fxxz|dyz) (fxyy|dyz) (fxyz|dyz) (fxzz|dyz) (fyyy|dyz) (fyyz|dyz) (fyzz|dyz) (fzzz|dyz)
    (fxxx|dzz) (fxxy|dzz) (fxxz|dzz) (fxyy|dzz) (fxyz|dzz) (fxzz|dzz) (fyyy|dzz) (fyyz|dzz) (fyzz|dzz) (fzzz|dzz)
    """
    oot_alpha_ket = 1 / (2 * alpha_ket)
    lower = overlap_fs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    lower_padded = np.pad(lower.reshape(-1), (1,0))

    # Compute first term of integral derivative equation.
    dx_1, dy_1, dz_1 = jax.jacfwd(overlap_fp, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)

    # coefficient array for second term. mimics number of 'x's in ket in the (f|p) function's returned matrix
    bx = np.tile(np.array([1,0,0]), 10).reshape(10,3).T
    # mimics structure of previous matrix, where nonzero values are replaced with number in numerical order 1,2,3
    take_x = lower_take_mask(bx)

    # coefficient array for second term. mimics number of 'y's in ket in the (f|p) function's returned matrix
    by = np.tile(np.array([0,1,0]), 10).reshape(10,3).T
    # mimics structure of previous matrix, where nonzero values are replaced with number in numerical order 1,2,3
    take_y = lower_take_mask(by)

    # coefficient array for second term. mimics number of 'z's in ket in the (f|p) function's returned matrix
    bz = np.tile(np.array([0,0,1]), 10).reshape(10,3).T
    # mimics structure of previous matrix, where nonzero values are replaced with number in numerical order 1,2,3
    take_z = lower_take_mask(bz)

    # Match proper primitive integral with correct coefficient for creating second term
    dx_2 = bx * np.take(lower_padded, take_x)
    dy_2 = by * np.take(lower_padded, take_y)
    dz_2 = bz * np.take(lower_padded, take_z)

    dx = oot_alpha_ket * (dx_1 + dx_2)
    dy = oot_alpha_ket * (dy_1 + dy_2)
    dz = oot_alpha_ket * (dz_1 + dz_2)
    return np.vstack((dx, dy[-2:], dz[-1]))

@jax.jit
def overlap_ff(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    # NOTE this returns redundant integrals NOTE
    """
    Computes and returns the following integrals as a matrix: 
    (fxxx|fxxx) (fxxy|fxxx) (fxxz|fxxx) (fxyy|fxxx) (fxyz|fxxx) (fxzz|fxxx) (fyyy|fxxx) (fyyz|fxxx) (fyzz|fxxx) (fzzz|fxxx)
    (fxxx|fxxy) (fxxy|fxxy) (fxxz|fxxy) (fxyy|fxxy) (fxyz|fxxy) (fxzz|fxxy) (fyyy|fxxy) (fyyz|fxxy) (fyzz|fxxy) (fzzz|fxxy)
    (fxxx|fxxz) (fxxy|fxxz) (fxxz|fxxz) (fxyy|fxxz) (fxyz|fxxz) (fxzz|fxxz) (fyyy|fxxz) (fyyz|fxxz) (fyzz|fxxz) (fzzz|fxxz)
    (fxxx|fxyy) (fxxy|fxyy) (fxxz|fxyy) (fxyy|fxyy) (fxyz|fxyy) (fxzz|fxyy) (fyyy|fxyy) (fyyz|fxyy) (fyzz|fxyy) (fzzz|fxyy)
    (fxxx|fxyz) (fxxy|fxyz) (fxxz|fxyz) (fxyy|fxyz) (fxyz|fxyz) (fxzz|fxyz) (fyyy|fxyz) (fyyz|fxyz) (fyzz|fxyz) (fzzz|fxyz)
    (fxxx|fxzz) (fxxy|fxzz) (fxxz|fxzz) (fxyy|fxzz) (fxyz|fxzz) (fxzz|fxzz) (fyyy|fxzz) (fyyz|fxzz) (fyzz|fxzz) (fzzz|fxzz)
    (fxxx|fyyy) (fxxy|fyyy) (fxxz|fyyy) (fxyy|fyyy) (fxyz|fyyy) (fxzz|fyyy) (fyyy|fyyy) (fyyz|fyyy) (fyzz|fyyy) (fzzz|fyyy)
    (fxxx|fyyz) (fxxy|fyyz) (fxxz|fyyz) (fxyy|fyyz) (fxyz|fyyz) (fxzz|fyyz) (fyyy|fyyz) (fyyz|fyyz) (fyzz|fyyz) (fzzz|fyyz)
    (fxxx|fyzz) (fxxy|fyzz) (fxxz|fyzz) (fxyy|fyzz) (fxyz|fyzz) (fxzz|fyzz) (fyyy|fyzz) (fyyz|fyzz) (fyzz|fyzz) (fzzz|fyzz)
    (fxxx|fzzz) (fxxy|fzzz) (fxxz|fzzz) (fxyy|fzzz) (fxyz|fzzz) (fxzz|fzzz) (fyyy|fzzz) (fyyz|fzzz) (fyzz|fzzz) (fzzz|fzzz)
    """
    oot_alpha_ket = 1 / (2 * alpha_ket)
    lower = overlap_fp(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    lower_padded = np.pad(lower.reshape(-1), (1,0))

    # Compute first term of integral derivative equation.
    fx_1, fy_1, fz_1 = jax.jacfwd(overlap_fd, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)

    # coefficient array for second term. mimics number of 'x's in ket in the (f|d) function's returned matrix
    bx = np.tile(np.array([2,1,1,0,0,0]), 10).reshape(10,6).T
    # mimics structure of previous matrix, where nonzero values are replaced with number in numerical order 1,2,3
    take_x = lower_take_mask(bx)

    # coefficient array for second term. mimics number of 'y's in ket in the (f|d) function's returned matrix
    by = np.tile(np.array([0,1,0,2,1,0]), 10).reshape(10,6).T
    # mimics structure of previous matrix, where nonzero values are replaced with number in numerical order 1,2,3
    take_y = lower_take_mask(by)

    # coefficient array for second term. mimics number of 'z's in ket in the (f|d) function's returned matrix
    bz = np.tile(np.array([0,0,1,0,1,2]), 10).reshape(10,6).T
    # mimics structure of previous matrix, where nonzero values are replaced with number in numerical order 1,2,3
    take_z = lower_take_mask(bz)

    # Match proper primitive integral with correct coefficient for creating second term
    fx_2 = bx * np.take(lower_padded, take_x)
    fy_2 = by * np.take(lower_padded, take_y)
    fz_2 = bz * np.take(lower_padded, take_z)

    fx = oot_alpha_ket * (fx_1 + fx_2)
    fy = oot_alpha_ket * (fy_1 + fy_2)
    fz = oot_alpha_ket * (fz_1 + fz_2)

    # slice of 'y' part is equal to the promoted angular momentum component of the result. d=2, f=3, g=4 etc..
    return np.vstack((fx, fy[-3:], fz[-1]))





