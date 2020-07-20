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
    alpha_sum = alpha_bra + alpha_ket
    return c1 * c2 * (np.pi / alpha_sum)**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(A-C, A-C)) / alpha_sum)

def overlap_ps(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (px|s) (py|s) (pz|s)
    """
    oot_alpha_bra = 1 / (2 * alpha_bra)
    first_term = np.asarray(jax.jacrev(overlap_ss, (0,1,2))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
    return oot_alpha_bra * first_term

def overlap_ds(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (dxx|s) (dxy|s) (dxz|s) (dyy|s) (dyz|s) (dzz|s) 
    """
    oot_alpha_bra = 1 / (2 * alpha_bra)
    dx_first_term, dy_first_term, dz_first_term = jax.jacfwd(overlap_ps, (0,1,2))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    lower = overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)

    dx = dx_first_term + np.array([lower,0,0])
    dy = dy_first_term[1:] + np.array([lower,0])
    dz = dz_first_term[-1] + lower
    return  oot_alpha_bra * np.hstack((dx, dy, dz))

def overlap_fs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (fxxx|s) (fxxy|s) (fxxz|s) (fxyy|s) (fxyz|s) (fxzz|s) (fyyy|s) (fyyz|s) (fyzz|s) (fzzz|s)
    """
    oot_alpha_bra = 1 / (2 * alpha_bra)
    # pad the lower angluar momentum integral (a-1i|b) with 1 zero on each side.
    lower = np.pad(overlap_ps(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2), 1)
    fx_first_term, fy_first_term, fz_first_term = jax.jacfwd(overlap_ds, (0,1,2))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)

    # First vector mimics number of x's in (dxx|s) (dxy|s) (dxz|s) (dyy|s) (dyz|s) (dzz|s) 
    # Second vector mimics structure of first vector, in numerical order 1,2,3...
    fx_second_term = np.array([2,1,1,0,0,0]) * np.take(lower, [1,2,3,0,0,0])
    fx = oot_alpha_bra * (fx_first_term + fx_second_term)

    # First vector mimics number of y's in (dxx|s) (dxy|s) (dxz|s) (dyy|s) (dyz|s) (dzz|s) 
    # Second vector mimics structure of first vector, in numerical order 1,2,3...
    fy_second_term = np.array([0,1,0,2,1,0]) * np.take(lower, [0,1,0,2,3,0])
    fy = oot_alpha_bra * (fy_first_term + fy_second_term)

    # First vector mimics number of z's in  (dxx|s) (dxy|s) (dxz|s) (dyy|s) (dyz|s) (dzz|s) 
    # Second vector mimics structure of first vector, in numerical order 1,2,3...
    fz_second_term = np.array([0,0,1,0,1,2]) * np.take(lower, [0,0,1,0,2,3])
    fz = oot_alpha_bra * (fz_first_term + fz_second_term)
    return np.hstack((fx, fy[-3:], fz[-1]))

def new_overlap_pp(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    '''
    Use relation [a|c+1i] = 1/2beta * (ai [a-1i|c] + ci [a|c-1i] - 2alpha [a+1i|c])
    to returns the upper triangle of 
    (px|px) (px|py) (px|pz)
    (py|px) (py|py) (py|pz)
    (pz|px) (pz|py) (pz|pz)
    '''
    args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    oot_alpha_ket = 1 / (2 * alpha_ket)
    tmp1 = overlap_ss(*args)
    bra_lower = np.hstack((tmp1, 0, 0, tmp1, 0, tmp1))
    return oot_alpha_ket * (bra_lower - 2 * alpha_bra * overlap_ds(*args))

def new_overlap_dp(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    '''
          ( a |c+1i)                (a-1i|c) (a|c-1i)      (a+1i|c)
a = dij, c = s
1i == x;  (dxx|px) = (1/2*gamma) * [2*(px|s) + 0 - 2*alpha*(fxxx|s)]  0
1i == y;  (dxx|py) = (1/2*gamma) * [0        + 0 - 2*alpha*(fxxy|s)]  1
1i == z;  (dxx|pz) = (1/2*gamma) * [0        + 0 - 2*alpha*(fxxz|s)]  2

1i == x;  (dxy|px) = (1/2*gamma) * [1*(py|s) + 0 - 2*alpha*(fxxy|s)]  1
1i == y;  (dxy|py) = (1/2*gamma) * [1*(px|s) + 0 - 2*alpha*(fxyy|s)]  3
1i == z;  (dxy|pz) = (1/2*gamma) * [0        + 0 - 2*alpha*(fxyz|s)]  4

1i == x;  (dxz|px) = (1/2*gamma) * [1*(pz|s) + 0 - 2*alpha*(fxxz|s)]  2
1i == y;  (dxz|py) = (1/2*gamma) * [0        + 0 - 2*alpha*(fxyz|s)]  4
1i == z;  (dxz|pz) = (1/2*gamma) * [1*(px|s) + 0 - 2*alpha*(fxzz|s)]  5

1i == x;  (dyy|px) = (1/2*gamma) * [0        + 0 - 2*alpha*(fxyy|s)]  3
1i == y;  (dyy|py) = (1/2*gamma) * [2*(py|s) + 0 - 2*alpha*(fyyy|s)]  6
1i == z;  (dyy|pz) = (1/2*gamma) * [0        + 0 - 2*alpha*(fyyz|s)]  7

1i == x;  (dyz|px) = (1/2*gamma) * [0        + 0 - 2*alpha*(fxyz|s)]  4
1i == y;  (dyz|py) = (1/2*gamma) * [1*(pz|s) + 0 - 2*alpha*(fyyz|s)]  7
1i == z;  (dyz|pz) = (1/2*gamma) * [1*(py|s) + 0 - 2*alpha*(fyzz|s)]  8

1i == x;  (dzz|px) = (1/2*gamma) * [0        + 0 - 2*alpha*(fxzz|s)]  5
1i == y;  (dzz|px) = (1/2*gamma) * [0        + 0 - 2*alpha*(fyzz|s)]  8
1i == z;  (dzz|px) = (1/2*gamma) * [2*(pz|s) + 0 - 2*alpha*(fzzz|s)]  9
    '''
    bra_down = overlap_ps(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    ket_down = 0
    bra_up   = overlap_fs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    
    result = np.array([2*bra_down[0] - 2 * alpha_bra * bra_up[0],
                       2 * alpha_bra * bra_up[1],
                       2 * alpha_bra * bra_up[2],
                       bra_down[1] - 2 * alpha_bra * bra_up[1],
                       bra_down[0] - 2 * alpha_bra * bra_up[1],
                       2 * alpha_bra * bra_up[4],
                       bra_down[2] - 2 * alpha_bra * bra_up[2],
                       2 * alpha_bra * bra_up[4],

    return 0


    #return oot_alpha_ket * (overlap_ss(*args) - 2 * alpha_bra * overlap_ds(*args))

def overlap_pp(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (px|s) (py|s) (pz|s)
    """
    oot_alpha_ket = 1 / (2 * alpha_ket)
    #first_term = np.asarray(jax.jacfwd(overlap_ps, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))   
    first_term = np.asarray(jax.jacrev(overlap_ps, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))   
    #first_term = np.asarray(jax.jacfwd(overlap_ps, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))   
    #first_term = np.asarray(jacfwd_overlap_ps(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
    return oot_alpha_ket * first_term





