import jax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=500)

def cartesian_product(*arrays):
    '''Generalized cartesian product of any number of arrays'''
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


# Function definitions. We always return a vector of all primitive values.
# NOTES:
# The base function just computes a single primitive. 
# The vectorized versions can compute many primitives with the same centers at the same time.
# All functions return a vector of primitives, the number of which is dependent on the angular momentum
# (s|p) creates 3 primitives. (p|p) creates 9 (redundant for now)

# investigate shapes of each function output
#A = np.array([0.0, 0.0, -0.849220457955])
#B = np.array([0.0, 0.0,  0.849220457955])
Ax = 0.0
Ay = 0.0
Az = -0.849220457955
Cx = 0.0
Cy = 0.0
Cz = 0.849220457955
alpha_bra = 0.5 
alpha_ket = 0.5 

c1_S = 0.4237772081237576
c1_P = 0.5993114751532237
c1_D = 0.489335770373359
c1_F = 0.3094831149945914
c1_G = 0.1654256833287603
c1_H = 0.07798241497612321
c1_I = 0.0332518134720999

c1 = c1_F
c2 = c1_F
#c2 = 0.4237772081237576


#@jax.jit
def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    ss = ((np.pi / (alpha_bra + alpha_ket))**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(A-C, A-C)) / (alpha_bra + alpha_ket)))
    return ss * c1 * c2

#@jax.jit
def overlap_ps(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (px|s) (py|s) (pz|s)
    """
    oot_alpha_bra = 1 / (2 * alpha_bra)
    first_term = np.asarray(jax.jacrev(overlap_ss, (0,1,2))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
    return oot_alpha_bra * first_term
    
#@jax.jit
def overlap_ds(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (dxx|s) (dxy|s) (dxz|s) (dyy|s) (dyz|s) (dzz|s) 
    """
    oot_alpha_bra = 1 / (2 * alpha_bra)
    dx_first_term, dy_first_term, dz_first_term = jax.jacfwd(overlap_ps, (0,1,2))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    lower = overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)

    dx_second_term = np.array([1,0,0]) * lower 
    dx = (dx_first_term + dx_second_term)

    dy_second_term = np.array([0,1,0]) * lower
    dy = (dy_first_term + dy_second_term)

    dz_second_term = np.array([0,0,1]) * lower 
    dz = (dz_first_term + dz_second_term)

    # NOTE may be able to build dy, dz terms from dx equations, subbing args,  may not work generally
    return  oot_alpha_bra * np.hstack((dx, dy[1:], dz[-1]))

#@jax.jit
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


#@jax.jit
def overlap_gs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (gxxxx|s) (gxxxy|s) (gxxxz|s) (gxxyy|s) (gxxyz|s) (gxxzz|s) (gxyyy|s) (gxyyz|s) (gxyzz|s) (gxzzz|s)
    (gyyyy|s) (gyyyz|s) (gyyzz|s) (gyzzz|s) 
    (gzzzz|s)
    """
    oot_alpha_bra = 1 / (2 * alpha_bra)
    # pad the lower angular momentum integral (a-1i|b) with 1 zero on each side.
    lower = np.pad(overlap_ds(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2), 1)

    gx_first_term, gy_first_term, gz_first_term = jax.jacfwd(overlap_fs, (0,1,2))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)

                 #this vector mimics 'x' structure in (f|s)  # this vector mimics structure of previous vector, in numerical order
    gx_second_term = np.array([3,2,2,1,1,1,0,0,0,0]) * np.take(lower, [1,2,3,4,5,6,0,0,0,0])
    gx = oot_alpha_bra * (gx_first_term + gx_second_term)

                 #this vector mimics 'y' structure in (f|s)  # this vector mimics structure of previous vector, in numerical order
    gy_second_term = np.array([0,1,0,2,1,0,3,2,1,0]) * np.take(lower, [0,1,0,2,3,0,4,5,6,0])
    gy = oot_alpha_bra * (gy_first_term + gy_second_term)

                 #this vector mimics 'z' structure in (f|s)  # this vector mimics structure of previous vector, in numerical order
    gz_second_term = np.array([0,0,1,0,1,2,0,1,2,3]) * np.take(lower, [0,0,1,0,2,3,0,4,5,6])
    gz = oot_alpha_bra * (gz_first_term + gz_second_term)
    return np.hstack((gx, gy[-4:], gz[-1]))

#@jax.jit
def overlap_hs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (hxxxxx|s) (hxxxxy|s) (hxxxxz|s) (hxxxyy|s) (hxxxyz|s) (hxxxzz|s) (hxxyyy|s) (hxxyyz|s) (hxxyzz|s) (hxxzzz|s)
    (hxyyyy|s) (hxyyyz|s) (hxyyzz|s) (hxyzzz|s) 
    (hxzzzz|s)
    (hyyyyy|s) (hyyyyz|s) (hyyyzz|s) (hyyzzz|s) (hyzzzz|s)
    (hzzzzz|s)
    """
    oot_alpha_bra = 1 / (2 * alpha_bra)
    # pad the lower angular momentum integral (a-1i|b) with 1 zero on each side.
    lower = np.pad(overlap_fs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2), 1)
    hx_first_term, hy_first_term, hz_first_term = jax.jacfwd(overlap_gs, (0,1,2))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)

                 #this vector mimics 'x' structure in (g|s)  # this vector mimics structure of previous vector, in numerical order
    hx_second_term = np.array([4,3,3,2,2,2,1,1,1,1,0,0,0,0,0]) * np.take(lower, [1,2,3,4,5,6,7,8,9,10,0,0,0,0,0])
    hx = oot_alpha_bra * (hx_first_term + hx_second_term)

                 #this vector mimics 'y' structure in (g|s)  # this vector mimics structure of previous vector, in numerical order
    hy_second_term = np.array([0,1,0,2,1,0,3,2,1,0,4,3,2,1,0]) * np.take(lower, [0,1,0,2,3,0,4,5,6,0,7,8,9,10,0])
    hy = oot_alpha_bra * (hy_first_term + hy_second_term)

                 #this vector mimics 'z' structure in (g|s)  # this vector mimics structure of previous vector, in numerical order
    hz_second_term = np.array([0,0,1,0,1,2,0,1,2,3,0,1,2,3,4]) * np.take(lower, [0,0,1,0,2,3,0,4,5,6,0,7,8,9,10])
    hz = oot_alpha_bra * (hz_first_term + hz_second_term)
    return np.hstack((hx, hy[-5:], hz[-1]))

#@jax.jit
def overlap_is(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    oot_alpha_bra = 1 / (2 * alpha_bra)
    # pad the lower angular momentum integral (a-1i|b) with 1 zero on each side.
    lower = np.pad(overlap_gs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2), 1)
    ix_first_term, iy_first_term, iz_first_term = jax.jacfwd(overlap_hs, (0,1,2))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)

                 #this vector mimics 'x' structure in (h|s)  # this vector mimics structure of previous vector, in numerical order
    ix_second_term = np.array([5,4,4,3,3,3,2,2,2,2,1,1,1,1,1,0,0,0,0,0,0]) * np.take(lower, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,0,0,0,0,0])
    ix = oot_alpha_bra * (ix_first_term + ix_second_term)
                              
                 #this vector mimics 'y' structure in (h|s)  # this vector mimics structure of previous vector, in numerical order
    iy_second_term = np.array([0,1,0,2,1,0,3,2,1,0,4,3,2,1,0,5,4,3,2,1,0]) * np.take(lower, [0,1,0,2,3,0,4,5,6,0,7,8,9,10,0,11,12,13,14,15,0])
    iy = oot_alpha_bra * (iy_first_term + iy_second_term)

                 #this vector mimics 'z' structure in (h|s)  # this vector mimics structure of previous vector, in numerical order
    iz_second_term = np.array([0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5]) * np.take(lower, [0,0,1,0,2,3,0,4,5,6,0,7,8,9,10,0,11,12,13,14,15])
    iz = oot_alpha_bra * (iz_first_term + iz_second_term)
    return np.hstack((ix, iy[-6:], iz[-1]))

#print(overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
#print(overlap_ps(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
#print(overlap_ds(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
#print(overlap_fs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
#print(overlap_gs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
#print(overlap_hs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
#print(overlap_is(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))


############
# KET TEST #
############


def overlap_dp(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a matrix: 
    (dxx|px) (dxy|px) (dxz|px) (dyy|px) (dyz|px) (dzz|px) 
    (dxx|py) (dxy|py) (dxz|py) (dyy|py) (dyz|py) (dzz|py) 
    (dxx|pz) (dxy|pz) (dxz|pz) (dyy|pz) (dyz|pz) (dzz|pz) 
    """
    oot_alpha_ket = 1 / (2 * alpha_ket)
    first_term = np.asarray(jax.jacfwd(overlap_ds, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
    return oot_alpha_ket * first_term

def overlap_dd(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    # NOTE this returns redundant integrals NOTE
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

    dx_first_term, dy_first_term, dz_first_term = np.asarray(jax.jacfwd(overlap_dp, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
    print('shape of lower')
    print(lower.shape)
    
       #this matrix should mimic the 'x' structure in the ket of (d|p)  # this vector mimics structure of previous vector, in numerical order

    #dx_second_term = np.array([1,1,1,1,1,1]) * np.take(lower, [1,2,3,4,5,6]

    #factor = np.array([[1,1,1,1,1,1],
    #                   [0,0,0,0,0,0],
    #                   [0,0,0,0,0,0]]) 
    #take = np.take(lower, [1,2,3,4,5,6] 

    #dx_second_term = factor * take

    print('shape of derivatives')
    print(dx_first_term.shape)
    print(dx_first_term)
    print(dy_first_term)
    print(dz_first_term)

    #return np.vstack((dx,dy[-2:],dz[-1]))

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

# note the differences here.  
# * use alpha_ket
# * multidimensional padding 
# * derivative w.r.t. args 3,4,5 to do the ket cartesian coordinates 
# * the coefficients of the second terms are now 2d, and the values are the same except they run down the columns now
# The structure of the coefficient of second terms now follows the 'x', 'y', 'z' structure in the kets of first term
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
    lower_padded = np.pad(lower, (1,0))

    dx_first_term, dy_first_term, dz_first_term, = jax.jacfwd(overlap_fp, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)

    # mimics 'x' ket-structure in (f|p)
    bx = np.array([[1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0]])

    # mimics structure of previous matrix, in numerical order 
    take_x = np.array([[1,2,3,4,5,6,7,8,9,10],
                       [0,0,0,0,0,0,0,0,0, 0],
                       [0,0,0,0,0,0,0,0,0, 0]])

    # mimics 'y' ket-structure in (f|p)
    by = np.array([[0,0,0,0,0,0,0,0,0,0],
                   [1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0]])

    # mimics structure of previous matrix, in numerical order 
    take_y = np.array([[0,0,0,0,0,0,0,0,0, 0],
                       [1,2,3,4,5,6,7,8,9,10],
                       [0,0,0,0,0,0,0,0,0, 0]])

    # mimics 'z' ket-structure in (f|p)
    bz = np.array([[0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0],
                   [1,1,1,1,1,1,1,1,1,1]])

    # mimics structure of previous matrix, in numerical order 
    take_z = np.array([[0,0,0,0,0,0,0,0,0, 0],
                       [0,0,0,0,0,0,0,0,0, 0],
                       [1,2,3,4,5,6,7,8,9,10]])

    dx_second_term = bx * np.take(lower_padded, take_x)
    dy_second_term = by * np.take(lower_padded, take_y)
    dz_second_term = bz * np.take(lower_padded, take_z)

    dx = oot_alpha_ket * (dx_first_term + dx_second_term)
    dy = oot_alpha_ket * (dy_first_term + dy_second_term)
    dz = oot_alpha_ket * (dz_first_term + dz_second_term)
    return np.vstack((dx, dy[-2:], dz[-1]))
    
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

    fx_first_term, fy_first_term, fz_first_term, = jax.jacfwd(overlap_fd, (3,4,5))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)

    # mimics number of 'x's in ket in (f|d)
    bx = np.array([[2,2,2,2,2,2,2,2,2,2], 
                   [1,1,1,1,1,1,1,1,1,1],
                   [1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0]])

    # mimics structure of previous matrix, in numerical order 
    take_x = np.array([[1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10], 
                       [11,12,13,14,15,16,17,18,19,20],
                       [21,22,23,24,25,26,27,28,29,30],
                       [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                       [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                       [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]])

    # mimics number of 'y's in ket in (f|d)
    by = np.array([[0,0,0,0,0,0,0,0,0,0], 
                   [1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0],
                   [2,2,2,2,2,2,2,2,2,2],
                   [1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0]])

    # mimics structure of previous matrix, in numerical order 
    take_y = np.array([[0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                       [1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10], 
                       [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                       [11,12,13,14,15,16,17,18,19,20],
                       [21,22,23,24,25,26,27,28,29,30],
                       [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]])

    # mimics number of 'z's in ket in (f|d)
    bz = np.array([[0,0,0,0,0,0,0,0,0,0], 
                   [0,0,0,0,0,0,0,0,0,0],
                   [1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0],
                   [1,1,1,1,1,1,1,1,1,1],
                   [2,2,2,2,2,2,2,2,2,2]])

    # mimics structure of previous matrix, in numerical order 
    take_z = np.array([[0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                       [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                       [1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10], 
                       [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                       [11,12,13,14,15,16,17,18,19,20],
                       [21,22,23,24,25,26,27,28,29,30]])

    # This take may be incorrect
    fx_second_term = bx * np.take(lower_padded, take_x)
    fy_second_term = by * np.take(lower_padded, take_y)
    fz_second_term = bz * np.take(lower_padded, take_z)

    fx = oot_alpha_ket * (fx_first_term + fx_second_term)
    fy = oot_alpha_ket * (fy_first_term + fy_second_term)
    fz = oot_alpha_ket * (fz_first_term + fz_second_term)
    return np.vstack((fx, fy[-3:], fz[-1]))


#print(overlap_fs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
#print(overlap_fd(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
print(overlap_ff(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))

#print(overlap_dd(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
#print(overlap_dd(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))

@jax.jit
def motherload(i):
    '''
    Make arguments be (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2, slice)
    and preprocess an array of a billion rows of these.
    But how to deal with contractions?
    Provide exp_combos, c_combos as args instead? pass to vmapped functions?
    
    Computes every possible integral. Can provide a slice argument to return proper angular momentum integral
    This is vmappable, laxable, etc
     '''
    s = overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    p = overlap_ps(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    d = overlap_ds(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    f = overlap_fs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    g = overlap_gs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    h = overlap_hs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    i = overlap_is(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    return np.hstack((s,p,d,f,g,h,i))
#print("compiling integral functions")
#print(motherload(0))
#print("compilation complete")
#print("computing 10000 sets of 84 primitive integrals...")
#for i in range(10000):
#    motherload(0)
#print("done")
# about 2 seconds
#result = jax.lax.map(motherload, np.arange(1000000))
#result = jax.lax.map(mapfunc, np.arange(100000))
#print(result.shape)

