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

c1_F = 0.3094831149945914
c1_G = 0.1654256833287603
c1_H = 0.07798241497612321
c1_I = 0.0332518134720999

c1 = c1_I
c2 = 0.4237772081237576


# NOTE: you can evaluate the first terms simulatneoulys with jacfwd( (0,1,2)), unpack the tuple
@jax.jit
def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    ss = ((np.pi / (alpha_bra + alpha_ket))**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(A-C, A-C)) / (alpha_bra + alpha_ket)))
    return ss * c1 * c2

@jax.jit
def overlap_ps(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (px|s) (py|s) (pz|s)
    """
    oot_alpha_bra = 1 / (2 * alpha_bra)
    first_term = np.asarray(jax.jacrev(overlap_ss, (0,1,2))(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
    return oot_alpha_bra * first_term
    
@jax.jit
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


# NOTE: you can evaluate the first terms simulatneoulys with jacfwd( (0,1,2)), unpack the tuple
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
print(overlap_is(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))

def motherload(i):
    '''
    Make arguments be (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2, slice)
    and preprocess an array of a billion rows of these.
    But how to deal with contractions?
    Provide exp_combos, c_combos as args instead? pass to vmapped functions?
    
    Computes every possible integral. Can provide a slice argument to return proper angular momentum integral
    This is vmappable, laxable, etc
     '''
    a = overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    b = overlap_ps(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    c = overlap_ds(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    d = overlap_fs(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    return np.hstack((a,b,c,d))

#result = jax.lax.map(mapfunc, np.arange(100000))
#result = jax.lax.map(mapfunc, np.arange(100000))
#print(result.shape)


