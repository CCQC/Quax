import jax
import jax.numpy as np
import numpy as onp
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

@jax.jit
def overlap_ss(A, B, aa, bb, c1=1, c2=1):
    ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * np.dot(A-B, A-B)) / (aa + bb)))
    return np.array([ss * c1 * c2])

@jax.jit
def overlap_ps(A, B, alpha_bra, alpha_ket,c1,c2):
    oot_alpha_bra = 1 / (2 * alpha_bra)
    return (oot_alpha_bra * jax.jacrev(overlap_ss,0)(A,B,alpha_bra,alpha_ket,c1,c2)).reshape(-1)

@jax.jit
def overlap_sp(A, B, alpha_bra, alpha_ket,c1,c2):
    return overlap_ps(B, A, alpha_ket, alpha_bra,c2,c1)

@jax.jit
def overlap_pp(A, B, alpha_bra, alpha_ket,c1,c2):
    # We are promoting the ket, so the factor is the ket exponent
    oot_alpha_ket = 1 / (2 * alpha_ket)
    # No second term, ai is 0 since we are promoting the ket and theres no AM in the ke
    return (oot_alpha_ket * (jax.jacfwd(overlap_ps, 1)(A,B,alpha_bra,alpha_ket,c1,c2))).reshape(-1)

@jax.jit
def overlap_ds(A,B,alpha_bra,alpha_ket,c1,c2):
    '''
    Returns a 1x6 array:
    (dxx,s) (dxy,s)  (dxz,s) (dyy,s) (dyz,s) (dzz,s) 
    '''
    # We are promoting the bra a second time, factor is bra exponent
    oot_alpha_bra = 1 / (2 * alpha_bra)
    #                      # This is of shape (3,3) all dij combos symmetric matrix    # Thus a_i factor has to be 3x3 identity, so that only 
    result = oot_alpha_bra * (jax.jacfwd(overlap_ps, 0)(A,B,alpha_bra,alpha_ket,c1,c2) + np.eye(3) * overlap_ss(A,B,alpha_bra,alpha_ket,c1,c2))  
    #return result.reshape(-1)
    # This result is a 3x3 array containing all (dxx,s) (dxy,s) (dyx,s), only need upper or lower triangle
    # Return upper triangle ((dxx, dxy, dxz, dyy, dyz, dzz) | s) as a vector
    iu = np.triu_indices(3)
    return result[iu]

@jax.jit
def overlap_sd(A,B,alpha_bra,alpha_ket,c1,c2):
    return overlap_ds(B,A,alpha_ket,alpha_bra,c2,c1)

@jax.jit
def overlap_dp(A,B,alpha_bra,alpha_ket,c1,c2): 
    '''
    Returns a 1x18 array:
    (dxx,px) (dxx,py) (dxx,pz) (dxy,px) (dxy,py) (dxy,pz) (dxz,px) (dxz,py) (dxz,pz) (dyy,px) (dyy,py) (dyy,pz) (dyz,px) (dyz,py) (dyz,pz) (dzz,px) (dzz,py) (dzz,pz)
    '''
    oot_alpha_ket = 1 / (2 * alpha_ket) # use ket, since we are promoting ket from s-->p
    # This is a 18x1 array of d by p functions. Could also use overlap_pp_block instead, i think? 
    return (oot_alpha_ket * jax.jacfwd(overlap_ds, 1)(A,B,alpha_bra,alpha_ket,c1,c2)).reshape(-1)

@jax.jit
def overlap_pd(A,B,alpha_bra,alpha_ket,c1,c2):
    return overlap_dp(B,A,alpha_ket,alpha_bra,c2,c1)

@jax.jit
def overlap_dd(A,B,alpha_bra,alpha_ket,c1,c2): 
    '''
    Returns flattened 6x6 array:
    (dxx,dxx) (dxx,dxy) (dxx,dxz) (dxx,dyy) (dxx,dyz) (dxx,dzz)
    (dxy,dxx) (dxy,dxy) (dxy,dxz) (dxy,dyy) (dxy,dyz) (dxy,dzz)
    (dxz,dxx) (dxz,dxy) (dxz,dxz) (dxz,dyy) (dxz,dyz) (dxz,dzz)
    (dyy,dxx) (dyy,dxy) (dyy,dxz) (dyy,dyy) (dyy,dyz) (dyy,dzz)
    (dyz,dxx) (dyz,dxy) (dyz,dxz) (dyz,dyy) (dyz,dyz) (dyz,dzz)
    (dzz,dxx) (dzz,dxy) (dzz,dxz) (dzz,dyy) (dzz,dyz) (dzz,dzz)
    '''
    oot_alpha_ket = 1 / (2 * alpha_ket) # use ket, since we are promoting ket from p-->d
    # The jacfwd (first) term is an 18x3 array           # ai coeffs are   # the second term is
    # (dxx,px) --> (dxx,dxx) (dxx, dxy), (dxx, dxz)      1, 0, 0           (dxx|s) (dxx|s) (dxx|s)
    # (dxx,py) --> (dxx,dyx) (dxx, dyy), (dxx, dyz)      0, 1, 0           (dxx|s) (dxx|s) (dxx|s)
    # (dxx,pz) --> (dxx,dzx) (dxx, dzy), (dxx, dzz)      0, 0, 1           (dxx|s) (dxx|s) (dxx|s)
    # (dxy,px) --> (dxy,dxx) (dxy, dxy), (dxy, dxz)      1, 0, 0           (dxy|s) (dxy|s) (dxy|s)
    # (dxy,py) --> (dxy,dyx) (dxy, dyy), (dxy, dyz)      0, 1, 0           (dxy|s) (dxy|s) (dxy|s)
    # (dxy,pz) --> (dxy,dzx) (dxy, dzy), (dxy, dzz)      0, 0, 1           (dxy|s) (dxy|s) (dxy|s)
    # ....                                               ...              
    # (dzz,px) --> (dzz,dxx) (dzz, dxy), (dzz, dxz)      1, 0, 0           (dzz|s) (dzz|s) (dzz|s)
    # (dzz,py) --> (dzz,dyx) (dzz, dyy), (dzz, dyz)      0, 1, 0           (dzz|s) (dzz|s) (dzz|s)
    # (dzz,pz) --> (dzz,dzx) (dzz, dzy), (dzz, dzz)      0, 0, 1           (dzz|s) (dzz|s) (dzz|s)
    first_term = jax.jacfwd(overlap_dp, 1)(A,B,alpha_bra,alpha_ket,c1,c2)
    factor = np.tile(np.eye(3),(6,1))
    tmp_second_term = overlap_ds(A,B,alpha_bra,alpha_ket,c1,c2)
    second_term = factor * np.repeat(tmp_second_term, 9).reshape(18,3)
    result = oot_alpha_ket * (first_term + second_term)
    iu1,iu2 = np.triu_indices(3)
    result = result.reshape(6,3,3)[:,iu1,iu2].reshape(6,6)
    return result.reshape(-1)


# Vectorized versions of overlap_ss
vectorized_overlap_ss = jax.jit(jax.vmap(overlap_ss, (None,None,0,0,0,0)))
vectorized_overlap_ps = jax.jit(jax.vmap(overlap_ps, (None,None,0,0,0,0)))
vectorized_overlap_sp = jax.jit(jax.vmap(overlap_sp, (None,None,0,0,0,0)))
vectorized_overlap_pp = jax.jit(jax.vmap(overlap_pp, (None,None,0,0,0,0)))
vectorized_overlap_ds = jax.jit(jax.vmap(overlap_ds, (None,None,0,0,0,0)))
vectorized_overlap_sd = jax.jit(jax.vmap(overlap_sd, (None,None,0,0,0,0)))
vectorized_overlap_dp = jax.jit(jax.vmap(overlap_dp, (None,None,0,0,0,0)))
vectorized_overlap_pd = jax.jit(jax.vmap(overlap_pd, (None,None,0,0,0,0)))
vectorized_overlap_dd = jax.jit(jax.vmap(overlap_dd, (None,None,0,0,0,0)))

from basis import basis_dict,geom,basis_set

nbf = basis_set.nbf()
nshells = len(basis_dict)
S = np.zeros((nbf,nbf))

overlap_funcs = {}
overlap_funcs['ss'] = vectorized_overlap_ss 
overlap_funcs['ps'] = vectorized_overlap_ps 
overlap_funcs['sp'] = vectorized_overlap_sp 
overlap_funcs['pp'] = vectorized_overlap_pp 
overlap_funcs['ds'] = vectorized_overlap_ds
overlap_funcs['sd'] = vectorized_overlap_sd
overlap_funcs['dp'] = vectorized_overlap_dp
overlap_funcs['pd'] = vectorized_overlap_pd
overlap_funcs['dd'] = vectorized_overlap_dd


S = np.zeros((nbf,nbf))

for i in range(nshells):
    for j in range(nshells):
        # Load data for this contracted integral
        c1 =    np.asarray(basis_dict[i]['coef'])
        c2 =    np.asarray(basis_dict[j]['coef'])
        exp1 =  np.asarray(basis_dict[i]['exp'])
        exp2 =  np.asarray(basis_dict[j]['exp'])
        atom1 = basis_dict[i]['atom']
        atom2 = basis_dict[j]['atom']
        row_idx = basis_dict[i]['idx']
        col_idx = basis_dict[j]['idx']
        row_idx_stride = basis_dict[i]['idx_stride']
        col_idx_stride = basis_dict[j]['idx_stride']
        A = geom[atom1]
        B = geom[atom2]
    
        # Function identifier
        lookup = basis_dict[i]['am'] +  basis_dict[j]['am']

        # Expand exponent and coefficeient data to compute all primitive combinations with vectorized functions
        exp_combos = cartesian_product(exp1,exp2)
        coeff_combos = cartesian_product(c1,c2)
        primitives = overlap_funcs[lookup](A,B,exp_combos[:,0],exp_combos[:,1],coeff_combos[:,0],coeff_combos[:,1])
        result = np.sum(primitives, axis=0)

        row_indices = np.repeat(row_idx, row_idx_stride)+ np.arange(row_idx_stride)
        col_indices = np.repeat(col_idx, col_idx_stride)+ np.arange(col_idx_stride)
        indices = cartesian_product(row_indices,col_indices)
        S = jax.ops.index_update(S, (indices[:,0],indices[:,1]), result)

print(S[:,20:])
# you could have some preprocessing step, which loops over the shell data, collects it and preps it to pass to one of several
# jax.lax.map'd functions

# sketch a function that does what the loop does, which is jittable
def ss_sketch(func, A,B, exp1,exp2,c1,c2, row_idx, row_idx_stride, col_idx, col_idx_stride):
    exp_combos = cartesian_product(exp1,exp2)
    coeff_combos = cartesian_product(c1,c2)
    primitives = overlap_funcs['ss'](A,B,exp_combos[:,0],exp_combos[:,1],coeff_combos[:,0],coeff_combos[:,1])
    result = np.sum(primitives, axis=0)
    return result
    


