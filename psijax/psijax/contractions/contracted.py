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
# Each function is vectorized, so it can take in many combinations of exponents and coefficents at once.

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
    oot_alpha_ket = 1 / (2 * alpha_ket)
    return (oot_alpha_ket * jax.jacrev(overlap_ss,1)(A,B,alpha_bra,alpha_ket,c1,c2)).reshape(-1)

@jax.jit
def overlap_pp(A, B, alpha_bra, alpha_ket,c1,c2):
    # We are promoting the ket, so the factor is the ket exponent
    oot_alpha_ket = 1 / (2 * alpha_ket)
    # No second term, ai is 0 since we are promoting the ket and theres no AM in the ke
    return (oot_alpha_ket * (jax.jacfwd(overlap_ps, 1)(A,B,alpha_bra,alpha_ket,c1,c2))).reshape(-1)

# Vectorized versions of overlap_ss
vectorized_overlap_ss = jax.jit(jax.vmap(overlap_ss, (None,None,0,0,0,0)))
vectorized_overlap_ps = jax.jit(jax.vmap(overlap_ps, (None,None,0,0,0,0)))
vectorized_overlap_sp = jax.jit(jax.vmap(overlap_sp, (None,None,0,0,0,0)))
vectorized_overlap_pp = jax.jit(jax.vmap(overlap_pp, (None,None,0,0,0,0)))

from basis import basis_dict,geom,basis_set

nbf = basis_set.nbf()
nshells = len(basis_dict)
S = np.zeros((nbf,nbf))

overlap_funcs = {}
overlap_funcs['ss'] = vectorized_overlap_ss 
overlap_funcs['ps'] = vectorized_overlap_ps 
overlap_funcs['sp'] = vectorized_overlap_sp 
overlap_funcs['pp'] = vectorized_overlap_pp 

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
        A = geom[atom1]
        B = geom[atom2]
    
        # Function identifier
        lookup = basis_dict[i]['am'] +  basis_dict[j]['am']

        # Expand exponent and coefficeient data to compute all primitive combinations with vectorized functions
        exp_combos = cartesian_product(exp1,exp2)
        coeff_combos = cartesian_product(c1,c2)
        primitives = overlap_funcs[lookup](A,B,exp_combos[:,0],exp_combos[:,1],coeff_combos[:,0],coeff_combos[:,1])
        result = np.sum(primitives, axis=0)

        if lookup == 'ss':
            row_idx_stride = 1 
            col_idx_stride = 1 
        if lookup == 'sp':
            row_idx_stride = 1 
            col_idx_stride = 3 
        if lookup == 'ps':
            row_idx_stride = 3
            col_idx_stride = 1
        if lookup == 'pp':
            row_idx_stride = 3
            col_idx_stride = 3 

        row_indices = np.repeat(row_idx, row_idx_stride)+ np.arange(row_idx_stride)
        col_indices = np.repeat(col_idx, col_idx_stride)+ np.arange(col_idx_stride)
        indices = cartesian_product(row_indices,col_indices)
        S = jax.ops.index_update(S, (indices[:,0],indices[:,1]), result)
        #print(lookup)
        #print(row_indices)
        #print(col_indices)
        #print(indices)
        #print(primitives.shape)
        #print(primitives)
        #print(row_idx, col_idx)
        #print(np.sum(primitives, axis=0))
        #print(primitives.shape)
        #S.append(result)
#print(np.asarray(S))
print(S)

