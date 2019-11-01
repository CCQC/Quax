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

@jax.jit
def overlap_ss(A, B, aa, bb):
    ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * np.dot(A-B, A-B)) / (aa + bb)))
    return ss

@jax.jit
def overlap_ps(A, B, alpha_bra, alpha_ket):
    oot_alpha_bra = 1 / (2 * alpha_bra)
    return oot_alpha_bra * jax.jacrev(overlap_ss,0)(A,B,alpha_bra,alpha_ket)

@jax.jit
def overlap_sp(A, B, alpha_bra, alpha_ket):
    return overlap_ps(B,A,alpha_ket,alpha_bra)

@jax.jit
def overlap_pp(A, B, alpha_bra, alpha_ket):
    # We are promoting the ket, so the factor is the ket exponent
    oot_alpha_ket = 1 / (2 * alpha_ket)
    # No second term, ai is 0 since we are promoting the ket and theres no AM in the ke
    return oot_alpha_ket * (jax.jacfwd(overlap_ps, 1)(A,B,alpha_bra,alpha_ket))


# Vectorized version of overlap_ss
vectorized_overlap_ss = jax.jit(jax.vmap(overlap_ss, (None,None,0,0)))
vectorized_overlap_ps = jax.jit(jax.vmap(overlap_ps, (None,None,0,0)))
vectorized_overlap_sp = jax.jit(jax.vmap(overlap_sp, (None,None,0,0)))
vectorized_overlap_pp = jax.jit(jax.vmap(overlap_pp, (None,None,0,0)))

from basis import basis_dict,geom,basis_set

nbf = basis_set.nbf()
nshells = len(basis_dict)
S = np.zeros((nbf,nbf))

overlap_funcs = {}
overlap_funcs['ss'] = vectorized_overlap_ss 
overlap_funcs['ps'] = vectorized_overlap_ps 
overlap_funcs['sp'] = vectorized_overlap_sp 
overlap_funcs['pp'] = vectorized_overlap_pp 

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
    
        lookup = basis_dict[i]['am'] +  basis_dict[j]['am']

        # Expand exponent data to compute all primitive combinations with vectorized overlap function
        exp_combos = cartesian_product(exp1,exp2)
        primitives = overlap_funcs[lookup](A,B,exp_combos[:,0],exp_combos[:,1])
        print(lookup)
        #print('prim',primitives.shape)
        #print(primitives)
        # Build coefficients products for contraction
        coefficients = np.einsum('i,j->ij', c1, c2).flatten()
        #print('coef',coefficients.shape)
        #print(np.broadcast_to(coefficients, primitives.shape))
        coefficients = np.broadcast_to(coefficients, primitives.shape)
        # Contract
        #result = np.sum(primitives * coefficients)
        product = primitives * coefficients
        #print(product.shape)
        #print(np.sum(product, axis=-1))
        result = np.sum(product, axis=-1)
        print(result)
        #result = np.sum(primitives * coefficients, axis=-1)
        #S = jax.ops.index_update(S, jax.ops.index[row_idx,col_idx], result)

#print(S)

#def mapfunction(idx):
#    i,j = idx
#
#    c1 =    np.asarray(basis_dict[i]['coef'])
#    c2 =    np.asarray(basis_dict[j]['coef'])
#    exp1 =  np.asarray(basis_dict[i]['exp'])
#    exp2 =  np.asarray(basis_dict[j]['exp'])
#    atom1 = basis_dict[i]['atom']
#    atom2 = basis_dict[j]['atom']
#    row_idx = basis_dict[i]['idx']
#    col_idx = basis_dict[j]['idx']
#    A = geom[atom1]
#    B = geom[atom2]
#    
#    lookup = basis_dict[i]['am'] +  basis_dict[j]['am']
#
#    # Expand exponent data to compute all primitive combinations with vectorized overlap function
#    exp_combos = cartesian_product(exp1,exp2)
#    primitives = overlap_funcs[lookup](A,B,exp_combos[:,0],exp_combos[:,1])
#    # Build coefficients products for contraction
#    coefficients = np.einsum('i,j->ij', c1, c2).flatten()
#    # Contract
#    result = np.sum(primitives * coefficients)
#    return result
#
#indices = []
#for i in range(nshells):
#    for j in range(i+1):
#        indices.append([i,j])
#indices = np.asarray(indices)
#new = jax.lax.map(mapfunction, indices)
#print(new)

#trythis  = jax.vmap(mapfunction, 0, 0)
#result = trythis(indices)
#print(result)

