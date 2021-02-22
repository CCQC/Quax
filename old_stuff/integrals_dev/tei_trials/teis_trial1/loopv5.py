import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product, find_unique_shells
from functools import partial
from jax.experimental import loops
from pprint import pprint
from eri import *

# Define molecule
molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         units bohr
                         """)

# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))

basis_name = 'cc-pvdz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
# Pytree test
#value_flat, value_tree = jax.tree_util.tree_flatten(basis_dict)
#print(value_flat)


max_prim = basis_set.max_nprimitive()
print(max_prim)
biggest_K = max_prim**4
#pprint(basis_dict)
nbf = basis_set.nbf()
nshells = len(basis_dict)
#unique_shell_quartets = find_unique_shells(nshells)

shell_quartets = old_cartesian_product(np.arange(nshells), np.arange(nshells), np.arange(nshells), np.arange(nshells))

def transform_basisdict(basis_dict, max_prim):
    '''
    Make it so all contractions are the same size in the basis dict by padding exp and coef values to 0 and 0?
    Also create 'indices' key which says where along axis the integral should go, but padded with -1's to maximum angular momentum size
    This allows you to pack them neatly into an array, and then worry about redundant computation later.
    '''
    new_dict = basis_dict.copy()
    for i in range(len(basis_dict)):
        current_exp = onp.asarray(basis_dict[i]['exp'])
        new_dict[i]['exp'] = np.asarray(onp.pad(current_exp, (0, max_prim - current_exp.shape[0])))
        current_coef = onp.asarray(basis_dict[i]['coef'])
        new_dict[i]['coef'] = np.asarray(onp.pad(current_coef, (0, max_prim - current_coef.shape[0])))
        idx, size = basis_dict[i]['idx'], basis_dict[i]['idx_stride']
        indices = onp.repeat(idx, size) + onp.arange(size)
    return new_dict

#TODO this is incorrect, mixes 0's and real values together, not what you want
basis_dict = transform_basisdict(basis_dict, max_prim)
pprint(basis_dict)

def preprocess(shell_quartets, basis_dict):
    coeffs = []
    exps = []
    atoms = []
    ams = []
    indices = []
    sizes = []
    for i in range(nshells):
        c1, exp1, atom1_idx, am1, idx1, size1 = onp.asarray(basis_dict[i]['coef']), onp.asarray(basis_dict[i]['exp']), basis_dict[i]['atom'], basis_dict[i]['am'], basis_dict[i]['idx'], basis_dict[i]['idx_stride']
        coeffs.append(c1)
        exps.append(exp1)
        atoms.append(atom1_idx)
        ams.append(am1)
        indices.append(idx1)
        sizes.append(size1)
    return np.asarray(coeffs), np.asarray(exps), np.asarray(atoms), np.asarray(ams), np.asarray(indices), np.asarray(sizes)

coeffs, exps, atoms, am, indices, sizes = preprocess(shell_quartets, basis_dict)


def sketch():
    G = np.zeros((nbf,nbf,nbf,nbf))
    G = add_to_G

def add_to_G(G, func, geom, coeffs, exps, atoms, am, indices):
    #dim_indices = np.repeat(indices, sizes) + np.arange(sizes)

    with loops.Scope() as s:
        def primitive(A, B, C, D, aa, bb, cc, dd, coeff, func):
            '''Geometry parameters, exponents, coefficients, angular momentum identifier'''
            args = (A, B, C, D, aa, bb, cc, dd, coeff) 
            primitive = np.where(coeff == 0,  0.0, func(*args))
            return primitive
    
        # Computes multiple primitives with same center, angular momentum 
        vectorized_primitive = jax.vmap(primitive, (None,None,None,None,0,0,0,0,0,None))

        ## Computes a contracted integral 
        #@jax.jit
        #@partial(jax.jit, static_argnums=(9))
        def contraction(A, B, C, D, aa, bb, cc, dd, coeff, am):
            primitives = vectorized_primitive(A, B, C, D, aa, bb, cc, dd, coeff, am)
            contraction = np.sum(primitives, axis=0)
            return contraction

        indx_array = np.arange(nshells**4).reshape(nshells,nshells,nshells,nshells) 
        #s.G = np.zeros((nbf,nbf,nbf,nbf))
        s.G = np.ones((nbf,nbf,nbf,nbf))
        idx_vec = np.arange(nbf)
        for i in s.range(nshells):
            A = geom[atoms[i]]
            aa = exps[i]
            c1 = coeffs[i]
            ami = am[i]
            idx1 = indices[i]
            for j in s.range(nshells):
                B = geom[atoms[j]]
                bb = exps[j]
                c2 = coeffs[j]
                amj = am[j]
                idx2 = indices[j]
                for k in s.range(nshells):
                    C = geom[atoms[k]]
                    cc = exps[k]
                    c3 = coeffs[k]
                    amk = am[k]
                    idx3 = indices[k]
                    for l in s.range(nshells):
                        D = geom[atoms[l]]
                        dd = exps[l]
                        c4 = coeffs[l]
                        aml = am[l]
                        idx4 = indices[l]

                        exp_combos = cartesian_product(aa,bb,cc,dd)
                        coeff_combos = np.prod(cartesian_product(c1,c2,c3,c4), axis=1)
                        am_vec = np.array([ami, amj, amk, aml]) 
                        # shape matches how far starting index should be extended
                        val = contraction(A,B,C,D, 
                                          exp_combos[:,0], 
                                          exp_combos[:,1], 
                                          exp_combos[:,2],
                                          exp_combos[:,3],
                                          coeff_combos, am_vec)
                        indices1 = np.repeat(idx1, val.shape[0]) + np.arange(val.shape[0])
                        indices2 = np.repeat(idx2, val.shape[1]) + np.arange(val.shape[1])
                        indices3 = np.repeat(idx3, val.shape[2]) + np.arange(val.shape[2])
                        indices4 = np.repeat(idx4, val.shape[3]) + np.arange(val.shape[3])
                        index_combos = cartesian_product(indices1,indices2,indices3,indices4)
                        #for a,v in enumerate(val.reshape(-1)):
                        #    s.G = jax.ops.index_update(s.G, index_combos[a], v)


                        #s.G = jax.ops.index_update(s.G, (indices1, indices2, indices3, indices4), val)
                        #s.G = jax.ops.index_add(s.G, (indices1, indices2, indices3, indices4), val)
                        #s.G = jax.ops.index_update(s.G, jax.ops.index[indices1, indices2, indices3, indices4], val)
                        #index_combos = cartesian_product(indices1,indices2,indices3,indices4)

                        #for _ in s.cond_range(np.sum(am) == 0):
                        #    s.G = jax.ops.index_update(s.G, (index_combos[:,0], index_combos[:,1], index_combos[:,2], index_combos[:,3]), val.reshape(-1))
                        #for _ in s.cond_range(np.sum(am) == 1):
                        #    s.G = jax.ops.index_update(s.G, (index_combos[:,0], index_combos[:,1], index_combos[:,2], index_combos[:,3]), val.reshape(-1))
                        #for _ in s.cond_range(val.shape == (1,1,1,1)):
                        #    s.G = jax.ops.index_update(s.G, (index_combos[:,0], index_combos[:,1], index_combos[:,2], index_combos[:,3]), val.reshape(-1))
                        #for _ in s.cond_range(val.shape == (3,1,1,1)):
                        #    s.G = jax.ops.index_update(s.G, (index_combos[:,0], index_combos[:,1], index_combos[:,2], index_combos[:,3]), val.reshape(-1))
                        #for _ in s.cond_range(val.shape == (1,1,1,3)):
                        #    s.G = jax.ops.index_update(s.G, (index_combos[:,0], index_combos[:,1], index_combos[:,2], index_combos[:,3]), val.reshape(-1))
                        #for _ in s.cond_range(val.shape != (1,1,1,3)):
                        #    s.G = jax.ops.index_update(s.G, (index_combos[:,0], index_combos[:,1], index_combos[:,2], index_combos[:,3]), val.reshape(-1))

                        
                        #test = np.all(index_combos != -1, axis=1)
                        #index_combos = index_combos[test]
                        #for idx in index_combos:
                        #    for _ in s.cond_range(np.all(idx > -1, axis=0)):
                        #s.G = np.where(np.all(index_combos > -1, axis=1), jax.ops.index_update(s.G, (index_combos[:,0], index_combos[:,1], index_combos[:,2], index_combos[:,3]), val.reshape(-1)), s.G)

                        #s.G = jax.ops.index_update(s.G, (index_combos[:,0], index_combos[:,1], index_combos[:,2], index_combos[:,3]), val.reshape(-1))
                        #s.G = jax.ops.index_update(s.G, (index_combos[:,0], index_combos[:,1], index_combos[:,2], index_combos[:,3]), np.pad(val, (0,81 - val.shape[0])))
                        #s.G = jax.ops.index_update(s.G, (index_combos[:,0], index_combos[:,1], index_combos[:,2], index_combos[:,3]), val.reshape(-1))
        return s.G
        #return s.G[:-1,:-1,:-1,:-1]

G = compute(geom, coeffs, exps, atoms, am, indices)

mints = psi4.core.MintsHelper(basis_set)
psi_G = np.asarray(onp.asarray(mints.ao_eri()))

##print(G)
for i in range(100):
    print(G.flatten()[i])
#, psi_G.flatten()[i])
#print(G[0,0,0,0])
#
#print(psi_G)

