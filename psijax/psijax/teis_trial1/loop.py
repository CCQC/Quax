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
    '''Make it so all contractions are the same size in the basis dict by padding exp and coef values to 0 and 0?
    This allows you to pack them neatly into an array, and then worry about redundant computation later.

    '''
    new_dict = basis_dict.copy()
    for i in range(len(basis_dict)):
        current_exp = onp.asarray(basis_dict[i]['exp'])
        new_dict[i]['exp'] = np.asarray(onp.pad(current_exp, (0, max_prim - current_exp.shape[0])))
        current_coef = onp.asarray(basis_dict[i]['coef'])
        new_dict[i]['coef'] = np.asarray(onp.pad(current_coef, (0, max_prim - current_coef.shape[0])))
    return new_dict


#TODO this is incorrect, mixes 0's and real values together, not what you want
basis_dict = transform_basisdict(basis_dict, max_prim)

#print("number of basis functions", nbf)
#print("number of shells ", nshells)
#print("number of shell quartets", shell_quartets.shape[0])
#print("Max primitives: ", max_prim)
#print("Biggest contraction: ", biggest_K)

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
                                                          #TODO hard coded, needs to subtract from largest am size (idx_stride)
        INDICES = onp.pad((onp.repeat(idx1, size1) + onp.arange(size1)), (0,3-size1), constant_values=-1)
        print(INDICES)
        

    return np.asarray(coeffs), np.asarray(exps), np.asarray(atoms), np.asarray(ams), np.asarray(indices), np.asarray(sizes)

        
coeffs, exps, atoms, am, indices, sizes = preprocess(shell_quartets, basis_dict)
print(indices.dtype)
print(sizes.dtype)
#print("coeffs", coeffs.shape)
#print("exps", exps.shape)
#print("atoms", atoms.shape)
#print("am", am.shape)
#print("indices", indices.shape)
#print("sizes", sizes.shape)
#print(am)

#print(exps[1])

#print(cartesian_product(exps[1],exps[1],exps[0],exps[0]))
#print(am)
#print('angular momentum of the 5th shell quartet')
#print(np.array([am[0,5], am[1,5], am[2,5], am[3,5]]))


def compute(geom, coeffs, exps, atoms, am, indices, sizes):
    #dim_indices = np.repeat(indices, sizes) + np.arange(sizes)
    print(indices)
    with loops.Scope() as s:
        def primitive(A, B, C, D, aa, bb, cc, dd, coeff, am):
            '''Geometry parameters, exponents, coefficients, angular momentum identifier'''
            args = (A, B, C, D, aa, bb, cc, dd, coeff) 
            
            with loops.Scope() as S:
                primitive = 0 # TEMP TODO
                for _ in S.cond_range(np.allclose(am,np.array([0,0,0,0]))):
                    #primitive = np.where((np.any((aa,bb,cc,dd)) == 0), 0.0, eri_ssss(*args))
                    primitive = np.where((np.count_nonzero(np.array([aa,bb,cc,dd])) == 4), eri_ssss(*args), 0.0)
                for _ in S.cond_range(np.allclose(am,np.array([1,0,0,0]))):
                    primitive = np.where((np.count_nonzero(np.array([aa,bb,cc,dd])) == 4), eri_psss(*args), 0.0)
                return primitive
    
        # Computes multiple primitives with same center, angular momentum 
        vectorized_primitive = jax.vmap(primitive, (None,None,None,None,0,0,0,0,0,None))

        ## Computes a contracted integral 
        #@jax.jit
        def contraction(A, B, C, D, aa, bb, cc, dd, coeff, am):
            primitives = vectorized_primitive(A, B, C, D, aa, bb, cc, dd, coeff, am)
            return np.sum(primitives, axis=0)

        #TEMP TODO until you figure out index shiz
        #s.G = np.zeros((nshells,nshells,nshells,nshells))
        # create with a 'dump' dimension because of index packing and creation issue
        s.G = np.zeros((nbf,nbf,nbf,nbf))
        idx_vec = np.arange(nbf)
        counti, countj, countk, countl = 0,0,0,0
        for i in s.range(nshells):
            A = geom[atoms[i]]
            aa = exps[i]
            c1 = coeffs[i]
            ami = am[i]
            idx1 = indices[i]
            size1 = sizes[i]
            for j in s.range(nshells):
                B = geom[atoms[j]]
                bb = exps[j]
                c2 = coeffs[j]
                amj = am[j]
                idx2 = indices[j]
                size2 = sizes[j]
                for k in s.range(nshells):
                    C = geom[atoms[k]]
                    cc = exps[k]
                    c3 = coeffs[k]
                    amk = am[k]
                    idx3 = indices[k]
                    size3 = sizes[k]
                    for l in s.range(nshells):
                        D = geom[atoms[l]]
                        dd = exps[l]
                        c4 = coeffs[l]
                        aml = am[l]
                        idx4 = indices[l]
                        size4 = sizes[l]

                        exp_combos = cartesian_product(aa,bb,cc,dd)
                        coeff_combos = np.prod(cartesian_product(c1,c2,c3,c4), axis=1)
                        am_vec = np.array([ami, amj, amk, aml]) 
                        val = contraction(A,B,C,D, 
                                          exp_combos[:,0], 
                                          exp_combos[:,1], 
                                          exp_combos[:,2],
                                          exp_combos[:,3],
                                          coeff_combos, am_vec)

                        # test whether indices can be abstract 
                        # This works because all val's are broadcastable to the indices (3,3,3,3)
                        fake = np.array([idx1,idx2,idx3])
                        #fake = np.array([counti,countj,countk])

                        s.G = jax.ops.index_update(s.G, (fake,fake,fake,fake), val)

                        #s.G = jax.ops.index_update(s.G, (idx_vec[counti:counti+size1],idx_vec[countj:countj+size2],idx_vec[countk:countk+size3],idx_vec[countl:countl+size4]), val)
                        # IDK this may still work, getting a NAN for some reason
                        #for v in val.flatten():
                        #    s.G = jax.ops.index_update(s.G, (idx_vec[counti],idx_vec[countj],idx_vec[countk],idx_vec[countl]), v)
                        #    counti += 1
                        #    countj += 1
                        #    countk += 1
                        #    countl += 1
                


                        #counti += size1
                        #countj += size2
                        #countk += size3
                        #countl += size4
    
                        
                        #s.G = jax.ops.index_update(s.G, (fake,fake,fake,fake), val)

                        #print(val)
                        # index handling
                        #size = 3**np.sum(am_vec)
                        #indices1 = np.tile(np.array([idx1]), np.array([size1])) + np.arange(size1)
                        #indices2 = np.tile(np.array([idx2]), np.array([size2])) + np.arange(size2)
                        #indices3 = np.tile(np.array([idx3]), np.array([size3])) + np.arange(size3)
                        #indices4 = np.tile(np.array([idx4]), np.array([size4])) + np.arange(size4)
                        # Get indices 
                        #indices1 = np.arange(size1)
                        #indices2 = np.arange(size2)
                        #indices3 = np.arange(size3)
                        #indices4 = np.arange(size4)
                        #index = cartesian_product(indices1, indices2, indices3, indices4)

                        #s.G = jax.ops.index_update(s.G, (index[:,0],index[:,1],index[:,2],index[:,3]), val)

                        #s.G = jax.ops.index_update(s.G, jax.ops.index[i,j,k,l], val)


                        #val = np.sum(A) * np.sum(B) * np.sum(C) * np.sum(D)
                        #s.G = jax.ops.index_update(s.G, jax.ops.index[i,j,k,l], val)

        return s.G


G = compute(geom, coeffs, exps, atoms, am, indices, sizes)
print(G[0,0,0,0])
#
#mints = psi4.core.MintsHelper(basis_set)
#psi_G = np.asarray(onp.asarray(mints.ao_eri()))
#print(psi_G)

