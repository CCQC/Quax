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

def get_indices(shell_quartets, basis_dict):
    '''
    Get all indices of ERIs in (nbf**4,4) array. 
    Record where each shell quartet starts and stops along the first axis of this index array.
    '''
    all_indices = []
    for i in range(nshells):
        idx1, size1 = basis_dict[i]['idx'], basis_dict[i]['idx_stride']
        indices1 = onp.repeat(idx1, size1) + onp.arange(size1)
        for j in range(nshells):
            idx2, size2 = basis_dict[j]['idx'], basis_dict[j]['idx_stride']
            indices2 = onp.repeat(idx2, size2) + onp.arange(size2)
            for k in range(nshells):
                idx3, size3 = basis_dict[k]['idx'], basis_dict[k]['idx_stride']
                indices3 = onp.repeat(idx3, size3) + onp.arange(size3)
                for l in range(nshells):
                    idx4, size4 = basis_dict[l]['idx'], basis_dict[l]['idx_stride']
                    indices4 = onp.repeat(idx4, size4) + onp.arange(size4)
                    indices = old_cartesian_product(indices1,indices2,indices3,indices4)
                    indices = onp.pad(indices, ((0, 81-indices.shape[0]),(0,0)), constant_values=-1)
                    all_indices.append(indices)
    return np.asarray(onp.asarray(all_indices))

indices = get_indices(shell_quartets, basis_dict)

def compute(geom, coeffs, exps, atoms, am, indices):
    #dim_indices = np.repeat(indices, sizes) + np.arange(sizes)

    with loops.Scope() as s:
        # DO ALL SORTING, CASE CHECKS IN LOOP,  
        def ssss_primitive(A,B,C,D,aa,bb,cc,dd,coeff):
            return np.where(coeff == 0, 0.0, eri_ssss(A,B,C,D,aa,bb,cc,dd,coeff))

        def ssss_contraction(A,B,C,D,aa,bb,cc,dd,coeff):
            primitives = jax.vmap(ssss_primitive, (None,None,None,None,0,0,0,0,0))(A,B,C,D,aa,bb,cc,dd,coeff)
            contraction = np.sum(primitives, axis=0)
            return contraction.reshape(-1)

        def psss_primitive(A,B,C,D,aa,bb,cc,dd,coeff):
            return np.where(coeff == 0, 0.0, eri_psss(A,B,C,D,aa,bb,cc,dd,coeff))

        def psss_contraction(A,B,C,D,aa,bb,cc,dd,coeff):
            vectorized_primitive = jax.vmap(psss_primitive, (None,None,None,None,0,0,0,0,0))
            primitives = vectorized_primitive(A,B,C,D,aa,bb,cc,dd,coeff)
            contraction = np.sum(primitives, axis=0)
            return contraction.reshape(-1)

        indx_array = np.arange(nshells**4).reshape(nshells,nshells,nshells,nshells) 
        #s.G = np.zeros((nbf,nbf,nbf,nbf))
        s.am = np.zeros(4)
        s.G = np.zeros((nbf+1,nbf+1,nbf+1,nbf+1))
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

                        am_vec = np.array([ami, amj, amk, aml]) 
                        s.am = jax.ops.index_update(s.am, jax.ops.index[0:], np.array([ami, amj, amk, aml]))
                        exp_combos = cartesian_product(aa,bb,cc,dd)
                        aa = exp_combos[:,0]
                        bb = exp_combos[:,1]
                        cc = exp_combos[:,2]
                        dd = exp_combos[:,3]
                        coeff_combos = np.prod(cartesian_product(c1,c2,c3,c4), axis=1)

                        place = indx_array[i,j,k,l]
                        index = indices[place]

                        #val = np.where(np.allclose(am_vec,np.array([0,0,0,0])), np.pad(ssss_contraction(A,B,C,D,aa,bb,cc,dd,coeff_combos), (0,80)), 
                        #      np.where(np.allclose(am_vec,np.array([1,0,0,0])), np.pad(psss_contraction(A,B,C,D,aa,bb,cc,dd,coeff_combos), (0,78)), 0.0))
                        #val = np.where(np.allclose(am_vec,np.array([0,0,0,0])), np.pad(ssss_contraction(A,B,C,D,aa,bb,cc,dd,coeff_combos), (0,80)), 
                        #      np.where(np.allclose(am_vec,np.array([1,0,0,0])), np.pad(psss_contraction(A,B,C,D,aa,bb,cc,dd,coeff_combos), (0,78)), 0.0))

                        # psss function itself is fine.
                        #val = np.pad(psss_contraction(A,B,C,D,aa,bb,cc,dd,coeff_combos), (0,78))

                        # always evaluates to false. am_vec needs to be modified i guess! #TODO TODO TODO TODO TODO TODO make into scope field?
                        #val = np.where(np.allclose(am_vec,np.array([1,0,0,0])), np.pad(psss_contraction(A,B,C,D,aa,bb,cc,dd,coeff_combos), (0,78)), 1.0)
                        #val = np.where(s.am[0] == 1, np.pad(psss_contraction(A,B,C,D,aa,bb,cc,dd,coeff_combos), (0,78)), 1.0)
                        val = np.where(ami == 1, np.pad(psss_contraction(A,B,C,D,aa,bb,cc,dd,coeff_combos), (0,78)), 1.0)
                        s.G = jax.ops.index_update(s.G, (index[:,0], index[:,1], index[:,2], index[:,3]), val)

                        #for _ in s.cond_range(np.allclose(am_vec,np.array([0,0,0,0]))):
                        #    val = ssss_contraction(A,B,C,D,aa,bb,cc,dd,coeff_combos)
                        #    t = val.shape[0]
                        #    s.G = jax.ops.index_update(s.G, (index[:t,0], index[:t,1], index[:t,2], index[:t,3]), val)

                        #for _ in s.cond_range(np.allclose(s.am,np.array([1,0,0,0]))):
                        #for _ in s.cond_range(i % 2 == 0):  # if p function in ishell

                        #args = (A,B,C,D,aa,bb,cc,dd,coeff_combos)
                        #val = jax.lax.cond(np.allclose(am_vec,np.array([1,0,0,0])), args, psss_contraction, np.array([0.0,0.0,0.0]), lambda x: x)
                        #s.G = jax.ops.index_update(s.G, (index[:,0], index[:,1], index[:,2], index[:,3]), np.pad(val, (0, 78)))

                        #if i > 0:
                        #    val = psss_contraction(A,B,C,D,aa,bb,cc,dd,coeff_combos)
                        #    s.G = jax.ops.index_update(s.G, (index[:,0], index[:,1], index[:,2], index[:,3]), np.pad(val, (0, 78)))


                        #for _ in s.cond_range(i > 0):  # if p function in ishell
                            #val = psss_contraction(A,B,C,D,aa,bb,cc,dd,coeff_combos)
                            #s.G = jax.ops.index_update(s.G, (index[:,0], index[:,1], index[:,2], index[:,3]), np.pad(val, (0, 78)))
                            #TODO debug
                            #val = ssss_contraction(A,B,C,D,aa,bb,cc,dd,coeff_combos)
                            #s.G = jax.ops.index_update(s.G, (index[:,0], index[:,1], index[:,2], index[:,3]), np.pad(val, (0, 80)))
                            #print(val)
                            #t = val.shape[0]
                            #s.G = jax.ops.index_update(s.G, (index[:t,0], index[:t,1], index[:t,2], index[:t,3]), val)

        return s.G[:-1,:-1,:-1,:-1]

G = compute(geom, coeffs, exps, atoms, am, indices)

mints = psi4.core.MintsHelper(basis_set)
psi_G = np.asarray(onp.asarray(mints.ao_eri()))

##print(G)
for i in range(100):
    print(G.flatten()[i], psi_G.flatten()[i])
#print(G[0,0,0,0])
#
#print(psi_G)

