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
        #TODO
        new_dict[i]['exp'] = np.asarray(onp.pad(current_exp, (0, max_prim - current_exp.shape[0])))
        current_coef = onp.asarray(basis_dict[i]['coef'])
        #TODO
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

def get_indices(shell_quartets, basis_dict):
    '''
    Get all indices of ERIs in (nbf**4,4) array. 
    Record where each shell quartet starts and stops along the first axis of this index array.
    '''
    all_indices = []
    starts = []
    stops = []
    start = 0
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
                    all_indices.append(indices)
                    
                    stop = start + indices.shape[0] #how much of the indices array this integral takes up
                    starts.append(start)
                    stops.append(stop)
                    start += indices.shape[0]
                    # this would be in the same order as that which appears in the JAX loop, 
                    # so theoretically could just stack them along the index axis?   

    # NOTE there may be an issue at the ending index 'stop' point, might complain about going out of range
    final_indices = np.asarray(onp.vstack(all_indices))
    starts = np.asarray(onp.asarray(starts))
    stops = np.asarray(onp.asarray(stops))
    return final_indices, starts, stops

new_indices, starts, stops = get_indices(shell_quartets, basis_dict)


def debug(geom, coeffs, exps, atoms, am):

    def primitive(A, B, C, D, aa, bb, cc, dd, coeff, am):
        '''Geometry parameters, exponents, coefficients, angular momentum identifier'''
        args = (A, B, C, D, aa, bb, cc, dd, coeff) 
        
        if (np.allclose(am,np.array([0,0,0,0]))):
                primitive = np.where(coeff == 0,  0.0, eri_ssss(*args)).reshape(-1)

        elif (np.allclose(am,np.array([1,0,0,0]))):
                primitive = np.where(coeff == 0,  0.0, eri_psss(*args)).reshape(-1)
        elif (np.allclose(am,np.array([0,1,0,0]))): # WRONG TODO
                primitive = np.where(coeff == 0,  0.0, eri_psss(*args)).reshape(-1)
        elif (np.allclose(am,np.array([0,0,1,0]))): # should be valid
                primitive = np.where(coeff == 0,  0.0, eri_psss(*args)).reshape(-1)
        elif (np.allclose(am,np.array([0,0,0,1]))): # WRONG TODO
                primitive = np.where(coeff == 0,  0.0, eri_psss(*args)).reshape(-1)

        elif np.allclose(am,np.array([1,1,0,0])):
                primitive = np.where(coeff == 0,  0.0, eri_ppss(*args)).reshape(-1)
        elif np.allclose(am,np.array([0,1,1,0])): #WRONG TODO 
                primitive = np.where(coeff == 0,  0.0, eri_ppss(*args)).reshape(-1)
        elif np.allclose(am,np.array([0,0,1,1])): #WRONG TODO 
                primitive = np.where(coeff == 0,  0.0, eri_ppss(*args)).reshape(-1)

        elif np.allclose(am,np.array([1,0,1,0])):
                primitive = np.where(coeff == 0,  0.0, eri_psps(*args)).reshape(-1)
        elif np.allclose(am,np.array([0,1,0,1])): #WRONG TODO
                primitive = np.where(coeff == 0,  0.0, eri_psps(*args)).reshape(-1)
        elif np.allclose(am,np.array([1,0,0,1])): #WRONG TODO
                primitive = np.where(coeff == 0,  0.0, eri_psps(*args)).reshape(-1)

        elif np.allclose(am,np.array([1,1,1,0])):
                  primitive = np.where(coeff == 0,  0.0, eri_ppps(*args)).reshape(-1)
        elif np.allclose(am,np.array([1,1,0,1])): #WRONG TODO
                  primitive = np.where(coeff == 0,  0.0, eri_ppps(*args)).reshape(-1)
        elif np.allclose(am,np.array([1,0,1,1])): #WRONG TODO
                  primitive = np.where(coeff == 0,  0.0, eri_ppps(*args)).reshape(-1)
        elif np.allclose(am,np.array([0,1,1,1])): #WRONG TODO
                primitive = np.where(coeff == 0,  0.0, eri_ppps(*args)).reshape(-1)

        elif np.allclose(am,np.array([1,1,1,1])):
                primitive = np.where(coeff == 0,  0.0, eri_pppp(*args)).reshape(-1)
        return primitive
    
    vectorized_primitive = jax.vmap(primitive, (None,None,None,None,0,0,0,0,0,None))
    #TODO DEBUG
 
    def contraction(A, B, C, D, aa, bb, cc, dd, coeff, am):
        primitives = vectorized_primitive(A, B, C, D, aa, bb, cc, dd, coeff, am)
        return np.sum(primitives, axis=0)
    #TODO DEBUG
        #TODO wronng dimension
    G = np.zeros((nbf,nbf,nbf,nbf))


    count = 0
    for i in range(nshells):
    #TODO DEBUG
        A = geom[atoms[i]]
        aa = exps[i]
        c1 = coeffs[i]
        ami = am[i]
        idx1 = indices[i]
        for j in range(nshells):
    #TODO DEBUG
            B = geom[atoms[j]]
            bb = exps[j]
            c2 = coeffs[j]
            amj = am[j]
            idx2 = indices[j]
            for k in range(nshells):
    #TODO DEBUG
                C = geom[atoms[k]]
                cc = exps[k]
                c3 = coeffs[k]
                amk = am[k]
                idx3 = indices[k]
                for l in range(nshells):
                    D = geom[atoms[l]]
                    dd = exps[l]
                    c4 = coeffs[l]
                    aml = am[l]
                    idx4 = indices[l]
 
                    exp_combos = cartesian_product(aa,bb,cc,dd)
                    coeff_combos = np.prod(cartesian_product(c1,c2,c3,c4), axis=1)
                    am_vec = np.array([ami, amj, amk, aml]) 
                    val = contraction(A,B,C,D, 
                                      exp_combos[:,0], 
                                      exp_combos[:,1], 
                                      exp_combos[:,2],
                                      exp_combos[:,3],
                                      coeff_combos, am_vec)
                    

                    index = new_indices[starts[count]:stops[count]]
                    print(index)
                    G = jax.ops.index_update(G, jax.ops.index[index[:,0], index[:,1], index[:,2], index[:,3]], val.reshape(-1))
                    count += 1
                    print(val.shape)
                    #if val != val:
                    #    print(coeff_combos)
                    #G = jax.ops.index_update(G, jax.ops.index[i,j,k,l], val)
    return G

G = debug(geom, coeffs, exps, atoms, am)

mints = psi4.core.MintsHelper(basis_set)
psi_G = np.asarray(onp.asarray(mints.ao_eri()))


#for i in range(100):
#    print(G.flatten()[i])
#    print(psi_G.flatten()[i])
print(np.allclose(G,psi_G))

