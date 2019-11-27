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

    for i in range(nshells):
        c1, exp1, atom1_idx, am1, idx1, size1 = onp.asarray(basis_dict[i]['coef']), onp.asarray(basis_dict[i]['exp']), basis_dict[i]['atom'], basis_dict[i]['am'], basis_dict[i]['idx'], basis_dict[i]['idx_stride']
        coeffs.append(c1)
        exps.append(exp1)
        atoms.append(atom1_idx)
        ams.append(am1)


    ## Each of these structures has shape (4, shell_quartets, max number of primitives in contraction)
    #coeffs = onp.asarray([ci, cj, ck, cl])
    #exps = onp.asarray([expi, expj, expk, expl])
    ## atoms has shape (4, shell_quartets)
    #atoms = onp.asarray([atomi, atomj, atomk, atoml])
    ## am has shape (4, shell_quartets)
    #am = onp.asarray([ami, amj, amk, aml])
    return np.asarray(coeffs), np.asarray(exps), np.asarray(atoms), np.asarray(ams)

        
coeffs, exps, atoms, am = preprocess(shell_quartets, basis_dict)
print("coeffs", coeffs.shape)
print("exps", exps.shape)
print("atoms", atoms.shape)
print("am", am.shape)
print(am)

#print(exps[1])

#print(cartesian_product(exps[1],exps[1],exps[0],exps[0]))
#print(am)
#print('angular momentum of the 5th shell quartet')
#print(np.array([am[0,5], am[1,5], am[2,5], am[3,5]]))


def compute(geom, coeffs, exps, atoms, am):
    with loops.Scope() as s:
        def primitive(A, B, C, D, aa, bb, cc, dd, coeff, am):
            '''Geometry parameters, exponents, coefficients, angular momentum identifier'''
            args = (A, B, C, D, aa, bb, cc, dd, coeff) 
            primitive = np.where((np.any((aa,bb,cc,dd)) == 0), 0.0, eri_ssss(*args))
            return primitive
    
        #    primitive =  np.where(e1 ==  0, 0.0,
        #                 np.where(am ==  0, overlap_ss(*args), 0.0))
        #    return primitive
        # Computes multiple primitive ss overlaps with same center, angular momentum 
        vectorized_primitive = jax.vmap(primitive, (None,None,None,None,0,0,0,0,0,None))

        ## Computes a contracted ss overlap 
        #@jax.jit
        def contraction(A, B, C, D, aa, bb, cc, dd, coeff, am):
            primitives = vectorized_primitive(A, B, C, D, aa, bb, cc, dd, coeff, am)
            return np.sum(primitives)


        # Just collect 1d arrays for each shell's coefficient, exponent, am, index, size
        # Wouldh avet ocall a cartesian product within the loop
        # The indices would need to be padded i believe in order to get them into an array
        # or just compute indices in the loop? 

        #TEMP until you figure out index shiz
        s.G = np.zeros((nshells,nshells,nshells,nshells))
        #s.G = np.zeros((nbf,nbf,nbf,nbf))
        for i in s.range(nshells):
            A = geom[atoms[i]]
            aa = exps[i]
            c1 = coeffs[i]
            ami = am[i]
            for j in s.range(nshells):
                B = geom[atoms[j]]
                bb = exps[j]
                c2 = coeffs[j]
                amj = am[j]
                for k in s.range(nshells):
                    C = geom[atoms[k]]
                    cc = exps[k]
                    c3 = coeffs[k]
                    amk = am[k]
                    for l in s.range(nshells):
                        D = geom[atoms[l]]
                        dd = exps[l]
                        c4 = coeffs[l]
                        aml = am[l]

                        #TODO dummy computation
                        exp_combos = cartesian_product(aa,bb,cc,dd)
                        coeff_combos = np.prod(cartesian_product(c1,c2,c3,c4), axis=1)
                        am_vec = np.array([ami, amj, amk, aml]) 
                        val = contraction(A,B,C,D, 
                                          exp_combos[:,0], 
                                          exp_combos[:,1], 
                                          exp_combos[:,2],
                                          exp_combos[:,3],
                                          coeff_combos, am_vec)
                        s.G = jax.ops.index_update(s.G, jax.ops.index[i,j,k,l], val)


                        #val = np.sum(A) * np.sum(B) * np.sum(C) * np.sum(D)
                        #s.G = jax.ops.index_update(s.G, jax.ops.index[i,j,k,l], val)

        return s.G


G = compute(geom, coeffs, exps, atoms, am)
#print(G)




