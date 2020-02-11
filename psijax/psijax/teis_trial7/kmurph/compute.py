import psi4
import jax 
from jax import lax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from jax.experimental import loops
from integrals_utils import primitive_eri, np_cartesian_product
from pprint import pprint
np.set_printoptions(linewidth=300)

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

max_prim = basis_set.max_nprimitive()
max_am = basis_set.max_am()
biggest_K = max_prim**4
nbf = basis_set.nbf()
nshells = len(basis_dict)
max_size = (max_am + 1) * (max_am + 2) // 2
shell_quartets = np_cartesian_product(np.arange(nshells), np.arange(nshells), np.arange(nshells), np.arange(nshells))
print("Number of basis functions: ",nbf)
print("Number of shells: ", nshells)
print("Number of shell quartets (redundant): ", shell_quartets.shape[0])
print("Max angular momentum: ", max_am)
print("Largest number of primitives: ", max_prim)
print("Largest contraction: ", biggest_K)
pprint(basis_dict)

def am_vectors(am, length=3):
    '''
    Builds up all possible angular momentum component vectors of with total angular momentum 'am'
    am = 2 ---> [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    Returns a generator which must be converted to an iterable,
    for example, call the following: [list(i) for i in am_vectors(2)]

    Works by building up each possibility :
    For a given value in reversed(range(am+1)), find all other possible values for other entries in length 3 vector
     value     am_vectors(am-value,length-1)    (value,) + permutation
       2 --->         [0,0]                 ---> [2,0,0] ---> dxx
       1 --->         [1,0]                 ---> [1,1,0] ---> dxy
         --->         [0,1]                 ---> [1,0,1] ---> dxz
       0 --->         [2,0]                 ---> [0,2,0] ---> dyy
         --->         [1,1]                 ---> [0,1,1] ---> dyz
         --->         [0,2]                 ---> [0,0,2] ---> dzz
    '''
    if length == 1:
        yield (am,)
    else:
        # reverse so angular momentum order is canonical, e.g., dxx dxy dxz dyy dyz dzz
        for value in reversed(range(am + 1)): 
            for permutation in am_vectors(am - value,length - 1):
                yield (value,) + permutation

print(np.array([list(i) for i in am_vectors(2)]))

#def expand_basisdict(basis_dict):
#    '''
#    Expand basis functions in dictionary 
#    '''

#am = 0
#print(int(0.5 * (am + 1) * (am + 2)))

print(np_cartesian_product(np.array([0,1,2,3,4,5]), np.array([0]),np.array([0,1,2,3,4,5]),np.array([0])))

def preprocess(shell_quartets, basis):
    exps = []                
    coeffs = []
    centers = []
    repeats = []
    total_am = []                 
    for quartet in shell_quartets:
        # Construct data rows all primitives in this shell quartet
        i,j,k,l = quartet
        c1, aa, atom1, am1, size1 = onp.asarray(basis[i]['coef']), onp.asarray(basis[i]['exp']), basis[i]['atom'], basis[i]['am'], basis[i]['idx_stride']
        c2, bb, atom2, am2, size2 = onp.asarray(basis[j]['coef']), onp.asarray(basis[j]['exp']), basis[j]['atom'], basis[j]['am'], basis[j]['idx_stride']
        c3, cc, atom3, am3, size3 = onp.asarray(basis[k]['coef']), onp.asarray(basis[k]['exp']), basis[k]['atom'], basis[k]['am'], basis[k]['idx_stride']
        c4, dd, atom4, am4, size4 = onp.asarray(basis[l]['coef']), onp.asarray(basis[l]['exp']), basis[l]['atom'], basis[l]['am'], basis[l]['idx_stride']

        am_vec1 = onp.array([list(i) for i in am_vectors(am1)])
        am_vec2 = onp.array([list(i) for i in am_vectors(am2)])
        am_vec3 = onp.array([list(i) for i in am_vectors(am3)])
        am_vec4 = onp.array([list(i) for i in am_vectors(am4)])

        tmp_indices = np_cartesian_product(np.arange(size1),np.arange(size2),np.arange(size3),np.arange(size4))
        all_am = np.hstack((am_vec1[tmp_indices[:,0]],am_vec2[tmp_indices[:,1]],am_vec3[tmp_indices[:,2]],am_vec4[tmp_indices[:,3]]))
        
        # Create exp, coeff arrays and pad to same size (largest contraction) so they can be put into an array
        # THIS REALLY ISN'T NECESSARY, JUST SIMPLER. YOU COULD IN THEORY JUST PASS THROUGH EACH CONTRACTION SIZE IN BATCHES WHEN COMPUTING (multiple vmap function evaluations)
        exp_combos = np_cartesian_product(aa,bb,cc,dd)  # of size (K, 4)
        coeff_combos = onp.prod(np_cartesian_product(c1,c2,c3,c4), axis=1)  # of size K
        K = exp_combos.shape[0]
        exps_padded = onp.pad(exp_combos, ((0, biggest_K - K), (0,0)))
        coeffs_padded = onp.pad(coeff_combos, (0, biggest_K - K))
        # Need to collect the 'repeats' arg for np.repeat after the loop  such that they are (nbf**4, K, 4) and (nbf**4, K, 1)
        exps.append(exps_padded)
        coeffs.append(coeffs_padded)
        repeats.append(all_am.shape[0])
        centers.append([atom1,atom2,atom3,atom4])                                          
        total_am.append(all_am)

    #print(tmp_indices.shape)
    #print(tmp_indices)

    exps = onp.asarray(exps)
    coeffs = onp.asarray(coeffs)
    repeats = onp.asarray(repeats)
    exps_final = onp.repeat(exps, repeats, axis=0)
    coeffs_final = onp.repeat(coeffs, repeats, axis=0)
    am_final = onp.vstack(total_am)

    print(exps_final.shape)
    print(coeffs_final.shape)
    print(am_final.shape)

    #TODO deal with centers, convert to array then repeat? NO NEED FOR LISTS  look at previous implementation
    print(centers)




preprocess(shell_quartets,basis_dict)

