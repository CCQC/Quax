import psi4
import jax.numpy as np
import jax
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product, find_unique_shells, am_vectors
from functools import partial
from jax.experimental import loops
from pprint import pprint
from tei import tei
np.set_printoptions(linewidth=500,edgeitems=10)

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

shell_quartets = old_cartesian_product(onp.arange(nshells), onp.arange(nshells), onp.arange(nshells), onp.arange(nshells))
print("Number of basis functions: ",nbf)
print("Number of shells: ", nshells)
print("Number of redundant shell quartets: ", shell_quartets.shape[0])
print("Max angular momentum: ", max_am)
print("Largest number of primitives: ", max_prim)
print("Largest contraction: ", biggest_K)


#def preprocess(shell_quartets, basis):
#    """Args: shell quartet indices, basis dictionary"""
#    exps = []
#    coeffs = []
#    centers = []
#    ams = []
#    all_indices = []
#    for quartet in shell_quartets:
#        i,j,k,l = quartet
#        #basis_dict[i]['am'], basis_dict[i]['idx'], basis_dict[i]['idx_stride']
#        c1, aa, atom1, am1, idx1, size1 = onp.asarray(basis[i]['coef']), onp.asarray(basis[i]['exp']), basis[i]['atom'], basis[i]['am'], basis[i]['idx'], basis[i]['idx_stride']
#        c2, bb, atom2, am2, idx2, size2 = onp.asarray(basis[j]['coef']), onp.asarray(basis[j]['exp']), basis[j]['atom'], basis[j]['am'], basis[j]['idx'], basis[j]['idx_stride']
#        c3, cc, atom3, am3, idx3, size3 = onp.asarray(basis[k]['coef']), onp.asarray(basis[k]['exp']), basis[k]['atom'], basis[k]['am'], basis[k]['idx'], basis[k]['idx_stride']
#        c4, dd, atom4, am4, idx4, size4 = onp.asarray(basis[l]['coef']), onp.asarray(basis[l]['exp']), basis[l]['atom'], basis[l]['am'], basis[l]['idx'], basis[l]['idx_stride']
#
#        exp_combos = old_cartesian_product(aa,bb,cc,dd)
#        coeff_combos = onp.prod(old_cartesian_product(c1,c2,c3,c4), axis=1)
#        am_vec = onp.array([am1, am2, am3, am4])
#
#        indices1 = onp.repeat(idx1, size1) + onp.arange(size1)
#        indices2 = onp.repeat(idx2, size2) + onp.arange(size2)
#        indices3 = onp.repeat(idx3, size3) + onp.arange(size3)
#        indices4 = onp.repeat(idx4, size4) + onp.arange(size4)
#        indices = old_cartesian_product(indices1,indices2,indices3,indices4)
#        indices = onp.pad(indices, ((0, 81-indices.shape[0]),(0,0)), constant_values=-1)
#        all_indices.append(indices)
#
#        # Pad exp, coeff arrays to same size (largest contraction) so they can be put into an array
#        K = exp_combos.shape[0]
#        exps.append(onp.pad(exp_combos, ((0, biggest_K - K), (0,0))))
#        coeffs.append(onp.pad(coeff_combos, (0, biggest_K - K)))
#        centers.append([atom1,atom2,atom3,atom4])
#        ams.append(am_vec)
#
#    exps = onp.asarray(exps)
#    coeffs = onp.asarray(coeffs)
#    centers = onp.asarray(centers)
#    am = onp.asarray(ams)
#    all_indices = onp.asarray(all_indices)
#    return np.asarray(exps), np.asarray(coeffs), centers, np.asarray(am), np.asarray(all_indices)

#print(exps.shape)
#print(coeffs.shape)
#print(centers.shape)
#print(am.shape)

def preprocess(shell_quartets, basis):
    '''
    Note: very inefficient. Would be better to generate shell-quartet data, then expand it as needed on the fly
    '''
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

        tmp_indices = old_cartesian_product(onp.arange(size1),onp.arange(size2),onp.arange(size3),onp.arange(size4))
        # This is of shape (size of quartet class, 12). For instance, (pp|pp) is (81,12)
        all_am = onp.hstack((am_vec1[tmp_indices[:,0]],am_vec2[tmp_indices[:,1]],am_vec3[tmp_indices[:,2]],am_vec4[tmp_indices[:,3]]))
        #print(all_am.shape)

        # Create exp, coeff arrays and pad to same size (largest contraction) so they can be put into an array
        # THIS REALLY ISN'T NECESSARY, JUST SIMPLER. YOU COULD IN THEORY JUST PASS THROUGH EACH CONTRACTION SIZE IN BATCHES WHEN COMPUTING (multiple vmap function evaluations)
        exp_combos = old_cartesian_product(aa,bb,cc,dd)  # of size (K, 4)
        coeff_combos = onp.prod(old_cartesian_product(c1,c2,c3,c4), axis=1)  # of size K
        K = exp_combos.shape[0]
        exps_padded = onp.pad(exp_combos, ((0, biggest_K - K), (0,0)), constant_values=1.0)
        coeffs_padded = onp.pad(coeff_combos, (0, biggest_K - K), constant_values=0.0)
        #ALTERNATIVE: just pad the basis dict to even contraction, then push the cartesian product into the function which evaluastes the integrals
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
    centers = onp.asarray(centers)
    repeats = onp.asarray(repeats)
    exps_final = onp.repeat(exps, repeats, axis=0)
    coeffs_final = onp.repeat(coeffs, repeats, axis=0)
    centers_final = onp.repeat(centers, repeats, axis=0)
    am_final = onp.vstack(total_am)
    print('exps shape',exps_final.shape)
    print('coeffs shape',coeffs_final.shape)
    print('am shape',am_final.shape)
    print('centers shape', centers_final.shape)
    return np.asarray(am_final), np.asarray(exps_final), np.asarray(centers_final), np.asarray(coeffs_final)

#am, exps, centers, coeffs = preprocess(shell_quartets,basis_dict)

#print(am.shape)

# map over primitives and sum to form contracted integrals  
#vectorized_prim = jax.vmap(lambda L,RA,RB,RC,RD,a,b,c,d,contraction : 
#                          np.where(contraction > 0.0, tei(L,RA,RB,RC,RD,a,b,c,d,contraction), 0.0),
#                          (None,None,None,None,None,0,0,0,0,0))

vectorized_prim = jax.jit(jax.vmap(lambda L,RA,RB,RC,RD,a,b,c,d,contraction : 
                          np.where(contraction > 0.0, tei(L,RA,RB,RC,RD,a,b,c,d,contraction), 0.0),
                          (None,None,None,None,None,0,0,0,0,0)))

def contracted_eri(L, RA, RB, RC, RD, a,b,c,d,contraction):
    result = vectorized_prim(L, RA, RB, RC, RD, a,b,c,d,contraction)
    return np.sum(result)

all_tei = jax.vmap(contracted_eri,(0,0,0,0,0,0,0,0,0,0))
#all_tei = jax.jit(jax.vmap(contracted_eri,(0,0,0,0,0,0,0,0,0,0)))

def compute(geom, basis_dict, shell_quartets):
    am, exps, centers, coeffs = preprocess(shell_quartets,basis_dict)
    center_vectors = geom[centers] # Shape (nbf**4, 4,3)                  
    RA = center_vectors[:,0,:] # Shape (nbf**4, 3)                        
    RB = center_vectors[:,1,:] # Shape (nbf**4, 3)                        
    RC = center_vectors[:,2,:] # Shape (nbf**4, 3)                        
    RD = center_vectors[:,3,:] # Shape (nbf**4, 3)                        

    a = exps[:,:,0]                                           
    b = exps[:,:,1]                                           
    c = exps[:,:,2]                                           
    d = exps[:,:,3]                                           
    #print(a.shape)
    #coeff = coeffs[0,:]                                       
    #print(coeffs)

    ##contracted_tei 
    test = all_tei(am,RA,RB,RC,RD,a,b,c,d,coeffs)
    #print(test)
    #print(np.sum(test, axis=1))
    return test


test = compute(geom, basis_dict, shell_quartets)

# TODO remove
test = compute(geom, basis_dict, shell_quartets)


# This runs into the same error is nested while loop vmaps
#jax.jacfwd(compute)(geom, basis_dict, shell_quartets)

