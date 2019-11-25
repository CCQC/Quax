import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product, find_unique_shells
from functools import partial

from eri import *

np.set_printoptions(linewidth=800, suppress=True, threshold=100)


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
nbf = basis_set.nbf()
nshells = len(basis_dict)
unique_shell_quartets = find_unique_shells(nshells)
print("number of basis functions", nbf)
print("number of shells ", nshells)
print("number of shell quartets", nshells**4)
print("number of unique shell quartets", unique_shell_quartets.shape[0])

def preprocess(unique_shell_quartets, basis_dict):
    basis_data = [] # A list of every primitive exponent, coeff of 4 centers
    centers = []
    angular_momenta = [] # list of string identifiers '1010' etc
    for quartet in unique_shell_quartets:
        i,j,k,l = quartet
        c1, exp1, atom1_idx, am1 = onp.asarray(basis_dict[i]['coef']), onp.asarray(basis_dict[i]['exp']), basis_dict[i]['atom'], basis_dict[i]['am'] 
        c2, exp2, atom2_idx, am2 = onp.asarray(basis_dict[j]['coef']), onp.asarray(basis_dict[j]['exp']), basis_dict[j]['atom'], basis_dict[j]['am']
        c3, exp3, atom3_idx, am3 = onp.asarray(basis_dict[k]['coef']), onp.asarray(basis_dict[k]['exp']), basis_dict[k]['atom'], basis_dict[k]['am']
        c4, exp4, atom4_idx, am4 = onp.asarray(basis_dict[l]['coef']), onp.asarray(basis_dict[l]['exp']), basis_dict[l]['atom'], basis_dict[l]['am']
    
        # Compute all primitive combinations of exponents and contraction coefficients for this shell 
        # Each column is i, j, k, l
        exp_combos = old_cartesian_product(exp1,exp2,exp3,exp4)
        # fuse the already fused normalization/ contraction coefficients together
        coeff_combos = onp.prod(old_cartesian_product(c1,c2,c3,c4), axis=1)

        # For every primitive, gather data necessary for the ERI shell computation. 
        # Lots of redundnacy here, since each primitive has same geometry. But this is the only way to get things into arrays. 
        # could sort them by contraction size and compute each one at a time, or pad them to make them the same size

        # TODO make function for permuting exponent and center indice data into canonical function forms psss psps ppss ppps pppp
        # have it return exponent list, center index list, have it take in
        am = onp.array([am1,am2,am3,am4])
        atom_idx_list = [atom1_idx,atom2_idx,atom3_idx,atom4_idx]


        for contraction in range(exp_combos.shape[0]):
            basis_data.append([exp_combos[contraction,0],
                               exp_combos[contraction,1],
                               exp_combos[contraction,2],
                               exp_combos[contraction,3],
                               coeff_combos[contraction]])
            centers.append(atom_idx_list)
            angular_momenta.append(am)
    return np.asarray(onp.asarray(basis_data)), centers, np.asarray(onp.asarray(angular_momenta))

#basis_data, centers, am  = preprocess(unique_shell_quartets, basis_dict)
    # Try passing redundant shells
basis_data, centers, angular_momenta  = preprocess(unique_shell_quartets, basis_dict)

final_centers = np.take(geom, centers, axis=0)


#def build_tei(geom, centers, basis_data, am):
    #G = np.zeros_like((nbf,nbf,nbf,nbf)) # how do i even generate a vector of size nunique?
    # leading axis is number of primitives
#    centers = np.take(geom, centers, axis=0)

#    return 0 

# This does not work since it is jit compiling based on index i,which is abstract, and the static_argnum 
#@partial(jax.jit, static_argnums=(2,))
def general(basis_data, centers, am):  
    A, B, C, D = centers
    aa, bb, cc, dd, coeff = basis_data
    args = (A, B, C, D, aa, bb, cc, dd, coeff)

    #print(am, end= ' ')
    #if tuple(am) == (0,0,0,0):
    #    val = eri_ssss(*args)
    #if tuple(am) == (1,0,0,0):
    #    val = eri_psss(*args).reshape(-1)
    #if tuple(am) == (1,0,1,0):
    #    val = eri_psps(*args).reshape(-1)
    #if tuple(am) == (1,1,0,0):
    #    val = eri_ppss(*args).reshape(-1)
    #if tuple(am) == (1,1,1,0):
    #    val = eri_ppps(*args).reshape(-1)
    #if tuple(am) == (1,1,1,1):
    #    val = eri_pppp(*args).reshape(-1)

    if np.allclose(am,np.array([0,0,0,0])):
        val = eri_ssss(*args)
    elif np.allclose(am,np.array([1,0,0,0])):
        val = eri_psss(*args).reshape(-1)
    elif np.allclose(am,np.array([1,0,1,0])):
        val = eri_psps(*args).reshape(-1)
    elif np.allclose(am,np.array([1,1,0,0])):
        val = eri_ppss(*args).reshape(-1)
    elif np.allclose(am,np.array([1,1,1,0])):
        val = eri_ppps(*args).reshape(-1)
    elif np.allclose(am,np.array([1,1,1,1])):
        val = eri_pppp(*args).reshape(-1)
    # TODO fix redundancies such as 1000 0100 0010 0001
    else:
        val = eri_pppp(*args).reshape(-1)
    print(val)
    return val

#new_general = jax.jit(general, static_argnums=(2,))

print(np.array([0,0,0,0]))
print(tuple(np.array([0,0,0,0])))

for i in range(1296):
#    print(angular_momenta[i], end=' ')
    print(general(basis_data[i], final_centers[i], angular_momenta[i]))
#    new_general(basis_data[i], final_centers[i], np.array([0,0,0,0]))
#    print('integral result is', general(basis_data, final_centers, am, i))

# jit with static argnums on am
# note: will recompile everytime it changes, I don't *think* it caches the different cases
# can raise an issue to request this as an option, would make life super easy. 
# Can take care of ordering with a simple argsort so that all of the same class appear at the same time.
#def test( am):
#    ssss = np.array([0,0,0,0])
#    
#    if am == 'ssss':
#    if am == [0,0,0,0]:
#    if am == ssss:  # least array instantiation
#
