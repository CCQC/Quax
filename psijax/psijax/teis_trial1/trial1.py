import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product, find_unique_shells

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
    centers = [] # A list of atom indices
    am = [] # list of string identifiers '1010' etc
    for quartet in unique_shell_quartets:
        i,j,k,l = quartet
        c1, exp1, atom1_idx, am1 = onp.asarray(basis_dict[i]['coef']), onp.asarray(basis_dict[i]['exp']), basis_dict[i]['atom'], basis_dict[i]['am'] 
        c2, exp2, atom2_idx, am2 = onp.asarray(basis_dict[j]['coef']), onp.asarray(basis_dict[j]['exp']), basis_dict[j]['atom'], basis_dict[j]['am']
        c3, exp3, atom3_idx, am3 = onp.asarray(basis_dict[k]['coef']), onp.asarray(basis_dict[k]['exp']), basis_dict[k]['atom'], basis_dict[k]['am']
        c4, exp4, atom4_idx, am4 = onp.asarray(basis_dict[l]['coef']), onp.asarray(basis_dict[l]['exp']), basis_dict[l]['atom'], basis_dict[l]['am']
    
        # Compute all primitive combinations of exponents and contraction coefficients for this shell 
        # Each column is i, j, k, l
        exp_combos = old_cartesian_product(exp1,exp2,exp3,exp4)
        coeff_combos = old_cartesian_product(c1,c2,c3,c4)

        # For every primitive, gather data necessary for the ERI shell computation. 
        # Lots of redundnacy here, since each primitive has same geometry. But this is the only way to get things into arrays. 
        # could sort them by contraction size and compute each one at a time, or pad them to make them the same size
        #print(exp_combos.shape[0])
        am_str = str(am1) + str(am2) + str(am3) + str(am4)
        for contraction in range(exp_combos.shape[0]):
            basis_data.append([exp_combos[contraction,0],
                               exp_combos[contraction,1],
                               exp_combos[contraction,2],
                               exp_combos[contraction,3],
                               coeff_combos[contraction,0], 
                               coeff_combos[contraction,1], 
                               coeff_combos[contraction,2], 
                               coeff_combos[contraction,3]])
            centers.append([atom1_idx, atom2_idx, atom3_idx, atom4_idx])
            am.append(am_str)

    return np.asarray(onp.asarray(basis_data)), centers, am 

basis_data, centers, am  = preprocess(unique_shell_quartets, basis_dict)
print(basis_data.shape)

np.take(geom, 


def build_tei(geom, centers, basis_data, am):
    return 0 



# jit with static argnums on am
# note: will recompile everytime it changes, I don't *think* it caches the different cases
# can raise an issue to request this as an option, would make life super easy. 
#@def test( am):
#@    
#@    if am == 'ssss':
#@
