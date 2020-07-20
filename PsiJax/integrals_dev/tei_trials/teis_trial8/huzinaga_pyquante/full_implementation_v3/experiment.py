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
# Homogenize the basis set dictionary
max_prim = basis_set.max_nprimitive()
max_am = basis_set.max_am()
#biggest_K = max_prim**4
#nbf = basis_set.nbf()
#nshells = len(basis_dict)
#max_size = (max_am + 1) * (max_am + 2) // 2
#
#shell_quartets = old_cartesian_product(onp.arange(nshells), onp.arange(nshells), onp.arange(nshells), onp.arange(nshells))
#
#

def homogenize_basisdict(basis_dict, max_prim):
    '''
    Make it so all contractions are the same size in the basis dict by padding exp and coef values 
    to 1 and 0. Also create 'indices' key which says where along axis the integral should go, 
    but padded with -1's to maximum angular momentum size
    This allows you to pack them neatly into an array, and then worry about redundant computation later
    '''
    new_dict = basis_dict.copy()
    for i in range(len(basis_dict)):
        # original version
        #current_exp = np.asarray(basis_dict[i]['exp'])
        #new_dict[i]['exp'] = np.asarray(np.pad(current_exp, (0, max_prim - current_exp.shape[0]), constant_values=1))
        #current_coef = np.asarray(basis_dict[i]['coef'])
        #new_dict[i]['coef'] = np.asarray(np.pad(current_coef, (0, max_prim - current_coef.shape[0])))

        # version without padding 
        new_dict[i]['coef'] = np.asarray(basis_dict[i]['coef'])  
        new_dict[i]['exp'] = np.asarray(basis_dict[i]['exp'])
    return new_dict


basis_dict = homogenize_basisdict(basis_dict, max_prim)

dummy_dict = {0 : np.array([0.1,0.2,0.3]), 1: np.array([0.1,0.2,0.3])}
print(dummy_dict)

# huh this works... can you skip all the preprocess nonsense altotehter? avoid the padding ?
from aux_experiment import contracted_tei

def experiment(geom, basis):
    nshells = len(basis)
    shell_quartets = cartesian_product(np.arange(nshells), np.arange(nshells), np.arange(nshells), np.arange(nshells))

    # build G... cant i just map over the shell quartets?

    # jax doesnt seem to like dictionary lookup in loop
    #TODO does it work if you remove strings from dictionary? use ints? idk
    # TODO dictionary does not work in jax loops, foriloop
    # TODO perhaps a standard pytree? however one issue says pytrees do not work
    # TODO still not clear why jitting doesnt cause issues, but trying to map the loop does.
    with loops.Scope() as s:
        s.test = 0.
        
        #for quartet in s.range(shell_quartets.shape[0]):
        for quartet in s.range(5):
            i,j,k,l = shell_quartets[quartet]
            c1, aa = dummy_dict[i][0], dummy_dict[i][1]
            #c1, aa, atom1, am1, idx1, size1 = basis[i]['coef'],  basis[i]['exp'], basis[i]['atom'], basis[i]['am'], basis[i]['idx'], basis[i]['idx_stride']
            #c2, bb, atom2, am2, idx2, size2 = basis[j]['coef'], basis[j]['exp'], basis[j]['atom'], basis[j]['am'], basis[j]['idx'], basis[j]['idx_stride']
            #c3, cc, atom3, am3, idx3, size3 = basis[k]['coef'], basis[k]['exp'], basis[k]['atom'], basis[k]['am'], basis[k]['idx'], basis[k]['idx_stride']
            #c4, dd, atom4, am4, idx4, size4 = basis[l]['coef'], basis[l]['exp'], basis[l]['atom'], basis[l]['am'], basis[l]['idx'], basis[l]['idx_stride']
            # This is just a dummy am, in the future would have to go through all possibilities 
            L = (am1,0,0,am2,0,0,am3,0,0,am4,0,0)
            A = geom[atom1]
            B = geom[atom2]
            C = geom[atom3]
            D = geom[atom4]
            s.test += contracted_tei(L,A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)


    #for i in range(5):
    #    i,j,k,l = 1,2,0,1

    #for quartet in shell_quartets:

    # this works
    #for i in range(5):
    #    i,j,k,l = shell_quartets[i]
    #    c1, aa, atom1, am1, idx1, size1 = basis[i]['coef'],  basis[i]['exp'], basis[i]['atom'], basis[i]['am'], basis[i]['idx'], basis[i]['idx_stride']
    #    c2, bb, atom2, am2, idx2, size2 = basis[j]['coef'], basis[j]['exp'], basis[j]['atom'], basis[j]['am'], basis[j]['idx'], basis[j]['idx_stride']
    #    c3, cc, atom3, am3, idx3, size3 = basis[k]['coef'], basis[k]['exp'], basis[k]['atom'], basis[k]['am'], basis[k]['idx'], basis[k]['idx_stride']
    #    c4, dd, atom4, am4, idx4, size4 = basis[l]['coef'], basis[l]['exp'], basis[l]['atom'], basis[l]['am'], basis[l]['idx'], basis[l]['idx_stride']
    #    # This is just a dummy am, in the future would have to go through all possibilities 
    #    L = (am1,0,0,am2,0,0,am3,0,0,am4,0,0)
    #    A = geom[atom1]
    #    B = geom[atom2]
    #    C = geom[atom3]
    #    D = geom[atom4]
    #    test += contracted_tei(L,A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)
    return test 


print(experiment(geom, basis_dict))
print(experiment(geom, basis_dict))


