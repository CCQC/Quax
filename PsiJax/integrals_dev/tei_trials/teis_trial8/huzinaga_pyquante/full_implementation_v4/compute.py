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
from tei import contracted_tei

# Define molecule
molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         units bohr
                         """)
# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))

basis_name = 'sto-3g'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
# Homogenize the basis set dictionary
max_prim = basis_set.max_nprimitive()
max_am = basis_set.max_am()
#biggest_K = max_prim**4
nbf = basis_set.nbf()
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
        current_exp = np.asarray(basis_dict[i]['exp'])
        new_dict[i]['exp'] = np.asarray(np.pad(current_exp, (0, max_prim - current_exp.shape[0]), constant_values=1))
        current_coef = np.asarray(basis_dict[i]['coef'])
        new_dict[i]['coef'] = np.asarray(np.pad(current_coef, (0, max_prim - current_coef.shape[0])))
    return new_dict

basis_dict = homogenize_basisdict(basis_dict, max_prim)

def experiment(geom, basis):
    nshells = len(basis)
    shell_quartets = cartesian_product(np.arange(nshells), np.arange(nshells), np.arange(nshells), np.arange(nshells))
    # Instead of using a dictionary object, just construct an (nbf, k) array for each thing: coeff, exp, atom, am, idx, idx_stride
    # This can be done from homogenize_basisdict padding to make all contractions the same size
    coeffs = []
    exps = []
    atoms = []
    ams = []
    indices = []
    dims = []
    for i in range(nshells):
        coeffs.append(basis[i]['coef'])
        exps.append(basis[i]['exp'])
        atoms.append(basis[i]['atom'])
        ams.append(basis[i]['am'])
        indices.append(basis[i]['idx'])
        dims.append(basis[i]['idx_stride'])

    coeffs = np.array(coeffs)
    exps = np.array(exps) 
    atoms = np.array(atoms)
    ams = np.array(ams)
    indices = np.array(indices)
    dims = np.array(dims)

    angular_momentum = np.array([[0,0,0], 
                                 [1,0,0],                             
                                 [0,1,0],
                                 [0,0,1],
                                 [2,0,0],
                                 [1,1,0],
                                 [1,0,1],
                                 [0,2,0],
                                 [0,1,1],
                                 [0,0,2]])

    leading_indices = np.array([0,1,4])

    #TODO jaxify 
    G = onp.zeros((nbf,nbf,nbf,nbf))
    # Loop over quartets of basis functions (shell quartets)
    for quartet in shell_quartets:
      # can decide whether to evalute here (uniqueness, magnitude) 
      b1,b2,b3,b4 = quartet
      c1, c2, c3, c4 = coeffs[b1], coeffs[b2], coeffs[b3], coeffs[b4]
      aa, bb, cc, dd = exps[b1], exps[b2], exps[b3], exps[b4]
      atom1, atom2, atom3, atom4 = atoms[b1], atoms[b2], atoms[b3], atoms[b4]
      A, B, C, D = geom[atom1], geom[atom2], geom[atom3], geom[atom4]
      am1,am2,am3,am4 = ams[b1], ams[b2], ams[b3], ams[b4]
      ld1, ld2, ld3, ld4 = leading_indices[am1],leading_indices[am2],leading_indices[am3],leading_indices[am4]
      #ld1,ld2,ld3,ld4 = get_am_indx(am1), get_am_indx(am2), get_am_indx(am3), get_am_indx(am4)
      # Loop over angular momentum distribution
      for d1 in range(dims[b1]):
        for d2 in range(dims[b2]):
          for d3 in range(dims[b3]):
            for d4 in range(dims[b4]):
              La = angular_momentum[d1 + ld1]
              Lb = angular_momentum[d2 + ld2]
              Lc = angular_momentum[d3 + ld3]
              Ld = angular_momentum[d4 + ld4]
              tei = contracted_tei(La,Lb,Lc,Ld,A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)
              # add to appropriate index in G
              i = indices[b1] + d1
              j = indices[b2] + d2
              k = indices[b3] + d3
              l = indices[b4] + d4
              print([i,j,k,l])
              G[i,j,k,l] = tei
    return G

G = experiment(geom, basis_dict)

mints = psi4.core.MintsHelper(basis_set)
psi_G = np.asarray(onp.asarray(mints.ao_eri()))
print(np.allclose(G, psi_G))
#print(G)
print(psi_G)


