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
pprint(basis_dict)
# Homogenize the basis set dictionary
max_prim = basis_set.max_nprimitive()
max_am = basis_set.max_am()
nprim = basis_set.nprimitive()
nbf = basis_set.nbf()
print("Number of basis functions: ", nbf)
print("Number of primitives: ", nprim)

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

    # Only goes up to f functions, to add more use shell iterator
    angular_momentum = np.array([[0,0,0], 
                                 [1,0,0],                             
                                 [0,1,0],
                                 [0,0,1],
                                 [2,0,0],
                                 [1,1,0],
                                 [1,0,1],
                                 [0,2,0],
                                 [0,1,1],
                                 [0,0,2], 
                                 [3,0,0],
                                 [2,1,0],
                                 [2,0,1],
                                 [1,2,0],
                                 [1,1,1],
                                 [1,0,2],
                                 [0,3,0],
                                 [0,2,1],
                                 [0,1,2],
                                 [0,0,3]])

    leading_indices = np.array([0,1,4,10])

    vmapped_contracted_tei = jax.vmap(contracted_tei, (0,None,None,None,None,None,None,None,None,None,None,None,None))

    with loops.Scope() as s:
      s.G = np.zeros((nbf,nbf,nbf,nbf))

      s.a = 0  # center A angular momentum iterator 
      s.b = 0  # center B angular momentum iterator 
      s.c = 0  # center C angular momentum iterator 
      s.d = 0  # center D angular momentum iterator 
      # Loop over quartets of basis functions (shell quartets)
      for q in s.range(shell_quartets.shape[0]):
        # can decide whether to evalute here (uniqueness, magnitude) 
        b1,b2,b3,b4 = shell_quartets[q]
        c1, c2, c3, c4 = coeffs[b1], coeffs[b2], coeffs[b3], coeffs[b4]
        aa, bb, cc, dd = exps[b1], exps[b2], exps[b3], exps[b4]
        atom1, atom2, atom3, atom4 = atoms[b1], atoms[b2], atoms[b3], atoms[b4]
        A, B, C, D = geom[atom1], geom[atom2], geom[atom3], geom[atom4]
        am1,am2,am3,am4 = ams[b1], ams[b2], ams[b3], ams[b4]
        ld1, ld2, ld3, ld4 = leading_indices[am1],leading_indices[am2],leading_indices[am3],leading_indices[am4]
        # Loop over angular momentum distributions
        # Collect angular momentum combinations and corresponding indices in fixed-shape arrays. 
        # ONLY UP TO D functions for now

        L_cache = np.zeros((1296,12))

        # What if you pull contracted_tei computation out of the loop and use vmapped version?
        count = 0
        s.a = 0
        for _ in s.while_range(lambda: s.a < dims[b1]):
          s.b = 0
          for _ in s.while_range(lambda: s.b < dims[b2]):
            s.c = 0
            for _ in s.while_range(lambda: s.c < dims[b3]):
              s.d = 0
              for _ in s.while_range(lambda: s.d < dims[b4]):
                La = angular_momentum[s.a + ld1]
                Lb = angular_momentum[s.b + ld2]
                Lc = angular_momentum[s.c + ld3]
                Ld = angular_momentum[s.d + ld4]
                print(La)

                tmp = np.hstack((La,Lb,Lc,Ld))
                L_cache = jax.ops.index_update(L_cache, jax.ops.index[count,:], tmp) 
                count += 1

                #tei = contracted_tei(La,Lb,Lc,Ld,A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)
                tei=1
                # add to appropriate index in G
                i = indices[b1] + s.a
                j = indices[b2] + s.b
                k = indices[b3] + s.c
                l = indices[b4] + s.d
                s.G = jax.ops.index_update(s.G, jax.ops.index[i,j,k,l], tei) 
                s.d += 1
              s.c += 1
            s.b += 1
          s.a += 1

        cutoff = s.a * s.b * s.c * s.d

        # Gets mad at cutoff... hmm
        # Array slice indices must have static start/stop/step to be used with Numpy indexing syntax. Try dynamic slice?
        #teis = vmapped_contracted_tei(L_cache[:cutoff], A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)
        teis = vmapped_contracted_tei(L_cache, A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)

        #print(L_cache)

      return s.G

#G = experiment(geom, basis_dict)
##
#mints = psi4.core.MintsHelper(basis_set)
#psi_G = np.asarray(onp.asarray(mints.ao_eri()))
#print(np.allclose(G, psi_G))


