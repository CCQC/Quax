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
        current_exp = np.asarray(basis_dict[i]['exp'])
        new_dict[i]['exp'] = np.asarray(np.pad(current_exp, (0, max_prim - current_exp.shape[0]), constant_values=1))
        current_coef = np.asarray(basis_dict[i]['coef'])
        new_dict[i]['coef'] = np.asarray(np.pad(current_coef, (0, max_prim - current_coef.shape[0])))

        # version without padding 
        #new_dict[i]['coef'] = np.asarray(basis_dict[i]['coef'])  
        #new_dict[i]['exp'] = np.asarray(basis_dict[i]['exp'])
    return new_dict

basis_dict = homogenize_basisdict(basis_dict, max_prim)
pprint(basis_dict)

# huh this works... can you skip all the preprocess nonsense altotehter? avoid the padding ?
from aux_experiment import contracted_tei

def experiment(geom, basis):
    nshells = len(basis)
    shell_quartets = cartesian_product(np.arange(nshells), np.arange(nshells), np.arange(nshells), np.arange(nshells))
    
    # Instead of using a dictionary object, just construct an (nbf, k) array for each thing: coeff, exp, atom, am, idx, idx_stride
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
    exps = np.array(coeffs) 
    atoms = np.array(atoms)
    ams = np.array(ams)
    indices = np.array(indices)
    dims = np.array(dims)

    print(indices)
    print(dims)

    # First to get things straight in your head, write out loops in standard way
    # PSEUDOCODE
    # loop over all basis functions/ shells (this is normally done with cartesian product)
    #for b1 in range(nshells):
    #  for b2 in range(nshells):
    #    for b3 in range(nshells):
    #      for b4 in range(nshells):
    #        # loop over different angular momentum distributions
            # p would be loop size 3, d would be loop size 6
    for quartet in shell_quartets:
      b1,b2,b3,b4 = quartet
      am1,am2,am3,am4 = ams[b1], ams[b2], ams[b3], ams[b4]
      print(am1,am2,am3,am4)
      for d1 in range(dims[b1]):
        for d2 in range(dims[b2]):
          for d3 in range(dims[b3]):
            for d4 in range(dims[b4]):
              i = indices[b1] + d1
              j = indices[b2] + d2
              k = indices[b3] + d3
              l = indices[b4] + d4
                    #print(i,j,k,l)
                    # how to translate this into angular momentum vector?
                    #La = (


            # can decide whether to evalute here (uniqueness, magnitude) 
            # loop over contractions/exponents for different primitives 
            #for p1 in c1.shape
            #  for p2 in c2.shape
            #    for p3 in c3.shape
            #      for p4 in c4.shape

             #               G[i,j,k,l] +=
                    
            



    #for quartet in shell_quartets:
    #    i,j,k,l = quartet
    #    c1, c2, c3, c4 = coeffs[i], coeffs[j], coeffs[k], coeffs[l]
    #    aa, bb, cc, dd = exps[i], exps[j], exps[k], exps[l]

    def body(quartet, coeffs, exps, atoms, ams):
        test = 0.
        i,j,k,l = quartet
        c1, c2, c3, c4 = coeffs[i], coeffs[j], coeffs[k], coeffs[l]
        aa, bb, cc, dd = exps[i], exps[j], exps[k], exps[l]
        atom1, atom2, atom3, atom4 = atoms[i], atoms[j], atoms[k], atoms[l]
        am1, am2, am3, am4 = ams[i], ams[j], ams[k], ams[l]
        L = (am1,0,0,am2,0,0,am3,0,0,am4,0,0)
        A = geom[atom1]
        B = geom[atom2]
        C = geom[atom3]
        D = geom[atom4]
        test += contracted_tei(L,A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)
        return test

    # TODO reformulate with a jax.lax.scan, filling in G as needed, 
    # TODO can perhaps create angular momentum mechanism with more loops 
    #vmapped = jax.vmap(body, (0, None, None, None, None))
    #test = vmapped(shell_quartets, coeffs, exps, atoms, ams)

    #test = 0
    #for quartet in shell_quartets:
    #    i,j,k,l = quartet
    #    c1, aa, atom1, am1, idx1, size1 = basis[i]['coef'],  basis[i]['exp'], basis[i]['atom'], basis[i]['am'], basis[i]['idx'], basis[i]['idx_stride']
    #    c2, bb, atom2, am2, idx2, size2 = basis[j]['coef'], basis[j]['exp'], basis[j]['atom'], basis[j]['am'], basis[j]['idx'], basis[j]['idx_stride']
    #    c3, cc, atom3, am3, idx3, size3 = basis[k]['coef'], basis[k]['exp'], basis[k]['atom'], basis[k]['am'], basis[k]['idx'], basis[k]['idx_stride']
    #    c4, dd, atom4, am4, idx4, size4 = basis[l]['coef'], basis[l]['exp'], basis[l]['atom'], basis[l]['am'], basis[l]['idx'], basis[l]['idx_stride']
    #    L = (am1,0,0,am2,0,0,am3,0,0,am4,0,0)
    #    A = geom[atom1]
    #    B = geom[atom2]
    #    C = geom[atom3]
    #    D = geom[atom4]
    #    test += contracted_tei(L,A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)
    #    print(test)
        

    #@jax.jit
    #def body(geom, basis, shell_quartet): 
    #    test = 0.
    #    i,j,k,l = shell_quartet
    #    c1, aa, atom1, am1, idx1, size1 = basis[i]['coef'],  basis[i]['exp'], basis[i]['atom'], basis[i]['am'], basis[i]['idx'], basis[i]['idx_stride']
    #    c2, bb, atom2, am2, idx2, size2 = basis[j]['coef'], basis[j]['exp'], basis[j]['atom'], basis[j]['am'], basis[j]['idx'], basis[j]['idx_stride']
    #    c3, cc, atom3, am3, idx3, size3 = basis[k]['coef'], basis[k]['exp'], basis[k]['atom'], basis[k]['am'], basis[k]['idx'], basis[k]['idx_stride']
    #    c4, dd, atom4, am4, idx4, size4 = basis[l]['coef'], basis[l]['exp'], basis[l]['atom'], basis[l]['am'], basis[l]['idx'], basis[l]['idx_stride']
    #    L = (am1,0,0,am2,0,0,am3,0,0,am4,0,0)
    #    A = geom[atom1]
    #    B = geom[atom2]
    #    C = geom[atom3]
    #    D = geom[atom4]
    #    test += contracted_tei(L,A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)
    #    return test


    #arr = np.arange(5)
    #magic = (None, dummy_dict, 0) 
    #test = jax.vmap(body, magic)(geom, basis, shell_quartets)

    # Interesting note: this does not work . unhashable type: JaxprTracer
    #test = 0.
    #for quartet in shell_quartets:
    #    test += body(geom, basis, quartet)

    # This however does work:
    #test = 0.
    #for a in range(5):
    #    test += body(geom, basis, shell_quartets[a])

    # this works
    #test = 0.
    #for a in range(5):
    #    test += body(geom, basis, shell_quartets[a])

        #i,j,k,l = shell_quartets[a]
        #c1, aa, atom1, am1, idx1, size1 = basis[i]['coef'],  basis[i]['exp'], basis[i]['atom'], basis[i]['am'], basis[i]['idx'], basis[i]['idx_stride']
        #c2, bb, atom2, am2, idx2, size2 = basis[j]['coef'], basis[j]['exp'], basis[j]['atom'], basis[j]['am'], basis[j]['idx'], basis[j]['idx_stride']
        #c3, cc, atom3, am3, idx3, size3 = basis[k]['coef'], basis[k]['exp'], basis[k]['atom'], basis[k]['am'], basis[k]['idx'], basis[k]['idx_stride']
        #c4, dd, atom4, am4, idx4, size4 = basis[l]['coef'], basis[l]['exp'], basis[l]['atom'], basis[l]['am'], basis[l]['idx'], basis[l]['idx_stride']
        ## This is just a dummy am, in the future would have to go through all possibilities 
        #L = (am1,0,0,am2,0,0,am3,0,0,am4,0,0)
        #A = geom[atom1]
        #B = geom[atom2]
        #C = geom[atom3]
        #D = geom[atom4]
        #test += contracted_tei(L,A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)
    return test 


print(experiment(geom, basis_dict))
print(experiment(geom, basis_dict))


