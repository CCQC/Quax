import psi4
import jax.numpy as np
import jax
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set,homogenize_basisdict
from integrals_utils import cartesian_product, old_cartesian_product, find_unique_shells, am_vectors
from functools import partial
from jax.experimental import loops
from pprint import pprint
from tei import primitive_quartet

# Define molecule
#molecule = psi4.geometry("""
#                         0 1
#                         H 0.0 0.0 -0.849220457955
#                         H 0.0 0.0  0.849220457955
#                         units bohr
#                         """)
molecule = psi4.geometry("""
                         0 2
                         H 0.0 0.0 -0.849220457955
                         units bohr
                         """)

# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))

basis_name = 'cc-pv5z'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
pprint(basis_dict)
# Homogenize the basis set dictionary
max_prim = basis_set.max_nprimitive()
max_am = basis_set.max_am()
nprim = basis_set.nprimitive()
#biggest_K = max_prim**4
nbf = basis_set.nbf()
print("Number of basis functions: ", nbf)
print("Number of primitives: ", nprim)
#basis_dict = homogenize_basisdict(basis_dict, max_prim)

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
        tmp_coeffs = basis[i]['coef']  
        tmp_exps = basis[i]['exp']  
        for j in tmp_coeffs:
            coeffs.append(j)
            atoms.append(basis[i]['atom'])
            ams.append(basis[i]['am'])
            indices.append(basis[i]['idx'])
            dims.append(basis[i]['idx_stride'])
        for j in tmp_exps:
            exps.append(j)

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

    with loops.Scope() as s:
      s.G = np.zeros((nbf,nbf,nbf,nbf))
      s.a = 0  # center A angular momentum iterator 
      s.b = 0  # center B angular momentum iterator 
      s.c = 0  # center C angular momentum iterator 
      s.d = 0  # center D angular momentum iterator 
      # Loop over primitive quartets, compute integral, add to appropriate index in G
      for p1 in s.range(nprim):
        for p2 in s.range(nprim):
          for p3 in s.range(nprim):
            for p4 in s.range(nprim):
              c1, c2, c3, c4 = coeffs[p1], coeffs[p2], coeffs[p3], coeffs[p4]
              aa, bb, cc, dd = exps[p1], exps[p2], exps[p3], exps[p4]
              atom1, atom2, atom3, atom4 = atoms[p1], atoms[p2], atoms[p3], atoms[p4]
              A, B, C, D = geom[atom1], geom[atom2], geom[atom3], geom[atom4]
              am1,am2,am3,am4 = ams[p1], ams[p2], ams[p3], ams[p4]
              ld1, ld2, ld3, ld4 = leading_indices[am1],leading_indices[am2],leading_indices[am3],leading_indices[am4]
              s.a = 0
              for _ in s.while_range(lambda: s.a < dims[p1]):
                s.b = 0
                for _ in s.while_range(lambda: s.b < dims[p2]):
                  s.c = 0
                  for _ in s.while_range(lambda: s.c < dims[p3]):
                    s.d = 0
                    for _ in s.while_range(lambda: s.d < dims[p4]):
                      La = angular_momentum[s.a + ld1]
                      Lb = angular_momentum[s.b + ld2]
                      Lc = angular_momentum[s.c + ld3]
                      Ld = angular_momentum[s.d + ld4]
                      tei = primitive_quartet(La,Lb,Lc,Ld,A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)
                      # add to appropriate index in G
                      i = indices[p1] + s.a
                      j = indices[p2] + s.b
                      k = indices[p3] + s.c
                      l = indices[p4] + s.d
                      s.G = jax.ops.index_add(s.G, jax.ops.index[i,j,k,l], tei) 
                      s.d += 1
                    s.c += 1
                  s.b += 1
                s.a += 1
      return s.G

G = experiment(geom, basis_dict)
mints = psi4.core.MintsHelper(basis_set)
psi_G = np.asarray(onp.asarray(mints.ao_eri()))
print("Matches Psi4: ", np.allclose(G, psi_G))

##print(onp.where(onp.equal(onp.asarray(G), onp.asarray(psi_G))))
print("Indices which are incorrect:")
problem_idx = onp.vstack(onp.where(~onp.isclose(G, psi_G))).T
print(problem_idx.shape)
print(problem_idx)

for idx in problem_idx:
    i,j,k,l = idx
    print(G[i,j,k,l],psi_G[i,j,k,l])












