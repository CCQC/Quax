import jax.numpy as np
import jax
import numpy as onp
#from jax.config import config; config.update("jax_enable_x64", True)
from integrals_utils import cartesian_product, old_cartesian_product, find_unique_shells, am_vectors
from functools import partial
from jax.experimental import loops
from tei import contracted_tei

geom = np.array([[0.0,0.0,-0.849220457955],
                 [0.0,0.0, 0.849220457955]])

basis_dict =  {0: {'am': 0,
     'atom': 0,
     'coef': [0.25510805774198103, 0.46009788806518803, 0.67841406712793],
     'exp': [33.87, 5.095, 1.159],
     'idx': 0,
     'idx_stride': 1},
 1: {'am': 0,
     'atom': 0,
     'coef': [0.30734305383061117],
     'exp': [0.3258],
     'idx': 1,
     'idx_stride': 1},
 2: {'am': 0,
     'atom': 0,
     'coef': [0.1292968441748187],
     'exp': [0.1027],
     'idx': 2,
     'idx_stride': 1},
 3: {'am': 1,
     'atom': 0,
     'coef': [2.184276984526831],
     'exp': [1.407],
     'idx': 3,
     'idx_stride': 3},
 4: {'am': 1,
     'atom': 0,
     'coef': [0.43649547399719835],
     'exp': [0.388],
     'idx': 6,
     'idx_stride': 3},
 5: {'am': 2,
     'atom': 0,
     'coef': [1.8135965626177861],
     'exp': [1.057],
     'idx': 9,
     'idx_stride': 6},
 6: {'am': 0,
     'atom': 1,
     'coef': [0.25510805774198103, 0.46009788806518803, 0.67841406712793],
     'exp': [33.87, 5.095, 1.159],
     'idx': 15,
     'idx_stride': 1},
 7: {'am': 0,
     'atom': 1,
     'coef': [0.30734305383061117],
     'exp': [0.3258],
     'idx': 16,
     'idx_stride': 1},
 8: {'am': 0,
     'atom': 1,
     'coef': [0.1292968441748187],
     'exp': [0.1027],
     'idx': 17,
     'idx_stride': 1},
 9: {'am': 1,
     'atom': 1,
     'coef': [2.184276984526831],
     'exp': [1.407],
     'idx': 18,
     'idx_stride': 3},
 10: {'am': 1,
      'atom': 1,
      'coef': [0.43649547399719835],
      'exp': [0.388],
      'idx': 21,
      'idx_stride': 3},
 11: {'am': 2,
      'atom': 1,
      'coef': [1.8135965626177861],
      'exp': [1.057],
      'idx': 24,
      'idx_stride': 6}}

def homogenize_basisdict(basis_dict, max_prim):
    new_dict = basis_dict.copy()
    for i in range(len(basis_dict)):
        current_exp = onp.asarray(basis_dict[i]['exp'])
        new_dict[i]['exp'] = onp.asarray(onp.pad(current_exp, (0, max_prim - current_exp.shape[0]), constant_values=1))
        current_coef = onp.asarray(basis_dict[i]['coef'])
        new_dict[i]['coef'] = onp.asarray(onp.pad(current_coef, (0, max_prim - current_coef.shape[0])))
    return new_dict

max_prim = 3 
basis = homogenize_basisdict(basis_dict,max_prim)
print(basis)
nbf = 30

def experiment(geom, basis):
    nshells = len(basis)
    shell_quartets = cartesian_product(np.arange(nshells), np.arange(nshells), np.arange(nshells), np.arange(nshells))
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

    with loops.Scope() as s:
      s.G = np.zeros((nbf,nbf,nbf,nbf))
      s.a = 0  # center A angular momentum iterator 
      s.b = 0  # center B angular momentum iterator 
      s.c = 0  # center C angular momentum iterator 
      s.d = 0  # center D angular momentum iterator 
      # Loop over quartets of basis functions (shell quartets)
      #for _ in s.while_range(lambda: s.q < shell_quartets.shape[0]):
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
                tei = contracted_tei(La,Lb,Lc,Ld,A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)
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
      return s.G

G = experiment(geom, basis_dict)
print('done')


