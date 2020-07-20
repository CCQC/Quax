import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product
np.set_printoptions(linewidth=800, suppress=True)
from pprint import pprint
from oei_s import * 
from oei_p import * 
from oei_d import * 
from oei_f import * 

# Define molecule
molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         units bohr
                         """)

#molecule = psi4.geometry("""
#                         0 1
#                         H 0.0 0.0 -0.849220457955
#                         H 0.0 0.0  0.849220457955
#                         H 0.0 0.0  2.000000000000
#                         H 0.0 0.0  3.000000000000
#                         H 0.0 0.0  4.000000000000
#                         H 0.0 0.0  5.000000000000
#                         H 0.0 0.0  6.000000000000
#                         H 0.0 0.0  7.000000000000
#                         H 0.0 0.0  8.000000000000
#                         H 0.0 0.0  9.000000000000
#                         H 0.0 0.0  10.000000000000
#                         H 0.0 0.0  11.000000000000
#                         H 0.0 0.0  12.000000000000
#                         H 0.0 0.0  13.000000000000
#                         H 0.0 0.0  14.000000000000
#                         H 0.0 0.0  15.000000000000
#                         units bohr
#                         """)


# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))
# Get Psi Basis Set and basis set dictionary objects
basis_name = 'cc-pvtz'
#basis_name = 'cc-pvdz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
# Number of basis functions, number of shells
nbf = basis_set.nbf()
print("number of basis functions", nbf)
nshells = len(basis_dict)

overlap_funcs = {}
overlap_funcs['00'] = jax.vmap(overlap_ss, (None,None,None,None,None,None,0,0,0,0))
overlap_funcs['10'] = jax.vmap(overlap_ps, (None,None,None,None,None,None,0,0,0,0))
overlap_funcs['11'] = jax.vmap(overlap_pp, (None,None,None,None,None,None,0,0,0,0))
overlap_funcs['20'] = jax.vmap(overlap_ds, (None,None,None,None,None,None,0,0,0,0))
overlap_funcs['21'] = jax.vmap(overlap_dp, (None,None,None,None,None,None,0,0,0,0))
overlap_funcs['22'] = jax.vmap(overlap_dd, (None,None,None,None,None,None,0,0,0,0))

overlap_funcs['30'] = jax.vmap(overlap_fs, (None,None,None,None,None,None,0,0,0,0))
overlap_funcs['31'] = jax.vmap(overlap_fp, (None,None,None,None,None,None,0,0,0,0))
overlap_funcs['32'] = jax.vmap(overlap_fd, (None,None,None,None,None,None,0,0,0,0))
overlap_funcs['33'] = jax.vmap(overlap_ff, (None,None,None,None,None,None,0,0,0,0))

print('computing overlap')
def build_overlap(geom, basis_dict, nbf, nshells, overlap_funcs):
    '''uses unique functions '''
    S = np.zeros((nbf,nbf))
    for i in range(nshells):
        for j in range(nshells):
            # Load data for this contracted integral
            # This is a slow part of the loop!
            c1 =    np.asarray(basis_dict[i]['coef'])
            c2 =    np.asarray(basis_dict[j]['coef'])
            exp1 =  np.asarray(basis_dict[i]['exp'])
            exp2 =  np.asarray(basis_dict[j]['exp'])
            atom1 = basis_dict[i]['atom']
            atom2 = basis_dict[j]['atom']
            row_idx = basis_dict[i]['idx']
            col_idx = basis_dict[j]['idx']
            row_idx_stride = basis_dict[i]['idx_stride']
            col_idx_stride = basis_dict[j]['idx_stride']
            # This is a slow part of the loop!
            Ax,Ay,Az = geom[atom1]
            Bx,By,Bz = geom[atom2]
    
            # Function identifier
            exp_combos = old_cartesian_product(exp1,exp2)
            coeff_combos = old_cartesian_product(c1,c2)

            bra = basis_dict[i]['am'] 
            ket = basis_dict[j]['am'] 
            if int(bra) < int(ket):
                lookup = str(basis_dict[j]['am']) +  str(basis_dict[i]['am'])
                primitives = overlap_funcs[lookup](Bx,By,Bz,Ax,Ay,Az,exp_combos[:,1],exp_combos[:,0],coeff_combos[:,1],coeff_combos[:,0])
            else:
                lookup = str(basis_dict[i]['am']) +  str(basis_dict[j]['am'])
                primitives = overlap_funcs[lookup](Ax,Ay,Az,Bx,By,Bz,exp_combos[:,0],exp_combos[:,1],coeff_combos[:,0],coeff_combos[:,1])

            # This fixes shaping error for dp vs pd, etc
            if int(bra) < int(ket):
                contracted = np.sum(primitives, axis=0).reshape(-1)
            else:
                contracted = np.sum(primitives, axis=0).T.reshape(-1)
            row_indices = np.repeat(row_idx, row_idx_stride) + np.arange(row_idx_stride)
            col_indices = np.repeat(col_idx, col_idx_stride) + np.arange(col_idx_stride)
            indices = old_cartesian_product(row_indices,col_indices)

            S = jax.ops.index_update(S, (indices[:,0],indices[:,1]), contracted)
    return S

my_S = build_overlap(geom, basis_dict, nbf, nshells, overlap_funcs)
#print(my_S)

#grad = jax.jacfwd(jax.jacfwd(build_overlap))(geom, basis_dict, nbf, nshells, overlap_funcs)
#print(grad)

mints = psi4.core.MintsHelper(basis_set)
psi_S = np.asarray(onp.asarray(mints.ao_overlap()))
#print(psi_S)
print(np.allclose(my_S, psi_S))
#print(np.isclose(my_S, psi_S))
