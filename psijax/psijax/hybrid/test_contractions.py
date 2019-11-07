import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product
np.set_printoptions(linewidth=500, suppress=True)
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

# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))
# Get Psi Basis Set and basis set dictionary objects
basis_name = 'cc-pvtz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
# Number of basis functions, number of shells
nbf = basis_set.nbf()
nshells = len(basis_dict)

#
overlap_funcs = {}
overlap_funcs['00'] = jax.jit(jax.vmap(overlap_ss, (None,None,None,None,None,None,0,0,0,0)))
overlap_funcs['10'] = jax.jit(jax.vmap(overlap_ps, (None,None,None,None,None,None,0,0,0,0)))
overlap_funcs['11'] = jax.jit(jax.vmap(overlap_pp, (None,None,None,None,None,None,0,0,0,0)))
overlap_funcs['20'] = jax.jit(jax.vmap(overlap_ds, (None,None,None,None,None,None,0,0,0,0)))
overlap_funcs['21'] = jax.jit(jax.vmap(overlap_dp, (None,None,None,None,None,None,0,0,0,0)))
overlap_funcs['22'] = jax.jit(jax.vmap(overlap_dd, (None,None,None,None,None,None,0,0,0,0)))



pprint(basis_dict)

def build_overlap_old(geom, basis_dict, nbf, nshells, overlap_funcs):
    '''uses redundant functions '''
    S = np.zeros((nbf,nbf))
    for i in range(nshells):
        for j in range(nshells):
            # Load data for this contracted integral
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
            Ax,Ay,Az = geom[atom1]
            Bx,By,Bz = geom[atom2]
    
            # Function identifier
            exp_combos = old_cartesian_product(exp1,exp2)
            coeff_combos = old_cartesian_product(c1,c2)

            bra = basis_dict[i]['am'] 
            ket = basis_dict[j]['am'] 

            lookup = basis_dict[i]['am'] +  basis_dict[j]['am']
            primitives = overlap_funcs[lookup](Ax,Ay,Az,Bx,By,Bz,exp_combos[:,0],exp_combos[:,1],coeff_combos[:,0],coeff_combos[:,1])
            contracted = np.sum(primitives, axis=0).reshape(-1)
            row_indices = np.repeat(row_idx, row_idx_stride)+ np.arange(row_idx_stride)
            col_indices = np.repeat(col_idx, col_idx_stride)+ np.arange(col_idx_stride)
            indices = old_cartesian_product(row_indices,col_indices)
            S = jax.ops.index_update(S, (indices[:,0],indices[:,1]), contracted)
    return S

def build_overlap(geom, basis_dict, nbf, nshells, overlap_funcs):
    '''uses unique functions '''
    S = np.zeros((nbf,nbf))
    for i in range(nshells):
        for j in range(nshells):
            # Load data for this contracted integral
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
            Ax,Ay,Az = geom[atom1]
            Bx,By,Bz = geom[atom2]
    
            # Function identifier
            exp_combos = old_cartesian_product(exp1,exp2)
            coeff_combos = old_cartesian_product(c1,c2)

            bra = basis_dict[i]['am'] 
            ket = basis_dict[j]['am'] 
            if int(bra) < int(ket):
                lookup = basis_dict[j]['am'] +  basis_dict[i]['am']
                primitives = overlap_funcs[lookup](Bx,By,Bz,Ax,Ay,Az,exp_combos[:,1],exp_combos[:,0],coeff_combos[:,1],coeff_combos[:,0])
            else:
                lookup = basis_dict[i]['am'] +  basis_dict[j]['am']
                primitives = overlap_funcs[lookup](Ax,Ay,Az,Bx,By,Bz,exp_combos[:,0],exp_combos[:,1],coeff_combos[:,0],coeff_combos[:,1])

            #BUG TODO I bet you need to handle (d|a) and (a|d) indices in different ways!
            contracted = np.sum(primitives, axis=0).reshape(-1)
            # Find index block where this (these) contracted integral(s) should go
            # BUG TODO THIS SCHEME FOR LOCATION FINDING IS WRONG
            row_indices = np.repeat(row_idx, row_idx_stride) + np.arange(row_idx_stride)
            col_indices = np.repeat(col_idx, col_idx_stride) + np.arange(col_idx_stride)
            indices = old_cartesian_product(row_indices,col_indices)
            print(lookup)
            print('indices shape', indices.shape)

            S = jax.ops.index_update(S, (indices[:,0],indices[:,1]), contracted)
    return S


def fast_build_overlap(geom, basis_dict, nbf, nshells, overlap_funcs):
    S = np.zeros((nbf,nbf))

    data = []
    contraction_ids = []
    num_contractions = 0
    repeats = []
    for i in range(nshells):
        for j in range(nshells):
            # Load data for this contracted integral
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
            Ax,Ay,Az = geom[atom1]
            Bx,By,Bz = geom[atom2]
            bra = basis_dict[i]['am'] 
            ket = basis_dict[j]['am'] 

            exp_combos = old_cartesian_product(exp1,exp2)
            coeff_combos = old_cartesian_product(c1,c2)

            for k in range(exp_combos.shape[0]):
                data.append(np.array([Ax,Ay,Az,Bx,By,Bz,exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],int(bra),int(ket)]))
            repeats.append(k+1)
            contraction_ids.append(num_contractions)
            num_contractions += 1

    # You need to rethink how to do this 
    # Here, we are looping over shells, and collecting arguments for computing a primitive,
    # but based on the angular momentum, a SINGLE ROW of `data` will actually be many primitives (3 for p, 6 for d, 10 for f, 15 for g)

    #cd_array = onp.asarray(contraction_ids)
    #repeats = onp.asarray(repeats)
    #contraction_ids = onp.asarray(contraction_ids)
    #segment_ids = onp.repeat(contraction_ids, repeats)
    #print(segment_ids)
    #print(segment_ids.shape)
    #segment_ids = onp.repeat(onp.arange(num_contractions)), cd_array)
    #print(segment_ids)
    #print(segment_ids.shape)
    
    #data_array = np.asarray(data)
    #print(data_array.shape)
    #print(segment_ids.shape)
    #print("unique segment ids")
    #print(onp.unique(segment_ids).shape)

    #test = jax.ops.segment_sum(np.asarray(data), np.asarray(segment_ids))
    #print(test.shape)
    #print('nbf',nbf)
            
    
my_S = build_overlap(geom, basis_dict, nbf, nshells, overlap_funcs)
print(my_S)
#my_S2 = build_overlap_old(geom, basis_dict, nbf, nshells, overlap_funcs)
#print(np.allclose(my_S, my_S2))

#print(my_S[:,:9])
# Psi4 data
mints = psi4.core.MintsHelper(basis_set)
psi_S = np.asarray(onp.asarray(mints.ao_overlap()))
print(psi_S)
print(np.allclose(my_S, psi_S))
print(np.isclose(my_S, psi_S))

#print(np.isclose((np.allclose(my_S,psi_S), True, False))

#print(np.equal(my_S, psi_S))
