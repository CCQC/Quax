import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product
np.set_printoptions(linewidth=800, suppress=True, threshold=10000000)
from pprint import pprint
import time
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
#                         """)

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
#basis_name = '6-31g'
#basis_name = 'sto-3g'
#basis_name = 'cc-pvdz'
basis_name = 'cc-pvtz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)


#pprint(basis_dict)
# Number of basis functions, number of shells
nbf = basis_set.nbf()
print("number of basis functions", nbf)
nshells = len(basis_dict)

def preprocess(geom, basis_dict, nshells):
    basis_data = []
    centers_bra = []
    centers_ket = []
    ss_indices = []
    ps_indices = []
    sp_indices = []
    pp_indices = []
    ds_indices = []
    sd_indices = []
    dp_indices = []
    pd_indices = []
    dd_indices = []
    primitive_locations = []

    for i in range(nshells):
        c1 =    onp.asarray(basis_dict[i]['coef'])
        exp1 =  onp.asarray(basis_dict[i]['exp'])
        atom1_idx = basis_dict[i]['atom']
        row_idx = basis_dict[i]['idx']
        row_idx_stride = basis_dict[i]['idx_stride']
        bra_am = basis_dict[i]['am']
        for j in range(nshells):
            c2 =    onp.asarray(basis_dict[j]['coef'])
            exp2 =  onp.asarray(basis_dict[j]['exp'])
            atom2_idx = basis_dict[j]['atom']
            col_idx = basis_dict[j]['idx']
            col_idx_stride = basis_dict[j]['idx_stride']
            ket_am = basis_dict[j]['am']

            exp_combos = old_cartesian_product(exp1,exp2)
            coeff_combos = old_cartesian_product(c1,c2)
            row_indices = onp.repeat(row_idx, row_idx_stride) + onp.arange(row_idx_stride)
            col_indices = onp.repeat(col_idx, col_idx_stride) + onp.arange(col_idx_stride)
            index = old_cartesian_product(row_indices,col_indices)

            for k in range(exp_combos.shape[0]):
                basis_data.append([exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],bra_am,ket_am])
                centers_bra.append(atom1_idx)
                centers_ket.append(atom2_idx)
                if   bra_am == 0 and ket_am == 0: ss_indices.append(index)
                elif bra_am == 1 and ket_am == 0: ps_indices.append(index)
                elif bra_am == 0 and ket_am == 1: sp_indices.append(index)
                elif bra_am == 1 and ket_am == 1: pp_indices.append(index)
                elif bra_am == 2 and ket_am == 0: ds_indices.append(index)
                elif bra_am == 0 and ket_am == 2: sd_indices.append(index)
                elif bra_am == 2 and ket_am == 1: dp_indices.append(index)
                elif bra_am == 1 and ket_am == 2: pd_indices.append(index)
                elif bra_am == 2 and ket_am == 2: dd_indices.append(index)

    #primitive_locations = onp.concatenate((onp.asarray(ss_indices, dtype=np.int64).flatten(),
    #                                       onp.asarray(ps_indices, dtype=np.int64).flatten(), 
    #                                       onp.asarray(sp_indices, dtype=np.int64).flatten(),
    #                                       onp.asarray(pp_indices, dtype=np.int64).flatten(),
    #                                       onp.asarray(ds_indices, dtype=np.int64).flatten(),
    #                                       onp.asarray(sd_indices, dtype=np.int64).flatten(),
    #                                       onp.asarray(dp_indices, dtype=np.int64).flatten(),
    #                                       onp.asarray(pd_indices, dtype=np.int64).flatten(),
    #                                       onp.asarray(dd_indices, dtype=np.int64).flatten())).reshape(-1,2)

    primitive_locations.append(np.asarray(onp.asarray(ss_indices, dtype=np.int64)).reshape(-1,2))
    primitive_locations.append(np.asarray(onp.asarray(ps_indices, dtype=np.int64)).reshape(-1,2))
    primitive_locations.append(np.asarray(onp.asarray(sp_indices, dtype=np.int64)).reshape(-1,2))
    primitive_locations.append(np.asarray(onp.asarray(pp_indices, dtype=np.int64)).reshape(-1,2))
    primitive_locations.append(np.asarray(onp.asarray(ds_indices, dtype=np.int64)).reshape(-1,2))
    primitive_locations.append(np.asarray(onp.asarray(sd_indices, dtype=np.int64)).reshape(-1,2))
    primitive_locations.append(np.asarray(onp.asarray(dp_indices, dtype=np.int64)).reshape(-1,2))
    primitive_locations.append(np.asarray(onp.asarray(pd_indices, dtype=np.int64)).reshape(-1,2))
    primitive_locations.append(np.asarray(onp.asarray(dd_indices, dtype=np.int64)).reshape(-1,2))

    #primitive_locations = np.asarray(primitive_locations)
    return np.asarray(onp.asarray(basis_data)), centers_bra, centers_ket, primitive_locations

print("starting preprocessing")
a = time.time()
basis_data, centers1, centers2, primitive_locations = preprocess(geom, basis_dict, nshells)
print(primitive_locations[0].shape)
print(primitive_locations[1].shape)
print(primitive_locations[2].shape)
print(primitive_locations[3].shape)
b = time.time()
print("preprocessing done")
print(b-a)

def build_overlap(geom, centers1, centers2, basis_data, primitive_locations):
    # Define overlap of zeros
    S = np.zeros((nbf,nbf))
    centers_bra = np.take(geom, centers1, axis=0)
    centers_ket = np.take(geom, centers2, axis=0)
    print("generating masks")
    ssmask = (basis_data[:,-2] == 0) & (basis_data[:,-1] == 0)

    print(ssmask.shape)
    print(primitive_locations[0][:,0].shape)
    psmask = (basis_data[:,-2] == 1) & (basis_data[:,-1] == 0)
    spmask = (basis_data[:,-2] == 0) & (basis_data[:,-1] == 1)
    ppmask = (basis_data[:,-2] == 1) & (basis_data[:,-1] == 1)
    dsmask = (basis_data[:,-2] == 2) & (basis_data[:,-1] == 0)
    sdmask = (basis_data[:,-2] == 0) & (basis_data[:,-1] == 2)
    dpmask = (basis_data[:,-2] == 2) & (basis_data[:,-1] == 1)
    pdmask = (basis_data[:,-2] == 1) & (basis_data[:,-1] == 2)
    ddmask = (basis_data[:,-2] == 2) & (basis_data[:,-1] == 2)
    print("masks generated")
    
    s_orb = np.any(ssmask)
    p_orb = np.any(psmask)
    d_orb = np.any(dsmask)

    all_primitives = np.array([])

    def ss_scan(carry, i):
        S, loc1, loc2, centers_bra, centers_ket, basis_data = carry
        Ax, Ay, Az = centers_bra[i]
        Cx, Cy, Cz = centers_ket[i]
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data[i]
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        val = overlap_ss(*args)
        S = jax.ops.index_add(S, (loc1[i],loc2[i]), val)
        new_carry = (S, loc1, loc2, centers_bra, centers_ket, basis_data)
        return new_carry, 0

    if s_orb: 
        final, _ = jax.lax.scan(ss_scan, (S, primitive_locations[0][:,0], primitive_locations[0][:,1], centers_bra[ssmask], centers_ket[ssmask], basis_data[ssmask]), np.arange(np.count_nonzero(ssmask)))
        S = final[0]

    def ps_scan(carry, i):
        S, loc1, loc2, centers_bra, centers_ket, basis_data = carry
        Ax, Ay, Az = centers_bra[i]
        Cx, Cy, Cz = centers_ket[i]
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data[i]
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        val = np.where((bra_am == 1) & (ket_am == 0), overlap_ps(*args).reshape(-1), 
              np.where((bra_am == 0) & (ket_am == 1), overlap_ps(*sgra).reshape(-1), 0.0))
        S = jax.ops.index_add(S, (loc1[i],loc2[i]), val)
        new_carry = (S, loc1, loc2, centers_bra, centers_ket, basis_data)
        return new_carry, 0

    def pp_scan(carry, i):
        S, loc1, loc2, centers_bra, centers_ket, basis_data = carry
        Ax, Ay, Az = centers_bra[i]
        Cx, Cy, Cz = centers_ket[i]
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data[i]
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        val = overlap_pp(*args).reshape(-1) 
        S = jax.ops.index_add(S, (loc1[i],loc2[i]), val)
        new_carry = (S, loc1, loc2, centers_bra, centers_ket, basis_data)
        return new_carry, 0

    if p_orb:
        # NOTE could combine this to one scan call, may make memory worse
        # NOTE have to reshape so that 3 primitives are summed into array at each scan call
        final, _ = jax.lax.scan(ps_scan, (S, primitive_locations[1][:,0].reshape(-1,3), primitive_locations[1][:,1].reshape(-1,3), centers_bra[psmask], centers_ket[psmask], basis_data[psmask]), np.arange(np.count_nonzero(psmask)))
        S = final[0]
        final, _ = jax.lax.scan(ps_scan, (S, primitive_locations[2][:,0].reshape(-1,3), primitive_locations[2][:,1].reshape(-1,3), centers_bra[spmask], centers_ket[spmask], basis_data[spmask]), np.arange(np.count_nonzero(spmask)))
        S = final[0]
        final, _ = jax.lax.scan(pp_scan, (S, primitive_locations[3][:,0].reshape(-1,9), primitive_locations[3][:,1].reshape(-1,9), centers_bra[ppmask], centers_ket[ppmask], basis_data[ppmask]), np.arange(np.count_nonzero(ppmask)))
        S = final[0]

    def ds_scan(carry, i):
        S, loc1, loc2, centers_bra, centers_ket, basis_data = carry
        Ax, Ay, Az = centers_bra[i]
        Cx, Cy, Cz = centers_ket[i]
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data[i]
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        val = np.where((bra_am == 2) & (ket_am == 0), overlap_ds(*args).reshape(-1), 
              np.where((bra_am == 0) & (ket_am == 2), overlap_ds(*sgra).reshape(-1), 0.0))
        S = jax.ops.index_add(S, (loc1[i],loc2[i]), val)
        new_carry = (S, loc1, loc2, centers_bra, centers_ket, basis_data)
        return new_carry, 0

    def dp_scan(carry, i):
        S, loc1, loc2, centers_bra, centers_ket, basis_data = carry
        Ax, Ay, Az = centers_bra[i]
        Cx, Cy, Cz = centers_ket[i]
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data[i]
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        val = np.where((bra_am == 2) & (ket_am == 1), overlap_dp(*args).T.reshape(-1), 
              np.where((bra_am == 1) & (ket_am == 2), overlap_dp(*sgra).reshape(-1), 0.0)) 
        S = jax.ops.index_add(S, (loc1[i],loc2[i]), val)
        new_carry = (S, loc1, loc2, centers_bra, centers_ket, basis_data)
        return new_carry, 0

    def dd_scan(carry, i):
        S, loc1, loc2, centers_bra, centers_ket, basis_data = carry
        Ax, Ay, Az = centers_bra[i]
        Cx, Cy, Cz = centers_ket[i]
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data[i]
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        val = overlap_dd(*args).reshape(-1)
        S = jax.ops.index_add(S, (loc1[i],loc2[i]), val)
        new_carry = (S, loc1, loc2, centers_bra, centers_ket, basis_data)
        return new_carry, 0

    if d_orb:
        final, _ = jax.lax.scan(ds_scan, (S, primitive_locations[4][:,0].reshape(-1,6), primitive_locations[4][:,1].reshape(-1,6), centers_bra[dsmask], centers_ket[dsmask], basis_data[dsmask]), np.arange(np.count_nonzero(dsmask)))
        S = final[0]
        final, _ = jax.lax.scan(ds_scan, (S, primitive_locations[5][:,0].reshape(-1,6), primitive_locations[5][:,1].reshape(-1,6), centers_bra[sdmask], centers_ket[sdmask], basis_data[sdmask]), np.arange(np.count_nonzero(sdmask)))
        S = final[0]

        final, _ = jax.lax.scan(dp_scan, (S, primitive_locations[6][:,0].reshape(-1,18), primitive_locations[6][:,1].reshape(-1,18), centers_bra[dpmask], centers_ket[dpmask], basis_data[dpmask]), np.arange(np.count_nonzero(dpmask)))
        S = final[0]
        final, _ = jax.lax.scan(dp_scan, (S, primitive_locations[7][:,0].reshape(-1,18), primitive_locations[7][:,1].reshape(-1,18), centers_bra[pdmask], centers_ket[pdmask], basis_data[pdmask]), np.arange(np.count_nonzero(pdmask)))
        S = final[0]

        final, _ = jax.lax.scan(dd_scan, (S, primitive_locations[8][:,0].reshape(-1,36), primitive_locations[8][:,1].reshape(-1,36), centers_bra[ddmask], centers_ket[ddmask], basis_data[ddmask]), np.arange(np.count_nonzero(ddmask)))
        S = final[0]


    return S

S = build_overlap(geom, centers1, centers2, basis_data, primitive_locations)
print(S)
#S = build_overlap(geom, centers1, centers2, basis_data, primitive_locations)
#print(S)
#grad = jax.jacfwd(build_overlap)(geom, centers1, centers2, basis_data, primitive_locations)
#print(grad.shape)
#cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(build_overlap)))(geom, centers1, centers2, basis_data, primitive_locations)
#print(hess)

mints = psi4.core.MintsHelper(basis_set)
psi_S = np.asarray(onp.asarray(mints.ao_overlap()))
print(np.allclose(S, psi_S))

