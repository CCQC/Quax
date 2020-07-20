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

# Define molecule
molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         units bohr
                         """)

molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         H 0.0 0.0  2.000000000000
                         H 0.0 0.0  3.000000000000
                         H 0.0 0.0  4.000000000000
                         H 0.0 0.0  5.000000000000
                         H 0.0 0.0  6.000000000000
                         H 0.0 0.0  7.000000000000
                         H 0.0 0.0  8.000000000000
                         H 0.0 0.0  9.000000000000
                         H 0.0 0.0  10.000000000000
                         H 0.0 0.0  11.000000000000
                         H 0.0 0.0  12.000000000000
                         H 0.0 0.0  13.000000000000
                         H 0.0 0.0  14.000000000000
                         H 0.0 0.0  15.000000000000
                         H 0.0 0.0  16.000000000000
                         H 0.0 0.0  17.000000000000
                         H 0.0 0.0  18.000000000000
                         H 0.0 0.0  19.000000000000
                         units bohr
                         """)

# Get Psi Basis Set and basis set dictionary objects
#basis_name = '6-31g'
#basis_name = 'sto-3g'
basis_name = 'cc-pvdz'
#basis_name = 'cc-pvtz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)

# hack to make big basis but small system
for key in basis_dict:
    basis_dict[key]['atom'] = 0

# Define molecule
molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         units bohr
                         """)

# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))


# Number of basis functions, number of shells
nbf = basis_set.nbf()
print("number of basis functions", nbf)

def preprocess(geom, basis_dict):
    nshells = len(basis_dict)
    ss_basis_data, ps_basis_data, sp_basis_data, pp_basis_data, ds_basis_data, sd_basis_data, dp_basis_data, pd_basis_data, dd_basis_data = [],[],[],[],[],[],[],[],[]
    ss_centers_bra, ps_centers_bra, sp_centers_bra, pp_centers_bra, ds_centers_bra, sd_centers_bra, dp_centers_bra, pd_centers_bra, dd_centers_bra = [],[],[],[],[],[],[],[],[]
    ss_centers_ket, ps_centers_ket, sp_centers_ket, pp_centers_ket, ds_centers_ket, sd_centers_ket, dp_centers_ket, pd_centers_ket, dd_centers_ket = [],[],[],[],[],[],[],[],[]
    ss_indices, ps_indices, sp_indices, pp_indices, ds_indices, sd_indices, dp_indices, pd_indices, dd_indices = [],[],[],[],[],[],[],[],[]
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

            # Compute all primitive combinations of exponents and contraction coefficients
            exp_combos = old_cartesian_product(exp1,exp2)
            coeff_combos = old_cartesian_product(c1,c2)
            row_indices = onp.repeat(row_idx, row_idx_stride) + onp.arange(row_idx_stride)
            col_indices = onp.repeat(col_idx, col_idx_stride) + onp.arange(col_idx_stride)
            index = old_cartesian_product(row_indices,col_indices)

            for k in range(exp_combos.shape[0]):
                if bra_am == 0 and ket_am == 0: 
                    ss_basis_data.append([exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],bra_am,ket_am])
                    ss_centers_bra.append(atom1_idx)
                    ss_centers_ket.append(atom2_idx)
                    ss_indices.append(index)
                elif bra_am == 1 and ket_am == 0: 
                    ps_basis_data.append([exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],bra_am,ket_am])
                    ps_centers_bra.append(atom1_idx)
                    ps_centers_ket.append(atom2_idx)
                    ps_indices.append(index)
                elif bra_am == 0 and ket_am == 1:
                    sp_basis_data.append([exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],bra_am,ket_am])
                    sp_centers_bra.append(atom1_idx)
                    sp_centers_ket.append(atom2_idx)
                    sp_indices.append(index)
                elif bra_am == 1 and ket_am == 1: 
                    pp_basis_data.append([exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],bra_am,ket_am])
                    pp_centers_bra.append(atom1_idx)
                    pp_centers_ket.append(atom2_idx)
                    pp_indices.append(index)
                elif bra_am == 2 and ket_am == 0: 
                    ds_basis_data.append([exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],bra_am,ket_am])
                    ds_centers_bra.append(atom1_idx)
                    ds_centers_ket.append(atom2_idx)
                    ds_indices.append(index)
                elif bra_am == 0 and ket_am == 2: 
                    sd_basis_data.append([exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],bra_am,ket_am])
                    sd_centers_bra.append(atom1_idx)
                    sd_centers_ket.append(atom2_idx)
                    sd_indices.append(index)
                elif bra_am == 2 and ket_am == 1: 
                    dp_basis_data.append([exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],bra_am,ket_am])
                    dp_centers_bra.append(atom1_idx)
                    dp_centers_ket.append(atom2_idx)
                    dp_indices.append(index)
                elif bra_am == 1 and ket_am == 2: 
                    pd_basis_data.append([exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],bra_am,ket_am])
                    pd_centers_bra.append(atom1_idx)
                    pd_centers_ket.append(atom2_idx)
                    pd_indices.append(index)
                elif bra_am == 2 and ket_am == 2: 
                    dd_basis_data.append([exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],bra_am,ket_am])
                    dd_centers_bra.append(atom1_idx)
                    dd_centers_ket.append(atom2_idx)
                    dd_indices.append(index)
    # Build increments, which tells where in the vectors a new integral type begins
    inc = []
    inc.append(0)
    inc.append(len(ss_indices))
    inc.append(inc[-1] + len(ps_indices) + len(sp_indices))
    inc.append(inc[-1] + len(pp_indices))
    inc.append(inc[-1] + len(ds_indices) + len(sd_indices))
    inc.append(inc[-1] + len(dp_indices) + len(pd_indices))
    inc.append(inc[-1] + len(dd_indices))

    basis_data = onp.concatenate((onp.asarray(ss_basis_data).reshape(-1),
                                  onp.asarray(ps_basis_data).reshape(-1),
                                  onp.asarray(sp_basis_data).reshape(-1),
                                  onp.asarray(pp_basis_data).reshape(-1),
                                  onp.asarray(ds_basis_data).reshape(-1),
                                  onp.asarray(sd_basis_data).reshape(-1),
                                  onp.asarray(dp_basis_data).reshape(-1),
                                  onp.asarray(pd_basis_data).reshape(-1),
                                  onp.asarray(dd_basis_data).reshape(-1))).reshape(-1,6)
    centers_bra = ss_centers_bra + ps_centers_bra + sp_centers_bra + pp_centers_bra + ds_centers_bra + sd_centers_bra + dp_centers_bra + pd_centers_bra + dd_centers_bra 
    centers_ket = ss_centers_ket + ps_centers_ket + sp_centers_ket + pp_centers_ket + ds_centers_ket + sd_centers_ket + dp_centers_ket + pd_centers_ket + dd_centers_ket 
    #centers_ket = ss_centers_ket + ps_centers_ket + sp_centers_ket + pp_centers_ket
    locations = {'ss':onp.asarray(ss_indices), 
                 'ps':onp.concatenate((onp.asarray(ps_indices),onp.asarray(sp_indices))),
                 'pp':onp.asarray(pp_indices),
                 'ds':onp.concatenate((onp.asarray(ds_indices),onp.asarray(sd_indices))),
                 'dp':onp.concatenate((onp.asarray(dp_indices),onp.asarray(pd_indices))),
                 'dd':onp.asarray(dd_indices)
                }
    return np.asarray(onp.asarray(basis_data)), centers_bra, centers_ket, locations, inc

print("starting preprocessing")
a = time.time()
basis_data, centers1, centers2, locations, inc = preprocess(geom, basis_dict)
print(inc)
b = time.time()
print("preprocessing done")
print(b-a)

def build_overlap(geom, centers1, centers2, basis_data, locations):
    # Define overlap of zeros
    S = np.zeros((nbf,nbf))
    print("processing geometry")
    # This step takes awhile. Probably better to reformulate using np.repeat.
    centers_bra = np.take(geom, centers1, axis=0) 
    centers_ket = np.take(geom, centers2, axis=0)
    print("Geometry processed")

    
    s_orb = np.any(basis_data[:,-2] == 0)
    p_orb = np.any(basis_data[:,-2] == 1)
    d_orb = np.any(basis_data[:,-2] == 2)
    print("am types bools generated")

    def ss_scan(carry, i):
        S, loc1, loc2, centers_bra, centers_ket, basis_data = carry
        A = centers_bra[i]
        C = centers_ket[i]
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data[i]
        args = (A, C, alpha_bra, alpha_ket, c1, c2)
        val = overlap_ss(*args)
        S = jax.ops.index_add(S, (loc1[i],loc2[i]), val)
        new_carry = (S, loc1, loc2, centers_bra, centers_ket, basis_data)
        return new_carry, 0

    if s_orb: 
        a,b = inc[0], inc[1]
        final, _ = jax.lax.scan(ss_scan, (S, locations['ss'][...,0], locations['ss'][...,1], centers_bra[a:b], centers_ket[a:b], basis_data[a:b]), np.arange(basis_data[a:b].shape[0]))
        S = final[0]
        print("ss integrals added")

    def ps_scan(carry, i):
        S, loc1, loc2, centers_bra, centers_ket, basis_data = carry
        A = centers_bra[i]
        C = centers_ket[i]
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data[i]
        args = (A, C, alpha_bra, alpha_ket, c1, c2)
        sgra = (C, A, alpha_ket, alpha_bra, c2, c1)
        val = np.where((bra_am == 1) & (ket_am == 0), overlap_ps(*args).reshape(-1), 
              np.where((bra_am == 0) & (ket_am == 1), overlap_ps(*sgra).reshape(-1), 0.0))
        S = jax.ops.index_add(S, (loc1[i],loc2[i]), val)
        new_carry = (S, loc1, loc2, centers_bra, centers_ket, basis_data)
        return new_carry, 0

    def pp_scan(carry, i):
        S, loc1, loc2, centers_bra, centers_ket, basis_data = carry
        A = centers_bra[i]
        C = centers_ket[i]
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data[i]
        args = (A, C, alpha_bra, alpha_ket, c1, c2)
        val = overlap_pp(*args).reshape(-1) 
        S = jax.ops.index_add(S, (loc1[i],loc2[i]), val)
        new_carry = (S, loc1, loc2, centers_bra, centers_ket, basis_data)
        return new_carry, 0

    if p_orb:
        a,b = inc[1], inc[2]
        final, _ = jax.lax.scan(ps_scan, (S, locations['ps'][...,0], locations['ps'][...,1], centers_bra[a:b], centers_ket[a:b], basis_data[a:b]), np.arange(basis_data[a:b].shape[0]))
        S = final[0]
        print("ps integrals added")
        a,b = inc[2], inc[3]
        final, _ = jax.lax.scan(pp_scan, (S, locations['pp'][...,0], locations['pp'][...,1], centers_bra[a:b], centers_ket[a:b], basis_data[a:b]), np.arange(basis_data[a:b].shape[0]))
        S = final[0]
        print("pp integrals added")

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
        a,b = inc[3], inc[4]
        final, _ = jax.lax.scan(ds_scan, (S, locations['ds'][...,0], locations['ds'][...,1], centers_bra[a:b], centers_ket[a:b], basis_data[a:b]), np.arange(basis_data[a:b].shape[0]))
        S = final[0]
        print("ds integrals added")

        a,b = inc[4], inc[5]
        final, _ = jax.lax.scan(dp_scan, (S, locations['dp'][...,0], locations['dp'][...,1], centers_bra[a:b], centers_ket[a:b], basis_data[a:b]), np.arange(basis_data[a:b].shape[0]))
        S = final[0]
        print("dp integrals added")

        a,b = inc[5], inc[6]
        final, _ = jax.lax.scan(dd_scan, (S, locations['dd'][...,0], locations['dd'][...,1], centers_bra[a:b], centers_ket[a:b], basis_data[a:b]), np.arange(basis_data[a:b].shape[0]))
        S = final[0]
        print("dd integrals added")

    return S

#S = build_overlap(geom, centers1, centers2, basis_data, locations)

#grad = jax.jacfwd(build_overlap)(geom, centers1, centers2, basis_data, primitive_locations)
#print(grad.shape)
#hess = jax.jacfwd(jax.jacfwd(build_overlap))(geom, centers1, centers2, basis_data, locations)
#cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(build_overlap)))(geom, centers1, centers2, basis_data, locations)
quar = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(build_overlap))))(geom, centers1, centers2, basis_data, locations)
print(quar.shape)
#quar = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(build_overlap))))(geom, centers1, centers2, basis_data, locations)
#print(quar.shape)

#mints = psi4.core.MintsHelper(basis_set)
#psi_S = np.asarray(onp.asarray(mints.ao_overlap()))
#print(np.allclose(S, psi_S))

