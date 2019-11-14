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
pprint(basis_dict)
# Number of basis functions, number of shells
nbf = basis_set.nbf()
print("number of basis functions", nbf)
nshells = len(basis_dict)


V_overlap_ss = jax.vmap(overlap_ss, (None,None,None,None,None,None,0,0,0,0))
V_overlap_ps = jax.vmap(overlap_ps, (None,None,None,None,None,None,0,0,0,0))
V_overlap_pp = jax.vmap(overlap_pp, (None,None,None,None,None,None,0,0,0,0))
V_overlap_ds = jax.vmap(overlap_ds, (None,None,None,None,None,None,0,0,0,0))
V_overlap_dp = jax.vmap(overlap_dp, (None,None,None,None,None,None,0,0,0,0))
V_overlap_dd = jax.vmap(overlap_dd, (None,None,None,None,None,None,0,0,0,0))

def preprocess(geom, basis_dict, nshells):
    basis_data = []
    am_data = []
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

    biggest_K = 9 #TODO hard coded

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

            # Geometry data
            centers_bra.append(atom1_idx)
            centers_ket.append(atom2_idx)

            # Basis function data
            exp_combos = old_cartesian_product(exp1,exp2)
            coeff_combos = old_cartesian_product(c1,c2)
            current_contract_K = exp_combos.shape[0]
            #TODO TODO TODO fixes nans
            exp_combos = onp.pad(exp_combos, ((0, biggest_K - current_contract_K), (0,0)))
            #exp_combos = onp.pad(exp_combos, ((0, biggest_K - current_contract_K), (0,0)), constant_values=1.0)
            #TODO TODO TODO fixes nan
            coeff_combos = onp.pad(coeff_combos, ((0, biggest_K - current_contract_K), (0,0)))
            basis_data.append([exp_combos[:,0], exp_combos[:,1], coeff_combos[:,0], coeff_combos[:,1]])

            # Angular momentum data
            am_data.append([bra_am, ket_am])

            #row_indices = onp.repeat(row_idx, row_idx_stride) + onp.arange(row_idx_stride)
            #col_indices = onp.repeat(col_idx, col_idx_stride) + onp.arange(col_idx_stride)
            #index = old_cartesian_product(row_indices,col_indices)
            
    #basis_data = onp.asarray(basis_data).transpose(0,2,1)
    basis_data = onp.asarray(basis_data)
    am_data = onp.asarray(am_data)
    return np.array(basis_data), np.array(am_data), centers_bra, centers_ket

print("starting preprocessing")
a = time.time()
basis_data, am_data, centers1, centers2 = preprocess(geom, basis_dict, nshells)
b = time.time()
print("preprocessing done")
print(b-a)

def build_overlap(geom, centers1, centers2, basis_data, am_data):
    # Define overlap of zeros
    S = np.zeros((nbf,nbf))
    centers_bra = np.take(geom, centers1, axis=0)
    centers_ket = np.take(geom, centers2, axis=0)

    print("generating masks")
    ssmask = (am_data[:,0] == 0) & (am_data[:,1] == 0)
    spmask = (am_data[:,0] == 0) & (am_data[:,1] == 1)
    psmask = (am_data[:,0] == 1) & (am_data[:,1] == 0)
    ppmask = (am_data[:,0] == 1) & (am_data[:,1] == 1)
    dsmask = (am_data[:,0] == 2) & (am_data[:,1] == 0)
    sdmask = (am_data[:,0] == 0) & (am_data[:,1] == 2)
    dpmask = (am_data[:,0] == 2) & (am_data[:,1] == 1)
    pdmask = (am_data[:,0] == 1) & (am_data[:,1] == 2)
    ddmask = (am_data[:,0] == 2) & (am_data[:,1] == 2)
    print("masks generated")
    
    s_orb = np.any(ssmask)
    p_orb = np.any(psmask)
    d_orb = np.any(dsmask)

    all_contracted = np.array([])

    def ssmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2 = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        # Dont compute dummy padded contraction values
        return np.where(alpha_bra == 0, 0.0, V_overlap_ss(*args))

    if s_orb: 
        ss_primitives = jax.lax.map(ssmap, (centers_bra[ssmask], centers_ket[ssmask], basis_data[ssmask]))
        ss_contracted = np.sum(ss_primitives, axis=1)
        all_contracted = np.concatenate((all_contracted, ss_contracted.reshape(-1)))

    def psmap(centers_bra, centers_ket, basis_data, am_data):
        #centers_bra, centers_ket, basis_data, am_data = inp
        bra_am, ket_am = am_data[0], am_data[1]
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2 = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        # Dont compute dummy padded contraction values
        return np.where(alpha_bra == 0, 0.0, 
               np.where((bra_am == 1) & (ket_am == 0), V_overlap_ps(*args).T, 
               np.where((bra_am == 0) & (ket_am == 1), V_overlap_ps(*sgra).T, 0.0)))

    def ppmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2 = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        # Dont compute dummy padded contraction values
        return np.where(alpha_bra == 0, 0.0, V_overlap_pp(*args).transpose((1,2,0)))

    if p_orb:
        # Create jit compiled, vectorized version so the second call is fast
        v_psmap = jax.jit(jax.vmap(psmap, in_axes=(0,0,0,0)))
    
        ps_primitives = v_psmap(centers_bra[psmask], centers_ket[psmask], basis_data[psmask], am_data[psmask])
        ps_contracted = np.sum(ps_primitives, axis=-1)

        sp_primitives = v_psmap(centers_bra[spmask], centers_ket[spmask], basis_data[spmask], am_data[spmask])
        sp_contracted = np.sum(sp_primitives, axis=-1)


        pp_primitives = jax.lax.map(ppmap, (centers_bra[ppmask], centers_ket[ppmask], basis_data[ppmask]))
        pp_contracted = np.sum(pp_primitives, axis=-1)

        all_contracted = np.concatenate((all_contracted, ps_contracted.reshape(-1), sp_contracted.reshape(-1), pp_contracted.reshape(-1)))

    def dsmap(centers_bra, centers_ket, basis_data, am_data):
        #centers_bra, centers_ket, basis_data, am_data = inp
        bra_am, ket_am = am_data[0], am_data[1]
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2 = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        # Dont compute dummy padded contraction values
        return np.where(alpha_bra == 0, 0.0, 
               np.where((bra_am == 2) & (ket_am == 0), V_overlap_ds(*args).T, 
               np.where((bra_am == 0) & (ket_am == 2), V_overlap_ds(*sgra).T, 0.0)))

    #def dpmap(inp):
    def dpmap(centers_bra, centers_ket, basis_data, am_data):
        #centers_bra, centers_ket, basis_data, am_data = inp
        bra_am, ket_am = am_data[0], am_data[1]
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2 = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        # Dont compute dummy padded contraction values
        return np.where(alpha_bra == 0, 0.0, 
               np.where((bra_am == 2) & (ket_am == 1), V_overlap_dp(*args).T, 
               np.where((bra_am == 1) & (ket_am == 2), V_overlap_dp(*sgra).T, 0.0)))

    def ddmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2 = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        # Dont compute dummy padded contraction values
        return np.where(alpha_bra == 0, 0.0, V_overlap_dd(*args).transpose((1,2,0)))

    if d_orb:
        v_dsmap = jax.jit(jax.vmap(dsmap, in_axes=(0,0,0,0)))
        v_dpmap = jax.jit(jax.vmap(dpmap, in_axes=(0,0,0,0)))

        #ds_primitives = jax.lax.map(dsmap, (centers_bra[dsmask], centers_ket[dsmask], basis_data[dsmask], am_data[dsmask]))
        ds_primitives = v_dsmap(centers_bra[dsmask], centers_ket[dsmask], basis_data[dsmask], am_data[dsmask])
        ds_contracted = np.sum(ds_primitives, axis=-1)

        #sd_primitives = jax.lax.map(dsmap, (centers_bra[sdmask], centers_ket[sdmask], basis_data[sdmask], am_data[sdmask]))
        sd_primitives = v_dsmap(centers_bra[sdmask], centers_ket[sdmask], basis_data[sdmask], am_data[sdmask])
        sd_contracted = np.sum(sd_primitives, axis=-1)

        #dp_primitives = jax.lax.map(dpmap, (centers_bra[dpmask], centers_ket[dpmask], basis_data[dpmask], am_data[dpmask]))
        dp_primitives = v_dpmap(centers_bra[dpmask], centers_ket[dpmask], basis_data[dpmask], am_data[dpmask])
        dp_contracted = np.sum(dp_primitives, axis=-1).T

        #pd_primitives = jax.lax.map(dpmap, (centers_bra[pdmask], centers_ket[pdmask], basis_data[pdmask], am_data[pdmask]))
        pd_primitives = v_dpmap(centers_bra[pdmask], centers_ket[pdmask], basis_data[pdmask], am_data[pdmask])
        pd_contracted = np.sum(pd_primitives, axis=-1)

        dd_primitives = jax.lax.map(ddmap, (centers_bra[ddmask], centers_ket[ddmask], basis_data[ddmask]))
        dd_contracted = np.sum(dd_primitives, axis=-1)

        all_contracted = np.concatenate((all_contracted, ds_contracted.reshape(-1), sd_contracted.reshape(-1), dp_contracted.reshape(-1), pd_contracted.reshape(-1), dd_contracted.reshape(-1)))

#    print(all_contracted.shape)
    return all_contracted

S = build_overlap(geom, centers1, centers2, basis_data, am_data)
#grad = jax.jacfwd(build_overlap)(geom, centers1, centers2, basis_data, am_data)
#print(grad.shape)
hess = jax.jacfwd(jax.jacfwd(build_overlap))(geom, centers1, centers2, basis_data, am_data)
#print(hess)

#mints = psi4.core.MintsHelper(basis_set)
#psi_S = np.asarray(onp.asarray(mints.ao_overlap()))
#print(np.allclose(S, psi_S))

