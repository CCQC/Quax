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

# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))

# Get Psi Basis Set and basis set dictionary objects
basis_name = 'cc-pvdz'
#basis_name = 'cc-pvdz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
pprint(basis_dict)
# Number of basis functions, number of shells
nbf = basis_set.nbf()
print("number of basis functions", nbf)
nshells = len(basis_dict)

def sidmask(sid, mask):
    uniq, counts = onp.unique(sid[mask], return_counts=True)
    newsid = np.repeat(np.arange(uniq.shape[0]), counts)
    return newsid

def preprocess(geom, basis_dict, nshells):
    basis_data = []
    centers_bra = []
    centers_ket = []
    ss_indices = []
    ps_indices = []
    sp_indices = []
    pp_indices = []
    ds_indices = []
    dp_indices = []
    dd_indices = []

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
                if bra_am == 0 and ket_am == 0:
                    ss_indices.append(index)
                elif bra_am == 1 and ket_am == 0:
                    ps_indices.append(index)
                elif bra_am == 0 and ket_am == 1:
                    sp_indices.append(index)
                elif bra_am == 1 and ket_am == 1:
                    pp_indices.append(index)

    ss_indices = np.asarray(onp.vstack(ss_indices))
    ps_indices = np.asarray(onp.vstack(ps_indices))
    sp_indices = np.asarray(onp.vstack(sp_indices))
    pp_indices = np.asarray(onp.vstack(pp_indices))
    primitive_locations = np.concatenate([ss_indices,ps_indices,sp_indices,pp_indices])
    return np.asarray(onp.asarray(basis_data)), centers_bra, centers_ket, primitive_locations 

print("starting preprocessing")
a = time.time()
basis_data, centers1, centers2, primitive_locations = preprocess(geom, basis_dict, nshells)
b = time.time()
print("preprocessing done")
print(b-a)

def build_overlap(geom, centers1, centers2, basis_data, primitive_locations):
    # Define overlap of zeros
    S = np.zeros((nbf,nbf))
    centers_bra = np.take(geom, centers1, axis=0)
    centers_ket = np.take(geom, centers2, axis=0)
    # all masks, centers, and basis data are based on total contracted shells.
    # in order to get indices of 
    print("generating masks")
    ssmask = (basis_data[:,-2] == 0) & (basis_data[:,-1] == 0)
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

    def ssmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        return overlap_ss(*args)

    if s_orb: 
        ss_primitives = jax.lax.map(ssmap, (centers_bra[ssmask], centers_ket[ssmask], basis_data[ssmask]))
        all_primitives = np.concatenate((all_primitives, ss_primitives))

    def psmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        return np.where((bra_am == 1) & (ket_am == 0), overlap_ps(*args).reshape(-1), 
               np.where((bra_am == 0) & (ket_am == 1), overlap_ps(*sgra).reshape(-1), 0.0))

    if p_orb:
        ps_primitives = jax.lax.map(psmap, (centers_bra[psmask], centers_ket[psmask], basis_data[psmask]))
        sp_primitives = jax.lax.map(psmap, (centers_bra[spmask], centers_ket[spmask], basis_data[spmask]))
        all_primitives = np.concatenate((all_primitives, ps_primitives.reshape(-1), sp_primitives.reshape(-1)))

    def ppmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        return overlap_pp(*args).reshape(-1) # TODO need to reshape?

    if p_orb:
        pp_primitives = jax.lax.map(ppmap, (centers_bra[ppmask], centers_ket[ppmask], basis_data[ppmask]))
        all_primitives = np.concatenate((all_primitives, pp_primitives.reshape(-1)))

    def dsmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        return np.where((bra_am == 2) & (ket_am == 0), overlap_ds(*args).reshape(-1), 
               np.where((bra_am == 0) & (ket_am == 2), overlap_ds(*sgra).reshape(-1), 0.0))

    if d_orb:
        ds_primitives = jax.lax.map(dsmap, (centers_bra[dsmask], centers_ket[dsmask], basis_data[dsmask]))
        ds_contracted = jax.ops.segment_sum(ds_primitives, sidmask(sid,dsmask))

    def dpmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        return np.where((bra_am == 2) & (ket_am == 1), overlap_dp(*args).reshape(-1), 
               np.where((bra_am == 1) & (ket_am == 2), overlap_dp(*sgra).reshape(-1), 0.0))

    if d_orb:
        dp_primitives = jax.lax.map(dpmap, (centers_bra[dpmask], centers_ket[dpmask], basis_data[dpmask]))
        dp_contracted = jax.ops.segment_sum(dp_primitives, sidmask(sid,dpmask))

    def ddmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        return overlap_dd(*args).reshape(-1) # TODO need to reshape?

    if d_orb:
        dd_primitives = jax.lax.map(ddmap, (centers_bra[ddmask], centers_ket[ddmask], basis_data[ddmask]))
        dd_contracted = jax.ops.segment_sum(dd_primitives, sidmask(sid,ddmask))

    # Just one call to index add 
    S = jax.ops.index_add(S, (primitive_locations[:,0], primitive_locations[:,1]), all_primitives)
    return S

S = build_overlap(geom, centers1, centers2, basis_data, primitive_locations)

mints = psi4.core.MintsHelper(basis_set)
psi_S = np.asarray(onp.asarray(mints.ao_overlap()))
print(np.allclose(S, psi_S))

