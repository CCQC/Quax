import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product
np.set_printoptions(linewidth=800, suppress=True, threshold=10000000)
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
print("number of basis functions", nbf)
nshells = len(basis_dict)

def preprocess(geom, basis_dict, nshells):
    basis_data = []
    centers_bra = []
    centers_ket = []
    indices = []
    sizes=[]
    segment = []
    segment_id = 0
    tmp_place = 0

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

            size = ((bra_am + 1) * (bra_am + 2) // 2) * ((ket_am + 1) * (ket_am + 2) // 2) 
            place = onp.arange(tmp_place, size + tmp_place) + segment_id
            tmp_place += size 
             
            for k in range(exp_combos.shape[0]):
                basis_data.append([exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],bra_am,ket_am])
                centers_bra.append(atom1_idx)
                centers_ket.append(atom2_idx)
                segment.append(segment_id)
            indices.append(index)
            sizes.append(size)
            segment_id += 1
    return np.asarray(onp.asarray(basis_data)), centers_bra, centers_ket, np.asarray(onp.vstack(indices)), np.asarray(onp.asarray(segment)), np.asarray(onp.asarray(sizes))

import time
a = time.time()
print("starting preprocessing")
basis_data, centers1, centers2, indices, sid, sizes = preprocess(geom, basis_dict, nshells)

print("preprocessing done")
b = time.time()
print(b-a)


def build_overlap(geom, centers1, centers2, basis_data, indices,sizes):
    centers_bra = np.take(geom, centers1, axis=0)
    centers_ket = np.take(geom, centers2, axis=0)
    # Vectors of all possible centers cartesian components for each primitive basis function
    #Ax, Ay, Az, Cx, Cy, Cz = np.split(np.hstack((centers_bra, centers_ket)), 6, axis=1)

    ssmask =  (basis_data[:,-2] == 0) & (basis_data[:,-1] == 0)
    psmask = ((basis_data[:,-2] == 1) & (basis_data[:,-1] == 0)) | ((basis_data[:,-2] == 0) & (basis_data[:,-1] == 1))
    ppmask =  (basis_data[:,-2] == 1) & (basis_data[:,-1] == 1)
    dsmask = ((basis_data[:,-2] == 2) & (basis_data[:,-1] == 0)) | ((basis_data[:,-2] == 0) & (basis_data[:,-1] == 2))
    dpmask = ((basis_data[:,-2] == 2) & (basis_data[:,-1] == 1)) | ((basis_data[:,-2] == 1) & (basis_data[:,-1] == 2))
    ddmask =  (basis_data[:,-2] == 2) & (basis_data[:,-1] == 2)

    def ssmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        return overlap_ss(*args)

    ss_primitives = jax.lax.map(ssmap, (centers_bra[ssmask], centers_ket[ssmask], basis_data[ssmask]))

    def psmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        return np.where((bra_am == 1) & (ket_am == 0), overlap_ps(*args).reshape(-1), 
               np.where((bra_am == 0) & (ket_am == 1), overlap_ps(*sgra).reshape(-1), 0.0))

    ps_primitives = jax.lax.map(psmap, (centers_bra[psmask], centers_ket[psmask], basis_data[psmask]))

    def ppmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        return overlap_pp(*args).reshape(-1) # TODO need to reshape?

    pp_primitives = jax.lax.map(ppmap, (centers_bra[ppmask], centers_ket[ppmask], basis_data[ppmask]))

    def dsmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        return np.where((bra_am == 2) & (ket_am == 0), overlap_ds(*args).reshape(-1), 
               np.where((bra_am == 0) & (ket_am == 2), overlap_ds(*sgra).reshape(-1), 0.0))

    ds_primitives = jax.lax.map(dsmap, (centers_bra[dsmask], centers_ket[dsmask], basis_data[dsmask]))

    def dpmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        return np.where((bra_am == 2) & (ket_am == 1), overlap_dp(*args).reshape(-1), 
               np.where((bra_am == 1) & (ket_am == 2), overlap_dp(*sgra).reshape(-1), 0.0))

    dp_primitives = jax.lax.map(dpmap, (centers_bra[dpmask], centers_ket[dpmask], basis_data[dpmask]))

    def ddmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        return overlap_dd(*args).reshape(-1) # TODO need to reshape?

    dd_primitives = jax.lax.map(ddmap, (centers_bra[ddmask], centers_ket[ddmask], basis_data[ddmask]))

    ## Collect primitives in proper order
    #primitives = np.zeros_like(indices[:,0])
    #primitives = jax.ops.index_update(primitives, ([ssmask,0],indices[ssmask,1]), ss_primitives)


    # If you want to do segment sum, 'Normalize' the segments first, have to be in numerical order.
    #uniq, unique_indices, counts = onp.unique(sid[ssmask],return_index=True, return_counts=True)
    #newsid = np.repeat(np.arange(uniq.shape[0]), counts)
    #ss_contracted = jax.ops.segment_sum(ss_primitives, newsid)

    #print(ss_contracted)
    # where does this go? ^^^^^^
    #print(unique_indices.shape)
    #print(np.repeat(ssmask, sizes).shape)

    # Build overlap
    #S = np.zeros((nbf,nbf))
    #S = jax.ops.index_update(S, (indices[np.repeat(ssmask, sizes),0],indices[np.repeat(ssmask, sizes),1]), ss_contracted)

    # how to mask indices?
    #print(indices[np.repeat(ssmask, sizes),0].shape)
    #print(ss_contracted.shape)
    #S = jax.ops.index_update(S, (indices[np.repeat(ssmask, sizes),0],indices[np.repeat(ssmask, sizes),1]), ss_contracted)
    #S = jax.ops.index_add(S, (indices[np.repeat(ssmask, sizes),0],indices[np.repeat(ssmask, sizes),1]), ss_primitives)
    #S = jax.ops.index_add(S, (indices[np.repeat(ssmask, sizes[ssmask]),0],indices[np.repeat(ssmask, sizes[ssmask]),1]), ss_primitives)
    #print(S)


build_overlap(geom, centers1, centers2, basis_data, indices, sizes)

