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
    primitive_index = 0
    basis_data = []
    centers_bra = []
    centers_ket = []
    sizes=[]
    segment = []
    indices = []
    segment_id = 0

    start_pair = []

    #TODO highest contraction size
    #K = 36
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
            # This index is where the CONTRACTED integral components go for this shell pair
            # How to get where primitive integrals go? has to have leading axis shape of nprimitive shells, not total primitives
            # Or, how to contract the primitives?
            index = old_cartesian_product(row_indices,col_indices)

            size = ((bra_am + 1) * (bra_am + 2) // 2) * ((ket_am + 1) * (ket_am + 2) // 2) 
             
            for k in range(exp_combos.shape[0]):
                basis_data.append([exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],bra_am,ket_am])
                centers_bra.append(atom1_idx)
                centers_ket.append(atom2_idx)
                segment.append(segment_id)
                start_pair.append([row_idx,col_idx])

                #start = primitive_index
                #stop = primitive_index + size
                #indx = onp.pad(onp.arange(start, stop), (0,K-size), constant_values=-1)
                indices.append(index)
                #primitive_index += size
                #print(bra_am,ket_am,indx)

            if bra_am == 0 and ket_am == 0:
                ss_indices.append(index)
            elif bra_am == 1 and ket_am == 0:
                ps_indices.append(index)
            elif bra_am == 0 and ket_am == 1:
                sp_indices.append(index)
            elif bra_am == 1 and ket_am == 1:
                pp_indices.append(index)

            sizes.append(size)
            segment_id += 1


    ss_indices = np.asarray(onp.vstack(ss_indices))
    ps_indices = np.asarray(onp.vstack(ps_indices))
    sp_indices = np.asarray(onp.vstack(sp_indices))
    pp_indices = np.asarray(onp.vstack(pp_indices))
    all_indices = [ss_indices,ps_indices,sp_indices,pp_indices]
    print(all_indices[0].shape)
    print(all_indices[1].shape)
    print(all_indices[2].shape)
    print(all_indices[3].shape)

    return np.asarray(onp.asarray(basis_data)), centers_bra, centers_ket, np.asarray(onp.vstack(indices)), np.asarray(onp.asarray(segment)), np.asarray(onp.asarray(sizes)), np.asarray(onp.asarray(start_pair)), all_indices

a = time.time()
print("starting preprocessing")
basis_data, centers1, centers2, indices, sid, sizes, start_pair, all_indices = preprocess(geom, basis_dict, nshells)
print("preprocessing done")
b = time.time()
print(b-a)

print("here")


#print(basis_data.shape)
#print(sid)
#print(sid.shape)
#print('indices size')
#print(indices.size)
#print('indices shape')
#print(indices.shape)

def build_overlap(geom, centers1, centers2, basis_data, indices,sizes):
    S = np.zeros((nbf,nbf))
    centers_bra = np.take(geom, centers1, axis=0)
    centers_ket = np.take(geom, centers2, axis=0)
    # all masks, centers, and basis data are based on total contracted shells. In order  (256,) going 
    # in order to get indices of 
    print("generating masks")
    ssmask =  (basis_data[:,-2] == 0) & (basis_data[:,-1] == 0)
    psmask = ((basis_data[:,-2] == 1) & (basis_data[:,-1] == 0))
    spmask = ((basis_data[:,-2] == 0) & (basis_data[:,-1] == 1))
    ppmask =  (basis_data[:,-2] == 1) & (basis_data[:,-1] == 1)
    
    dsmask = ((basis_data[:,-2] == 2) & (basis_data[:,-1] == 0)) | ((basis_data[:,-2] == 0) & (basis_data[:,-1] == 2))
    dpmask = ((basis_data[:,-2] == 2) & (basis_data[:,-1] == 1)) | ((basis_data[:,-2] == 1) & (basis_data[:,-1] == 2))
    ddmask =  (basis_data[:,-2] == 2) & (basis_data[:,-1] == 2)
    print("masks generated")
    
    s_orb = np.any(ssmask)
    p_orb = np.any(psmask)
    d_orb = np.any(dsmask)

    def ssmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        return overlap_ss(*args)

    if s_orb: 
        ss_primitives = jax.lax.map(ssmap, (centers_bra[ssmask], centers_ket[ssmask], basis_data[ssmask]))
        #S = jax.ops.index_update(S, (all_indices[0][:,0], all_indices[0][:,1]), ss_primitives)
        ss_contracted = jax.ops.segment_sum(ss_primitives, sidmask(sid,ssmask))
        S = jax.ops.index_update(S, (all_indices[0][:,0], all_indices[0][:,1]), ss_contracted)

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
        #S = jax.ops.index_add(S, (all_indices[1][:,0], all_indices[1][:,1]), ps_primitives.flatten())
        #S = jax.ops.index_add(S, (all_indices[2][:,0], all_indices[2][:,1]), sp_primitives.flatten())
        ps_contracted = jax.ops.segment_sum(ps_primitives, sidmask(sid,psmask))
        sp_contracted = jax.ops.segment_sum(sp_primitives, sidmask(sid,spmask))
        S = jax.ops.index_update(S, (all_indices[1][:,0], all_indices[1][:,1]), ps_contracted.reshape(-1))
        S = jax.ops.index_update(S, (all_indices[2][:,0], all_indices[2][:,1]), sp_contracted.reshape(-1))

    def ppmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        return overlap_pp(*args).reshape(-1) # TODO need to reshape?

    if p_orb:
        pp_primitives = jax.lax.map(ppmap, (centers_bra[ppmask], centers_ket[ppmask], basis_data[ppmask]))
        pp_contracted = jax.ops.segment_sum(pp_primitives, sidmask(sid,ppmask))


        #print('pp contracted')
        #print(pp_contracted.shape)

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
        print('ds contracted')
        print(ds_contracted.shape)

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
        print('dp contracted')
        print(dp_contracted.shape)

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
        print('dd contracted')
        print(dd_contracted.shape)

#    # These masks are wrong
#    print(indices.shape)
#    print(indices[:,0][ssmask])
#    print(indices[:,1][ssmask])
#    print(indices[:,0][ssmask].shape)
#
#    print(indices[:,0][psmask])
#    print(indices[:,1][psmask])
#    print(indices[:,0][psmask].shape)
#
#    print(indices[:,0][ppmask])
#    print(indices[:,1][ppmask])
#    print(indices[:,0][ppmask].shape)
#
#    # HYPOTHESIS: the unique indices (return index = True in np.unique) of sid[ssmask] will give the right masks of indices
#    print(sid[ssmask])
#    print(sid[psmask])

    # This method generates an overlap that is not symmetric, problem, yea?
    #S = jax.ops.index_update(S, (indices[:,0],indices[:,1]), np.concatenate((ss_contracted, ps_contracted.flatten(), pp_contracted.flatten(), ds_contracted.flatten(), dp_contracted.flatten(), dd_contracted.flatten())))
    #S = jax.ops.index_update(S, (indices[:,0],indices[:,1]), np.concatenate((ss_contracted, ps_contracted.flatten(), pp_contracted.flatten())))



    # If you want to do segment sum, 'Normalize' the segments first, have to be in numerical order.
    #uniq, unique_indices, counts = onp.unique(sid[ssmask],return_index=True, return_counts=True)
    #newsid = np.repeat(np.arange(uniq.shape[0]), counts)
    #ss_contracted = jax.ops.segment_sum(ss_primitives, newsid)
    # Build overlap
    #S = np.zeros((nbf,nbf))
    #S = jax.ops.index_update(S, (indices[np.repeat(ssmask, sizes),0],indices[np.repeat(ssmask, sizes),1]), ss_contracted)

    # how to mask indices?
    #print(indices[np.repeat(ssmask, sizes),0].shape)
    #print(ss_contracted.shape)
    #S = jax.ops.index_update(S, (indices[np.repeat(ssmask, sizes),0],indices[np.repeat(ssmask, sizes),1]), ss_contracted)
    #S = jax.ops.index_add(S, (indices[np.repeat(ssmask, sizes),0],indices[np.repeat(ssmask, sizes),1]), ss_primitives)
    #S = jax.ops.index_add(S, (indices[np.repeat(ssmask, sizes[ssmask]),0],indices[np.repeat(ssmask, sizes[ssmask]),1]), ss_primitives)
    print(S)


build_overlap(geom, centers1, centers2, basis_data, indices, sizes)

