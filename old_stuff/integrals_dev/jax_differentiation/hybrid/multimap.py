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

molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         H 0.0 0.0  2.000000000000
                         H 0.0 0.0  3.000000000000
                         """)

# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))

# Get Psi Basis Set and basis set dictionary objects
#basis_name = 'sto-3g'
#basis_name = '6-31g'
#basis_name = 'cc-pvdz'
#basis_name = 'cc-pvtz'
basis_name = '6-311++g'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)

basis_dict[16] = basis_dict[0]
basis_dict[17] = basis_dict[1]
basis_dict[18] = basis_dict[2]
basis_dict[19] = basis_dict[3]
basis_dict[20] = basis_dict[4]
basis_dict[21] = basis_dict[5]
basis_dict[22] = basis_dict[6]
basis_dict[23] = basis_dict[7]
basis_dict[24] = basis_dict[8]
basis_dict[25] = basis_dict[9]
basis_dict[26] = basis_dict[10]
basis_dict[27] = basis_dict[11]
basis_dict[28] = basis_dict[12]
basis_dict[29] = basis_dict[13]
basis_dict[30] = basis_dict[14]
basis_dict[31] = basis_dict[15]

basis_dict[32] = basis_dict[0]
basis_dict[33] = basis_dict[1]
basis_dict[34] = basis_dict[2]
basis_dict[35] = basis_dict[3]
basis_dict[36] = basis_dict[4]
basis_dict[37] = basis_dict[5]
basis_dict[38] = basis_dict[6]
basis_dict[39] = basis_dict[7]
basis_dict[40] = basis_dict[8]
basis_dict[41] = basis_dict[9]
basis_dict[42] = basis_dict[10]
basis_dict[43] = basis_dict[11]
basis_dict[44] = basis_dict[12]
basis_dict[45] = basis_dict[13]
basis_dict[46] = basis_dict[14]
basis_dict[47] = basis_dict[15]

basis_dict[48] = basis_dict[0]
basis_dict[49] = basis_dict[1]
basis_dict[50] = basis_dict[2]
basis_dict[51] = basis_dict[3]
basis_dict[52] = basis_dict[4]
basis_dict[53] = basis_dict[5]
basis_dict[54] = basis_dict[6]
basis_dict[55] = basis_dict[7]
basis_dict[56] = basis_dict[8]
basis_dict[57] = basis_dict[9]
basis_dict[58] = basis_dict[10]
basis_dict[59] = basis_dict[11]
nbf = 60

#pprint(basis_dict)
#pprint(basis_dict)
# Number of basis functions, number of shells
#nbf = basis_set.nbf()
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

    primitive_locations = onp.concatenate((onp.asarray(ss_indices, dtype=np.int64).flatten(),
                                           onp.asarray(ps_indices, dtype=np.int64).flatten(), 
                                           onp.asarray(sp_indices, dtype=np.int64).flatten(),
                                           onp.asarray(pp_indices, dtype=np.int64).flatten(),
                                           onp.asarray(ds_indices, dtype=np.int64).flatten(),
                                           onp.asarray(sd_indices, dtype=np.int64).flatten(),
                                           onp.asarray(dp_indices, dtype=np.int64).flatten(),
                                           onp.asarray(pd_indices, dtype=np.int64).flatten(),
                                           onp.asarray(dd_indices, dtype=np.int64).flatten())).reshape(-1,2)

    primitive_locations = np.asarray(primitive_locations)
    return np.asarray(onp.asarray(basis_data)), centers_bra, centers_ket, primitive_locations

print("starting preprocessing")
a = time.time()
basis_data, centers1, centers2, primitive_locations = preprocess(geom, basis_dict, nshells)
print(basis_data.shape)
print(len(centers1))
print(len(centers2))
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
        return overlap_pp(*args).reshape(-1) 

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
        sd_primitives = jax.lax.map(dsmap, (centers_bra[sdmask], centers_ket[sdmask], basis_data[sdmask]))
        all_primitives = np.concatenate((all_primitives, ds_primitives.reshape(-1), sd_primitives.reshape(-1)))

    def dpmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        #return np.where((bra_am == 2) & (ket_am == 1), overlap_dp(*args).reshape(-1),  #TEMP TODO
        return np.where((bra_am == 2) & (ket_am == 1), overlap_dp(*args).T.reshape(-1), 
               np.where((bra_am == 1) & (ket_am == 2), overlap_dp(*sgra).reshape(-1), 0.0)) 

    if d_orb:
        dp_primitives = jax.lax.map(dpmap, (centers_bra[dpmask], centers_ket[dpmask], basis_data[dpmask]))
        pd_primitives = jax.lax.map(dpmap, (centers_bra[pdmask], centers_ket[pdmask], basis_data[pdmask])) 
        all_primitives = np.concatenate((all_primitives, dp_primitives.reshape(-1), pd_primitives.reshape(-1)))


    def ddmap(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        return overlap_dd(*args).reshape(-1) # TODO need to reshape?

    if d_orb:
        dd_primitives = jax.lax.map(ddmap, (centers_bra[ddmask], centers_ket[ddmask], basis_data[ddmask]))
        all_primitives = np.concatenate((all_primitives, dd_primitives.reshape(-1)))

    # Just one call to index add 
    #print(all_primitives.shape)
    # This tells you if memory use is because of primitives or indeX_add

    #@jax.jit
    #def build():
    #    S = np.zeros((nbf,nbf))
    #    
    #    def update(S, i,j, val):
    #        S = jax.ops.index_add(S, (i,j), val)
    #        return S
    #    
    #    vecmap = jax.jit(jax.vmap(update, (None,0,0,0)))
    #    S = vecmap(S, primitive_locations[:,0], primitive_locations[:,1], all_primitives)
    #    return S

    ##TODO this works
    #new_S = jax.ops.index_add(S, (primitive_locations[:,0], primitive_locations[:,1]), all_primitives)
    #return new_S
    return all_primitives

    #S = build()

    #@jax.jit
    #def update(all_primitives, primitive_locations):
    #    S = np.zeros((nbf,nbf))
    #    S = jax.ops.index_add(S, (primitive_locations[:,0], primitive_locations[:,1]), all_primitives)
    #    return S

    #return all_primitives
    #S = jax.ops.index_add(S, (primitive_locations[:,0], primitive_locations[:,1]), all_primitives)
    #S = update(all_primitives, primitive_locations)
    #return S

S = build_overlap(geom, centers1, centers2, basis_data, primitive_locations)
#print(S.shape)
#grad = jax.jacfwd(build_overlap)(geom, centers1, centers2, basis_data, primitive_locations)
#print(grad.shape)
#cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(build_overlap)))(geom, centers1, centers2, basis_data, primitive_locations)
#print(hess)
quar = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(build_overlap))))(geom, centers1, centers2, basis_data, primitive_locations)
print(quar.shape)

#mints = psi4.core.MintsHelper(basis_set)
#psi_S = np.asarray(onp.asarray(mints.ao_overlap()))
#print(np.allclose(S, psi_S))

