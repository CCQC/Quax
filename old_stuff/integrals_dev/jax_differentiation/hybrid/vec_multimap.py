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

#
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
#                         H 0.0 0.0  16.000000000000
#                         H 0.0 0.0  17.000000000000
#                         H 0.0 0.0  18.000000000000
#                         H 0.0 0.0  19.000000000000
#                         H 0.0 0.0  20.000000000000
#                         H 0.0 0.0  21.000000000000
#                         H 0.0 0.0  22.000000000000
#                         H 0.0 0.0  23.000000000000
#                         H 0.0 0.0  24.000000000000
#                         H 0.0 0.0  25.000000000000
#                         H 0.0 0.0  26.000000000000
#                         H 0.0 0.0  27.000000000000
#                        units bohr
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
#                         H 0.0 0.0  16.000000000000
#                         H 0.0 0.0  17.000000000000
#                         H 0.0 0.0  18.000000000000
#                         H 0.0 0.0  19.000000000000
#                         H 0.0 0.0  20.000000000000
#                         H 0.0 0.0  21.000000000000
#                         H 0.0 0.0  22.000000000000
#                         H 0.0 0.0  23.000000000000
#                         H 0.0 0.0  24.000000000000
#                         H 0.0 0.0  25.000000000000
#                         H 0.0 0.0  26.000000000000
#                         H 0.0 0.0  27.000000000000
#                         H 0.0 0.0  28.000000000000
#                         H 0.0 0.0  29.000000000000
#                         H 0.0 0.0  30.000000000000
#                         H 0.0 0.0  31.000000000000
#                         H 0.0 0.0  32.000000000000
#                         H 0.0 0.0  33.000000000000
#                         H 0.0 0.0  34.000000000000
#                         H 0.0 0.0  35.000000000000
#                         H 0.0 0.0  36.000000000000
#                         H 0.0 0.0  37.000000000000
#                         H 0.0 0.0  38.000000000000
#                         H 0.0 0.0  39.000000000000
#                         H 0.0 0.0  40.000000000000
#                         H 0.0 0.0  41.000000000000
#                         H 0.0 0.0  42.000000000000
#                         H 0.0 0.0  43.000000000000
#                         H 0.0 0.0  44.000000000000
#                         H 0.0 0.0  45.000000000000
#                         H 0.0 0.0  46.000000000000
#                         H 0.0 0.0  47.000000000000
#                         H 0.0 0.0  48.000000000000
#                         H 0.0 0.0  49.000000000000
#                         H 0.0 0.0  50.000000000000
#                         H 0.0 0.0  51.000000000000
#                         H 0.0 0.0  52.000000000000
#                         H 0.0 0.0  53.000000000000
#                         H 0.0 0.0  54.000000000000
#                         H 0.0 0.0  55.000000000000
#                         H 0.0 0.0  56.000000000000
#                         H 0.0 0.0  57.000000000000
#                         H 0.0 0.0  58.000000000000
#                         H 0.0 0.0  59.000000000000
#                         H 0.0 0.0  60.000000000000
#                         H 0.0 0.0  61.000000000000
#                         H 0.0 0.0  62.000000000000
#                         H 0.0 0.0  63.000000000000
#                         H 0.0 0.0  64.000000000000
#                         H 0.0 0.0  65.000000000000
#                         H 0.0 0.0  66.000000000000
#                         H 0.0 0.0  67.000000000000
#                         H 0.0 0.0  68.000000000000
#                         H 0.0 0.0  69.000000000000
#                         H 0.0 0.0  120.000000000000
#                         H 0.0 0.0  121.000000000000
#                         H 0.0 0.0  122.000000000000
#                         H 0.0 0.0  123.000000000000
#                         H 0.0 0.0  124.000000000000
#                         H 0.0 0.0  125.000000000000
#                         H 0.0 0.0  126.000000000000
#                         H 0.0 0.0  127.000000000000
#                         H 0.0 0.0  128.000000000000
#                         H 0.0 0.0  129.000000000000
#                         H 0.0 0.0  130.000000000000
#                         H 0.0 0.0  131.000000000000
#                         H 0.0 0.0  132.000000000000
#                         H 0.0 0.0  133.000000000000
#                         H 0.0 0.0  134.000000000000
#                         H 0.0 0.0  135.000000000000
#                         H 0.0 0.0  136.000000000000
#                         H 0.0 0.0  137.000000000000
#                         H 0.0 0.0  138.000000000000
#                         H 0.0 0.0  139.000000000000
#                         H 0.0 0.0  140.000000000000
#                         H 0.0 0.0  141.000000000000
#                         H 0.0 0.0  142.000000000000
#                         H 0.0 0.0  143.000000000000
#                         H 0.0 0.0  144.000000000000
#                         H 0.0 0.0  145.000000000000
#                         H 0.0 0.0  146.000000000000
#                         H 0.0 0.0  147.000000000000
#                         H 0.0 0.0  148.000000000000
#                         H 0.0 0.0  149.000000000000
#                         H 0.0 0.0  150.000000000000
#                         H 0.0 0.0  151.000000000000
#                         H 0.0 0.0  152.000000000000
#                         H 0.0 0.0  153.000000000000
#                         H 0.0 0.0  154.000000000000
#                         H 0.0 0.0  155.000000000000
#                         H 0.0 0.0  156.000000000000
#                         H 0.0 0.0  157.000000000000
#                         H 0.0 0.0  158.000000000000
#                         H 0.0 0.0  159.000000000000
#                         H 0.0 0.0  160.000000000000
#                         H 0.0 0.0  161.000000000000
#                         H 0.0 0.0  162.000000000000
#                         H 0.0 0.0  163.000000000000
#                         H 0.0 0.0  164.000000000000
#                         H 0.0 0.0  165.000000000000
#                         H 0.0 0.0  166.000000000000
#                         H 0.0 0.0  167.000000000000
#                         H 0.0 0.0  168.000000000000
#                         H 0.0 0.0  169.000000000000
#                         units bohr
#                         """)


# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))

# Get Psi Basis Set and basis set dictionary objects
basis_name = 'cc-pvtz'
#basis_name = 'cc-pvdz'
#basis_name = '6-31g'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
pprint(basis_dict)
# Number of basis functions, number of shells
nbf = basis_set.nbf()
print("number of basis functions", nbf)
nshells = len(basis_dict)

def test():
    ss_indices = []
    ps_indices = []
    sp_indices = []
    pp_indices = []

    for i in range(nshells):
        bra_am = basis_dict[i]['am']
        row_idx = basis_dict[i]['idx']
        for j in range(nshells):
            ket_am = basis_dict[j]['am']
            col_idx = basis_dict[j]['idx']
            if   bra_am == 0 and ket_am == 0: 
                ss_indices.append([row_idx, col_idx])
            elif bra_am == 0 and ket_am == 1: 
                sp_indices.append([row_idx, col_idx])
                sp_indices.append([row_idx, col_idx + 1])
                sp_indices.append([row_idx, col_idx + 2])
            elif bra_am == 1 and ket_am == 0:
                ps_indices.append([row_idx, col_idx])
                ps_indices.append([row_idx + 1, col_idx])
                ps_indices.append([row_idx + 2, col_idx])
            elif bra_am == 1 and ket_am == 1:
                pp_indices.append([row_idx, col_idx])
                pp_indices.append([row_idx, col_idx + 1])
                pp_indices.append([row_idx, col_idx + 2])
                pp_indices.append([row_idx + 1, col_idx])
                pp_indices.append([row_idx + 1, col_idx + 1])
                pp_indices.append([row_idx + 1, col_idx + 2])
                pp_indices.append([row_idx + 2, col_idx])
                pp_indices.append([row_idx + 2, col_idx + 1])
                pp_indices.append([row_idx + 2, col_idx + 2])

    indices = onp.concatenate((onp.asarray(ss_indices), onp.asarray(sp_indices), onp.asarray(ps_indices), onp.asarray(pp_indices)))
    flattened_indices = onp.ravel_multi_index(indices.T, (nbf,nbf))
    return flattened_indices

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

    biggest_K = 64 #TODO hard coded
    oei_indices = old_cartesian_product(onp.arange(nbf),onp.arange(nbf))
    #print(oei_indices)
    bf_idx = 0
    for i in range(nshells):
        c1 =    onp.asarray(basis_dict[i]['coef'])
        exp1 =  onp.asarray(basis_dict[i]['exp'])
        atom1_idx = basis_dict[i]['atom']
        row_idx = basis_dict[i]['idx']
        row_idx_stride = basis_dict[i]['idx_stride']
        bra_am = basis_dict[i]['am']

        #bra_dim = ((bra_am + 1) * (bra_am + 2) // 2)
        #for p in range(bra_dim): 
        #    ROWS.append(p+i)

        for j in range(nshells):
            c2 =    onp.asarray(basis_dict[j]['coef'])
            exp2 =  onp.asarray(basis_dict[j]['exp'])
            atom2_idx = basis_dict[j]['atom']
            col_idx = basis_dict[j]['idx']
            col_idx_stride = basis_dict[j]['idx_stride']
            ket_am = basis_dict[j]['am']

            #bra_dim = ((bra_am + 1) * (bra_am + 2) // 2)
            #ket_dim = ((ket_am + 1) * (ket_am + 2) // 2)
            #size = ((bra_am + 1) * (bra_am + 2) // 2) * ((ket_am + 1) * (ket_am + 2) // 2) 

            #COL_IDX = COL_IDX + ((ket_am + 1) * (ket_am + 2) // 2)
            #for p in range(bra_dim): 
            #    for q in range(ket_dim): 
            #        ROWS.append(i+p)
            #        COLS.append(j+q)


            # Find out what kind of integral this is
            if   bra_am == 0 and ket_am == 0: target = ss_indices
            elif bra_am == 1 and ket_am == 0: target = ps_indices
            elif bra_am == 0 and ket_am == 1: target = sp_indices
            elif bra_am == 1 and ket_am == 1: target = pp_indices
            elif bra_am == 2 and ket_am == 0: target = ds_indices
            elif bra_am == 0 and ket_am == 2: target = sd_indices
            elif bra_am == 2 and ket_am == 1: target = dp_indices
            elif bra_am == 1 and ket_am == 2: target = pd_indices
            elif bra_am == 2 and ket_am == 2: target = dd_indices

            size = ((bra_am + 1) * (bra_am + 2) // 2) * ((ket_am + 1) * (ket_am + 2) // 2) 
            for component in range(size):
                #print(bra_am, ket_am, oei_indices[bf_idx])
                target.append(oei_indices[bf_idx])
                bf_idx += 1

            # Geometry data
            centers_bra.append(atom1_idx)
            centers_ket.append(atom2_idx)

            # Basis function data
            exp_combos = old_cartesian_product(exp1,exp2)
            coeff_combos = old_cartesian_product(c1,c2)
            current_K = exp_combos.shape[0] # Size of this contraction
            exp_combos = onp.pad(exp_combos, ((0, biggest_K - current_K), (0,0)))
            coeff_combos = onp.pad(coeff_combos, ((0, biggest_K - current_K), (0,0)))
            basis_data.append([exp_combos[:,0], exp_combos[:,1], coeff_combos[:,0], coeff_combos[:,1]])

            # Angular momentum data
            am_data.append([bra_am, ket_am])

            #row_indices = onp.repeat(row_idx, row_idx_stride) + onp.arange(row_idx_stride)
            #col_indices = onp.repeat(col_idx, col_idx_stride) + onp.arange(col_idx_stride)
            #index = old_cartesian_product(row_indices,col_indices)

    #print(ss_indices)
    #print(onp.asarray(ss_indices))
    #print(onp.asarray(sp_indices))
    #print(onp.asarray(sp_indices))
    #print("PP")
    #print(onp.asarray(pp_indices))

    #indices = onp.concatenate((onp.asarray(ss_indices), onp.asarray(ps_indices), onp.asarray(sp_indices), onp.asarray(pp_indices)))
    #indices = onp.concatenate((onp.asarray(ss_indices), onp.asarray(sp_indices), onp.asarray(ps_indices), onp.asarray(pp_indices)))
    indices = onp.concatenate((onp.asarray(ss_indices), onp.asarray(ps_indices), onp.asarray(sp_indices), onp.asarray(pp_indices)))
    #print(indices[:,0])
    
    #indices = onp.ravel_multi_index((indices[:,0],indices[:,1]), (nbf*nbf,2))
    indices = onp.ravel_multi_index(indices.T, (nbf,nbf))

            
    basis_data = onp.asarray(basis_data)
    am_data = onp.asarray(am_data)
    return np.array(basis_data), np.array(am_data), centers_bra, centers_ket

print("starting preprocessing")
a = time.time()
basis_data, am_data, centers1, centers2 = preprocess(geom, basis_dict, nshells)
#indices = test()
b = time.time()
print("preprocessing done")
print(b-a)

def build_overlap(geom, centers1, centers2, basis_data, am_data):
    # Define overlap of zeros
    S = np.zeros((nbf,nbf))
    centers_bra = np.take(geom, centers1, axis=0)
    centers_ket = np.take(geom, centers2, axis=0)


    print("generating masks")
    # Generate boolean masks of each integral type
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

    def psmap(inp):
        centers_bra, centers_ket, basis_data, am_data = inp
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
        #return np.where(alpha_bra == 0, 0.0, V_overlap_pp(*args))

    if p_orb:
        # TODO it would probably be much faster to concatenate the spmask and psmask, just do one lax.map call. for ps/sp, ds/sd, dp/pd
        # This would be better if it didnt have to be created here, but started out in this form. 
        # It would likely be better to not use boolean masks at all, and just have clever loop constructs in preprocess 
        #sp_ps_primitives = jax.lax.map(psmap, (np.concatenate((centers_bra[spmask], centers_bra[psmask])),  
        #                                       np.concatenate((centers_ket[spmask], centers_ket[psmask])),
        #                                       np.concatenate((basis_data[spmask], basis_data[psmask])),
        #                                       np.concatenate((am_data[spmask], am_data[psmask]))))
        #sp_ps_contracted = np.sum(sp_ps_primitives, axis=-1)

        ps_primitives = jax.lax.map(psmap, (centers_bra[psmask], centers_ket[psmask], basis_data[psmask], am_data[psmask]))
        ps_contracted = np.sum(ps_primitives, axis=-1)

        sp_primitives = jax.lax.map(psmap, (centers_bra[spmask], centers_ket[spmask], basis_data[spmask], am_data[spmask]))
        sp_contracted = np.sum(sp_primitives, axis=-1)

        pp_primitives = jax.lax.map(ppmap, (centers_bra[ppmask], centers_ket[ppmask], basis_data[ppmask]))
        pp_contracted = np.sum(pp_primitives, axis=-1)

        all_contracted = np.concatenate((all_contracted, sp_contracted.reshape(-1), ps_contracted.reshape(-1), pp_contracted.reshape(-1)))
        #all_contracted = np.concatenate((all_contracted, sp_ps_contracted.reshape(-1), pp_contracted.reshape(-1)))

    def dsmap(inp):
        centers_bra, centers_ket, basis_data, am_data = inp
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

    def dpmap(inp):
        centers_bra, centers_ket, basis_data, am_data = inp
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

        ds_primitives = jax.lax.map(dsmap, (centers_bra[dsmask], centers_ket[dsmask], basis_data[dsmask], am_data[dsmask]))
        ds_contracted = np.sum(ds_primitives, axis=-1)

        sd_primitives = jax.lax.map(dsmap, (centers_bra[sdmask], centers_ket[sdmask], basis_data[sdmask], am_data[sdmask]))
        sd_contracted = np.sum(sd_primitives, axis=-1)

        dp_primitives = jax.lax.map(dpmap, (centers_bra[dpmask], centers_ket[dpmask], basis_data[dpmask], am_data[dpmask]))
        dp_contracted = np.sum(dp_primitives, axis=-1).T

        pd_primitives = jax.lax.map(dpmap, (centers_bra[pdmask], centers_ket[pdmask], basis_data[pdmask], am_data[pdmask]))
        pd_contracted = np.sum(pd_primitives, axis=-1)

        dd_primitives = jax.lax.map(ddmap, (centers_bra[ddmask], centers_ket[ddmask], basis_data[ddmask]))
        dd_contracted = np.sum(dd_primitives, axis=-1)

        all_contracted = np.concatenate((all_contracted, ds_contracted.reshape(-1), sd_contracted.reshape(-1), dp_contracted.reshape(-1), pd_contracted.reshape(-1), dd_contracted.reshape(-1)))
    print(all_contracted.shape)

    #new = all_contracted[indices]

    #basic = np.arange(nbf*nbf)
    #tmp = basic[indices]
    #print(tmp)
    #print(indices)

    #print(np.array([0,1,16,17,18,5,6,19,20,21]))
    #print(indices[:10])

    #print(all_contracted[np.array([0,1,16,17,18,5,6,19,20,21])])

    # THIS WORKS
    #S = np.zeros((nbf*nbf))
    #final = jax.ops.index_update(S, indices, all_contracted).reshape(nbf,nbf)
    #final = all_contracted[indices].reshape(nbf,nbf)
    #return final
        
    return all_contracted

S = build_overlap(geom, centers1, centers2, basis_data, am_data)
#print(S.shape)
##grad = jax.jacfwd(build_overlap)(geom, centers1, centers2, basis_data, am_data)
##print(grad.shape)
hess = jax.jacfwd(jax.jacfwd(build_overlap))(geom, centers1, centers2, basis_data, am_data)
##print(hess.shape)
#cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(build_overlap)))(geom, centers1, centers2, basis_data, am_data)
#print(cube.shape)

#mints = psi4.core.MintsHelper(basis_set)
#psi_S = np.asarray(onp.asarray(mints.ao_overlap()))
#print(np.allclose(S, psi_S))

