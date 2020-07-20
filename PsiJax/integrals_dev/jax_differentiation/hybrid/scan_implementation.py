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
#                         units bohr
#                         """)

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
                         units bohr
                         """)



# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))

# Get Psi Basis Set and basis set dictionary objects
basis_name = 'cc-pvdz'
#basis_name = 'cc-pvtz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
# Number of basis functions, number of shells
nbf = basis_set.nbf()
print("number of basis functions", nbf)
nshells = len(basis_dict)

def preprocess(geom, basis_dict, nshells):
    segment_id = 0
    primitive_index = 0
    basis_data = []
    segment = []
    centers_bra = []
    centers_ket = []
    sizes = []
    indices = []
    #TODO highest size
    K = 36
    for i in range(nshells):
        c1 =    onp.asarray(basis_dict[i]['coef'])
        exp1 =  onp.asarray(basis_dict[i]['exp'])
        atom1_idx = basis_dict[i]['atom']
        am_bra = basis_dict[i]['am']
        for j in range(nshells):
            c2 =    onp.asarray(basis_dict[j]['coef'])
            exp2 =  onp.asarray(basis_dict[j]['exp'])
            atom2_idx = basis_dict[j]['atom']
            am_ket = basis_dict[j]['am']
            exp_combos = old_cartesian_product(exp1,exp2)
            coeff_combos = old_cartesian_product(c1,c2)
            size = ((am_bra + 1) * (am_bra + 2) // 2) *  ((am_ket + 1) * (am_ket + 2) // 2) 
            identifiers = onp.arange(size) + segment_id
            for k in range(exp_combos.shape[0]):
                basis_data.append([exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],am_bra,am_ket])
                centers_bra.append(atom1_idx)
                centers_ket.append(atom2_idx)
                sizes.append(size)
                # Every primitive component needs a unique segment ID
                for a in identifiers:
                    segment.append(a)

                start = primitive_index
                stop = primitive_index + size
                indx = onp.pad(onp.arange(start, stop), (0,K-size), constant_values=-1)
                indices.append(indx)
                primitive_index += size

            segment_id += 1 * identifiers.shape[0]
    return np.asarray(onp.asarray(basis_data)), np.asarray(onp.asarray(segment)), centers_bra, centers_ket, np.asarray(onp.asarray(indices)), np.asarray(onp.asarray(sizes))

import time
a = time.time()
print("starting preprocessing")
basis_data, sid, centers1, centers2, update_indices, sizes = preprocess(geom, basis_dict, nshells)

print("preprocessing done")
b = time.time()
print(b-a)

def overlap_scan(geom, centers1, centers2, basis_data, update_indices, sizes, sid):
    centers_bra = np.take(geom, centers1, axis=0)
    centers_ket = np.take(geom, centers2, axis=0)

    # Scan over range(primitive) indices
    # Carry an overlap vector and a slice vector

    #@jax.jit
    def compute(carry, i):
        overlap, basis_data, update_indices, sizes = carry
        Ax, Ay, Az = centers_bra[i]
        Cx, Cy, Cz = centers_ket[i]
        #TEMP TODO see if this works first
        #Ax, Ay, Az = 0.0,0.0,0.0
        #Cx, Cy, Cz = 0.0,0.0,0.0
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data[i]
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        K = 36
        val = np.where((bra_am == 0) & (ket_am == 0), np.pad(overlap_ss(*args).reshape(-1), (0,K-1),constant_values=-100),
              np.where((bra_am == 1) & (ket_am == 0), np.pad(overlap_ps(*args).reshape(-1), (0,K-3),constant_values=-100),
              np.where((bra_am == 0) & (ket_am == 1), np.pad(overlap_ps(*sgra).reshape(-1), (0,K-3),constant_values=-100),
              np.where((bra_am == 1) & (ket_am == 1), np.pad(overlap_pp(*args).reshape(-1), (0,K-9),constant_values=-100),
              np.where((bra_am == 2) & (ket_am == 0), np.pad(overlap_ds(*args).reshape(-1), (0,K-6),constant_values=-100),
              np.where((bra_am == 0) & (ket_am == 2), np.pad(overlap_ds(*sgra).reshape(-1), (0,K-6),constant_values=-100),
              np.where((bra_am == 2) & (ket_am == 1), np.pad(overlap_dp(*args).reshape(-1), (0,K-18),constant_values=-100),
              np.where((bra_am == 1) & (ket_am == 2), np.pad(overlap_dp(*sgra).reshape(-1), (0,K-18),constant_values=-100),
              np.where((bra_am == 2) & (ket_am == 2), np.pad(overlap_dd(*args).reshape(-1), (0,K-36),constant_values=-100), np.zeros(K))))))))))


        indx = update_indices[i]
        # indx will always be of the form [val1, val2, val3, .... -1] for all padded -100 values
        # This approach 
        overlap = jax.ops.index_update(overlap, indx, val)
        new_carry = (overlap, basis_data, update_indices, sizes)
        return new_carry, 0
    
    n_prim = np.sum(sizes)
    #overlap = np.zeros(int(n_prim))
    overlap = np.zeros((73984)) #TMP TODO
    indices = np.arange(sizes.shape[0])
    carry, junk = jax.lax.scan(compute, (overlap, basis_data, update_indices, sizes), indices)
    primitives = carry[0]
    print(primitives.shape)
    return primitives


    # THIS IS CORRECT, but wherer to put it?
    #contracted = jax.ops.segment_sum(primitives, sid) 
    #print(contracted)

    #print(contracted.reshape(30,30))

#    return primitives


overlap_scan(geom, centers1, centers2, basis_data, update_indices, sizes, sid)
jax.jacfwd(jax.jacfwd(overlap_scan))(geom, centers1, centers2, basis_data, update_indices, sizes, sid)
#overlap_scan(geom, centers1, centers2, basis_data, update_indices, sizes, sid)

    #for i in range(basis_data.shape[0]):
    #    val = compute(centers_bra[i], centers_ket[i], basis_data[i])
    #    alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data[i]
    #    size = ((bra_am + 1) * (bra_am + 2) // 2) * ((ket_am + 1) * (ket_am + 2) // 2)
    #    result = val[:size.astype(int)]
        #print(val[:size.astype(int)])
        #overlap.append(val[:size])

        #size = ((bra_am + 1) * (bra_am + 2) // 2) *  ((ket_am + 1) * (ket_am + 2) // 2) 
        #return np.take(val, np.arange(size))
        #size = ((bra_am + 1) * (bra_am + 2) // 2) *  ((ket_am + 1) * (ket_am + 2) // 2) 
        #return val[bra_am.astype(int)]#np.take(val, np.arange(bra_am))
        #return val[:size.astype(int)]#np.take(val, np.arange(bra_am))

#    #Three different ways TODO TEST this looks better... test TODO
#    vectorized = jax.jit(jax.vmap(compute, (0,0,0)))
#    #vectorized = jax.vmap(compute, (0,0,0))
#    tmp_primitives = vectorized(centers_bra,centers_ket,basis_data)
#
#    #tmp_primitives = jax.lax.map(compute, (centers_bra, centers_ket, basis_data))
#    contracted = jax.ops.segment_sum(tmp_primitives, sid)
#
#    mask = (contracted >= -99)
#    # Final primitive values
#    contracted = contracted[mask]
#    return contracted




