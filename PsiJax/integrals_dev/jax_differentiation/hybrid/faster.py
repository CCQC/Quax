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
#

# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))

# Get Psi Basis Set and basis set dictionary objects
#basis_name = 'cc-pvdz'
basis_name = 'cc-pvtz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
# Number of basis functions, number of shells
nbf = basis_set.nbf()
print("number of basis functions", nbf)
nshells = len(basis_dict)

def preprocess(geom, basis_dict, nshells):
    segment_id = 0
    basis_data = []
    segment = []
    centers_bra = []
    centers_ket = []
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
            for k in range(exp_combos.shape[0]):
                basis_data.append([exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],am_bra,am_ket])
                centers_bra.append(atom1_idx)
                centers_ket.append(atom2_idx)
                segment.append(segment_id)
            segment_id += 1
    return np.asarray(onp.asarray(basis_data)), np.asarray(onp.asarray(segment)), centers_bra, centers_ket

import time
a = time.time()
print("starting preprocessing")
basis_data, sid, centers1, centers2 = preprocess(geom, basis_dict, nshells)
print("preprocessing done")
b = time.time()
print(b-a)

def build_overlap(geom, centers1, centers2, basis_data, sid):
    centers_bra = np.take(geom, centers1, axis=0)
    centers_ket = np.take(geom, centers2, axis=0)

    #def compute(centers_bra, centers_ket, basis_data):
    def compute(inp):
        centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        val = np.where((bra_am == 0) & (ket_am == 0), np.pad(overlap_ss(*args).reshape(-1), (0,35),constant_values=-100),
              np.where((bra_am == 1) & (ket_am == 0), np.pad(overlap_ps(*args).reshape(-1), (0,33),constant_values=-100),
              np.where((bra_am == 0) & (ket_am == 1), np.pad(overlap_ps(*sgra).reshape(-1), (0,33),constant_values=-100),
              np.where((bra_am == 1) & (ket_am == 1), np.pad(overlap_pp(*args).reshape(-1), (0,27),constant_values=-100),
              np.where((bra_am == 2) & (ket_am == 0), np.pad(overlap_ds(*args).reshape(-1), (0,30),constant_values=-100),
              np.where((bra_am == 0) & (ket_am == 2), np.pad(overlap_ds(*sgra).reshape(-1), (0,30),constant_values=-100),
              np.where((bra_am == 2) & (ket_am == 1), np.pad(overlap_dp(*args).reshape(-1), (0,18),constant_values=-100),
              np.where((bra_am == 1) & (ket_am == 2), np.pad(overlap_dp(*sgra).reshape(-1), (0,18),constant_values=-100),
              np.where((bra_am == 2) & (ket_am == 2), overlap_dd(*args).reshape(-1), np.zeros(36))))))))))
        return val

    #Three different ways TODO this looks better...  
    #vectorized = jax.jit(jax.vmap(compute, (0,0,0)))
    #vectorized = jax.vmap(compute, (0,0,0))
    #tmp_primitives = vectorized(centers_bra,centers_ket,basis_data)
    tmp_primitives = jax.lax.map(compute, (centers_bra, centers_ket, basis_data))
    return tmp_primitives
    #contracted = jax.ops.segment_sum(tmp_primitives, sid)

    #mask = (contracted >= -99)
    # Final primitive values
    #contracted = contracted[mask]
    #return contracted

#result = build_overlap(geom, centers1, centers2, basis_data, sid)
result = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(build_overlap))))(geom, centers1, centers2, basis_data, sid)
#print(result.shape)
#result = build_overlap(geom, centers1, centers2, basis_data, sid)


#print(result)

def new_build_overlap(geom, centers1, centers2, basis_data, sid):
    centers_bra = np.take(geom, centers1, axis=0)
    centers_ket = np.take(geom, centers2, axis=0)

    def compute(centers_bra, centers_ket, basis_data, test):
    #def compute(inp):
    #    centers_bra, centers_ket, basis_data = inp
        Ax, Ay, Az = centers_bra
        Cx, Cy, Cz = centers_ket
        alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = basis_data
        args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
        val = np.where((bra_am == 0) & (ket_am == 0), np.pad(overlap_ss(*args).reshape(-1), (0,35),constant_values=-100),
              np.where((bra_am == 1) & (ket_am == 0), np.pad(overlap_ps(*args).reshape(-1), (0,33),constant_values=-100),
              np.where((bra_am == 0) & (ket_am == 1), np.pad(overlap_ps(*sgra).reshape(-1), (0,33),constant_values=-100),
              np.where((bra_am == 1) & (ket_am == 1), np.pad(overlap_pp(*args).reshape(-1), (0,27),constant_values=-100),
              np.where((bra_am == 2) & (ket_am == 0), np.pad(overlap_ds(*args).reshape(-1), (0,30),constant_values=-100),
              np.where((bra_am == 0) & (ket_am == 2), np.pad(overlap_ds(*sgra).reshape(-1), (0,30),constant_values=-100),
              np.where((bra_am == 2) & (ket_am == 1), np.pad(overlap_dp(*args).reshape(-1), (0,18),constant_values=-100),
              np.where((bra_am == 1) & (ket_am == 2), np.pad(overlap_dp(*sgra).reshape(-1), (0,18),constant_values=-100),
              np.where((bra_am == 2) & (ket_am == 2), overlap_dd(*args).reshape(-1), np.zeros(36))))))))))
        return val[0:test]

    #Three different ways TODO this looks better...  
    vectorized = jax.jit(jax.vmap(jax.jit(compute, static_argnums=(3,)), (0,0,0,None)))
    #vectorized = jax.vmap(compute, (0,0,0))
    tmp_primitives = vectorized(centers_bra,centers_ket,basis_data, 1)
    print(tmp_primitives.shape)
    #tmp_primitives = jax.lax.map(compute, (centers_bra, centers_ket, basis_data))
    #contracted = jax.ops.segment_sum(tmp_primitives, sid)

    #mask = (contracted >= -99)
    # Final primitive values
    #contracted = contracted[mask]
    #return contracted

#result = new_build_overlap(geom, centers1, centers2, basis_data, sid)

