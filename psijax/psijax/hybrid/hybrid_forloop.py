import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product
np.set_printoptions(linewidth=800, suppress=True)
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
basis_name = 'cc-pvtz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
pprint(basis_dict)
# Number of basis functions, number of shells
nbf = basis_set.nbf()
print("number of basis functions", nbf)
nshells = len(basis_dict)

@jax.jit
def primitive_overlap(args,sgra,am_bra,am_ket):
    K = 36 #NOTE only up to (d|d)
    primitives = np.where((am_bra == 0) & (am_ket == 0), np.pad(overlap_ss(*args).reshape(-1), (0,K-1),constant_values=-100),
                 np.where((am_bra == 1) & (am_ket == 0), np.pad(overlap_ps(*args).reshape(-1), (0,K-3),constant_values=-100),
                 np.where((am_bra == 0) & (am_ket == 1), np.pad(overlap_ps(*sgra).reshape(-1), (0,K-3),constant_values=-100),
                 np.where((am_bra == 1) & (am_ket == 1), np.pad(overlap_pp(*args).reshape(-1), (0,K-9),constant_values=-100),
                 np.where((am_bra == 2) & (am_ket == 0), np.pad(overlap_ds(*args).reshape(-1), (0,K-6),constant_values=-100),
                 np.where((am_bra == 0) & (am_ket == 2), np.pad(overlap_ds(*sgra).reshape(-1), (0,K-6),constant_values=-100),
                 np.where((am_bra == 2) & (am_ket == 1), np.pad(overlap_dp(*args).reshape(-1), (0,K-18),constant_values=-100),
                 np.where((am_bra == 1) & (am_ket == 2), np.pad(overlap_dp(*sgra).reshape(-1), (0,K-18),constant_values=-100),
                 np.where((am_bra == 2) & (am_ket == 2), np.pad(overlap_dd(*args).reshape(-1), (0,K-36),constant_values=-100), np.zeros(K))))))))))
    return primitives

vec_prim_overlap = jax.jit(jax.vmap(primitive_overlap, ((None,None,None,None,None,None,0,0,0,0),(None,None,None,None,None,None,0,0,0,0),None,None)))


def compute_overlap(geom,basis_dict, nbf, nshells):
    # first loop and get geometry data squared away, then do real loop. This is somehow faster
    bra_atoms = []
    ket_atoms = []
    for i in range(nshells):
        atom1 = basis_dict[i]['atom']
        for j in range(nshells):
            atom2 = basis_dict[j]['atom']
            bra_atoms.append(atom1)
            ket_atoms.append(atom2)
    centers_bra = np.take(geom, bra_atoms, axis=0)
    centers_ket = np.take(geom, ket_atoms, axis=0)

    for i in range(nshells):
        c1 =    onp.asarray(basis_dict[i]['coef'])
        exp1 =  onp.asarray(basis_dict[i]['exp'])
        row_idx = basis_dict[i]['idx']
        row_idx_stride = basis_dict[i]['idx_stride']
        am_bra = basis_dict[i]['am'] 
        Ax, Ay, Az = centers_bra[i]
        for j in range(nshells):
            c2 =    onp.asarray(basis_dict[j]['coef'])
            exp2 =  onp.asarray(basis_dict[j]['exp'])
            col_idx = basis_dict[j]['idx']
            col_idx_stride = basis_dict[j]['idx_stride']
            am_ket = basis_dict[j]['am'] 

            exp_combos = old_cartesian_product(exp1,exp2)
            coeff_combos = old_cartesian_product(c1,c2)


            Cx, Cy, Cz = centers_ket[j]
            args = (Ax, Ay, Az, Cx, Cy, Cz, exp_combos[:,0],exp_combos[:,1],coeff_combos[:,0], coeff_combos[:,1])
            sgra = (Cx, Cy, Cz, Ax, Ay, Az, exp_combos[:,1],exp_combos[:,0],coeff_combos[:,1], coeff_combos[:,0])
            primitives = vec_prim_overlap(args, sgra, am_bra, am_ket)


compute_overlap(geom, basis_dict, nbf, nshells)


    
#my_S = build_overlap(geom, basis_dict, nbf, nshells)
#print(my_S)
#
#mints = psi4.core.MintsHelper(basis_set)
#psi_S = np.asarray(onp.asarray(mints.ao_overlap()))
#print(np.allclose(my_S, psi_S))
##print(np.equal(my_S, psi_S))
