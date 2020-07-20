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
basis_name = 'cc-pvdz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)

# hack to make big basis but small system
for key in basis_dict:
    basis_dict[key]['atom'] = 0
molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         units bohr
                         """)

# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))

# Number of basis functions, number of shells
pprint(basis_dict)
nbf = basis_set.nbf()
nshells = len(basis_dict)

# Computes a single primitive overlap
@jax.jit
def primitive(Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2, am):
    '''Geometry parameters, exponents, coefficients, angular momentum index'''
    args = (Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2)
    sgra = (Cx, Cy, Cz, Ax, Ay, Az, e2, e1, c2, c1)
    primitive =  np.where(am ==  0, overlap_ss(*args),
                 np.where(am ==  1, overlap_ps(*args)[0],
                 np.where(am ==  2, overlap_ps(*args)[1],
                 np.where(am ==  3, overlap_ps(*args)[2],
                 np.where(am ==  4, overlap_ps(*sgra)[0],
                 np.where(am ==  5, overlap_ps(*sgra)[1],
                 np.where(am ==  6, overlap_ps(*sgra)[2],
                 np.where(am ==  7, overlap_pp(*args)[0,0],
                 np.where(am ==  8, overlap_pp(*args)[0,1],
                 np.where(am ==  9, overlap_pp(*args)[0,2],
                 np.where(am == 10, overlap_pp(*args)[1,0],
                 np.where(am == 11, overlap_pp(*args)[1,1],
                 np.where(am == 12, overlap_pp(*args)[1,2],
                 np.where(am == 13, overlap_pp(*args)[2,0],
                 np.where(am == 14, overlap_pp(*args)[2,1],
                 np.where(am == 15, overlap_pp(*args)[2,2],0.0))))))))))))))))
    return primitive

# Computes multiple primitives in a contracted shell (same geometries, vectors of exp/coef with different values, same shape; same angular momentum)
vectorized_primitive = jax.vmap(primitive, (None,None,None,None,None,None,0,0,0,0,None))

@jax.jit
def contraction(Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2, am):
    primitives = vectorized_primitive(Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2, am)
    return np.sum(primitives)

#overlap = []
#
#for i in range(nshells):
#    c1 =    onp.asarray(basis_dict[i]['coef'])
#    exp1 =  onp.asarray(basis_dict[i]['exp'])
#    atom1_idx = basis_dict[i]['atom']
#    bra_am = basis_dict[i]['am']
#    for j in range(nshells):
#        c2 =    onp.asarray(basis_dict[j]['coef'])
#        exp2 =  onp.asarray(basis_dict[j]['exp'])
#        atom2_idx = basis_dict[j]['atom']
#        ket_am = basis_dict[j]['am']
#
#        exp_combos = old_cartesian_product(exp1,exp2)
#        coeff_combos = old_cartesian_product(c1,c2)
#
#        size = ((bra_am + 1) * (bra_am + 2) // 2) * ((ket_am + 1) * (ket_am + 2) // 2) 
#        Ax, Ay, Az = np.take(geom,atom1_idx,axis=0)
#        Cx, Cy, Cz = np.take(geom,atom2_idx,axis=0)
#
#        #Ax,Ay,Az = geom[atom1_idx]
#        #Cx,Cy,Cz = geom[atom2_idx]
#
#        if bra_am == 0 and ket_am == 0: am=0
#        if bra_am == 1 and ket_am == 0: am=1
#        if bra_am == 0 and ket_am == 1: am=4
#        if bra_am == 1 and ket_am == 1: am=7
#
#        for component in range(size):
#            integral = contraction(Ax, Ay, Az, Cx, Cy, Cz, exp_combos[:,0], exp_combos[:,1], coeff_combos[:,0], coeff_combos[:,1], am)
#            overlap.append(integral)
#            am += 1
    

def preprocess(geom, basis_dict, nshells):
    basis_data = []
    centers = []
    for i in range(nshells):
        c1 =    onp.asarray(basis_dict[i]['coef'])
        exp1 =  onp.asarray(basis_dict[i]['exp'])
        atom1_idx = basis_dict[i]['atom']
        bra_am = basis_dict[i]['am']
        for j in range(nshells):
            c2 =    onp.asarray(basis_dict[j]['coef'])
            exp2 =  onp.asarray(basis_dict[j]['exp'])
            atom2_idx = basis_dict[j]['atom']
            ket_am = basis_dict[j]['am']
    
            exp_combos = old_cartesian_product(exp1,exp2)
            coeff_combos = old_cartesian_product(c1,c2)
            size = ((bra_am + 1) * (bra_am + 2) // 2) * ((ket_am + 1) * (ket_am + 2) // 2) 

            if bra_am == 0 and ket_am == 0: am=0
            if bra_am == 1 and ket_am == 0: am=1
            if bra_am == 0 and ket_am == 1: am=4
            if bra_am == 1 and ket_am == 1: am=7

            for component in range(size):
                #integral = contraction(Ax, Ay, Az, Cx, Cy, Cz, exp_combos[:,0], exp_combos[:,1], coeff_combos[:,0], coeff_combos[:,1], am)
                basis_data.append([exp_combos[:,0], exp_combos[:,1], coeff_combos[:,0], coeff_combos[:,1], am])
                centers.append([atom1_idx, atom2_idx])
                am += 1

    return basis_data, np.asarray(onp.asarray(centers))

basis_data, centers = preprocess(geom, basis_dict, nshells)
print(len(basis_data))
print(centers.shape)

def build_overlap(geom, centers, basis_data):
    centers_bra = np.take(geom, centers[:,0], axis=0) 
    centers_ket = np.take(geom, centers[:,1], axis=0)
    overlap = []
    for i in range(len(basis_data)):
        exp1,exp2,c1,c2,am = basis_data[i]
        Ax, Ay, Az = centers_bra[i]
        Cx, Cy, Cz = centers_ket[i]
        integral = contraction(Ax, Ay, Az, Cx, Cy, Cz, exp1,exp2,c1,c2, am)
        overlap.append(integral)
    return np.array(overlap)

# 10 basis functions
#S = build_overlap(geom,centers,basis_data)                                                    # 0:03.49elapsed 119%CPU (0avgtext+0avgdata 229156maxresident)k
#grad = jax.jacfwd(build_overlap)(geom,centers,basis_data)                                     # 0:06.70elapsed 111%CPU (0avgtext+0avgdata 250964maxresident)k
#hess = jax.jacfwd(jax.jacfwd(build_overlap))(geom,centers,basis_data)                         # 0:13.67elapsed 105%CPU (0avgtext+0avgdata 288816maxresident)k
#cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(build_overlap)))(geom,centers,basis_data)             # 0:32.74elapsed 102%CPU (0avgtext+0avgdata 386228maxresident)k
#quar = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(build_overlap))))(geom,centers,basis_data) # 1:23.61elapsed 101%CPU (0avgtext+0avgdata 664232maxresident)k

# 100 basis functions
#S = build_overlap(geom,centers,basis_data)                                                    # 0:22.72elapsed 158%CPU (0avgtext+0avgdata 267064maxresident)k
#grad = jax.jacfwd(build_overlap)(geom,centers,basis_data)                                     # 0:51.87elapsed 132%CPU (0avgtext+0avgdata 328716maxresident)k
#hess = jax.jacfwd(jax.jacfwd(build_overlap))(geom,centers,basis_data)                         # 1:16.01elapsed 128%CPU (0avgtext+0avgdata 438692maxresident)k
#cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(build_overlap)))(geom,centers,basis_data)             # 1:53.11elapsed 124%CPU (0avgtext+0avgdata 744012maxresident)k
#quar = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(build_overlap))))(geom,centers,basis_data) # 3:10.26elapsed 121%CPU (0avgtext+0avgdata 1974316maxresident)k
#print(quar.shape)


