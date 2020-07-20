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

#molecule = psi4.geometry("""
#                         0 1
#                         H 0.0 0.0 -0.849220457955
#                         H 0.0 0.0  0.849220457955
#                         H 0.0 0.0  2.000000000000
#                         H 0.0 0.0  3.000000000000
#                         H 0.0 0.0  4.000000000000
#                         H 0.0 0.0  5.000000000000
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

#overlap_funcs = {}
#
#overlap_funcs['00'] = jax.vmap(overlap_ss, (None,None,None,None,None,None,0,0,0,0))
#overlap_funcs['10'] = jax.vmap(overlap_ps, (None,None,None,None,None,None,0,0,0,0))
#overlap_funcs['11'] = jax.vmap(overlap_pp, (None,None,None,None,None,None,0,0,0,0))
#overlap_funcs['20'] = jax.vmap(overlap_ds, (None,None,None,None,None,None,0,0,0,0))
#overlap_funcs['21'] = jax.vmap(overlap_dp, (None,None,None,None,None,None,0,0,0,0))
#overlap_funcs['22'] = jax.vmap(overlap_dd, (None,None,None,None,None,None,0,0,0,0))
#
#overlap_funcs['30'] = jax.vmap(overlap_fs, (None,None,None,None,None,None,0,0,0,0))
#overlap_funcs['31'] = jax.vmap(overlap_fp, (None,None,None,None,None,None,0,0,0,0))
#overlap_funcs['32'] = jax.vmap(overlap_fd, (None,None,None,None,None,None,0,0,0,0))
#overlap_funcs['33'] = jax.vmap(overlap_ff, (None,None,None,None,None,None,0,0,0,0))

@jax.jit
def map_primitive_overlap(inp):
    Ax,Ay,Az,Cx,Cy,Cz,exp1,exp2,c1,c2,am_bra,am_ket = inp
    args = (Ax, Ay, Az, Cx, Cy, Cz,exp1,exp2,c1,c2)
    sgra = (Cx, Cy, Cz, Ax, Ay, Az,exp2,exp1,c2,c1)
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

@jax.jit
def compute_primitive_overlap(Ax,Ay,Az,Cx,Cy,Cz,exp1,exp2,c1,c2,am_bra,am_ket):
    args = (Ax, Ay, Az, Cx, Cy, Cz,exp1,exp2,c1,c2)
    sgra = (Cx, Cy, Cz, Ax, Ay, Az,exp2,exp1,c2,c1)
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

vectorized_primitive_overlaps = jax.jit(jax.vmap(compute_primitive_overlap, (None,None,None,None,None,None,0,0,0,0,None,None)))
vectorized_primitive_overlaps2 = jax.jit(jax.vmap(compute_primitive_overlap, (0,0,0,0,0,0,0,0,0,0,0,0)))


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
    #return primitives
    return primitives

vec_prim_overlap = jax.jit(jax.vmap(primitive_overlap, ((None,None,None,None,None,None,0,0,0,0),(None,None,None,None,None,None,0,0,0,0),None,None)))

@jax.jit
def contracted_overlap(Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2, am_bra, am_ket):
    #Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2 = basis_data
    args = (Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2)
    sgra = (Cx, Cy, Cz, Ax, Ay, Az, e2, e1, c2, c1)
    K = 36 #NOTE only up to (d|d)
    primitives = np.where((am_bra == 0) & (am_ket == 0), np.pad(overlap_ss(*args).reshape(-1), (0,K-1),constant_values=0),
                 np.where((am_bra == 1) & (am_ket == 0), np.pad(overlap_ps(*args).reshape(-1), (0,K-3),constant_values=0),
                 np.where((am_bra == 0) & (am_ket == 1), np.pad(overlap_ps(*sgra).reshape(-1), (0,K-3),constant_values=0),
                 np.where((am_bra == 1) & (am_ket == 1), np.pad(overlap_pp(*args).reshape(-1), (0,K-9),constant_values=0),
                 np.where((am_bra == 2) & (am_ket == 0), np.pad(overlap_ds(*args).reshape(-1), (0,K-6),constant_values=0),
                 np.where((am_bra == 0) & (am_ket == 2), np.pad(overlap_ds(*sgra).reshape(-1), (0,K-6),constant_values=0),
                 np.where((am_bra == 2) & (am_ket == 1), np.pad(overlap_dp(*args).T.reshape(-1), (0,K-18),constant_values=0),
                 np.where((am_bra == 1) & (am_ket == 2), np.pad(overlap_dp(*sgra).reshape(-1), (0,K-18),constant_values=0),
                 np.where((am_bra == 2) & (am_ket == 2), np.pad(overlap_dd(*args).reshape(-1), (0,K-36),constant_values=0), np.zeros(K))))))))))
    return np.sum(primitives, axis=-1)
    #return primitives

vectorized_contracted_overlap = jax.vmap(contracted_overlap, (None, None, None, None, None, None, 0, 0, 0, 0, None, None))

print('computing overlap')
def build_overlap(geom, basis_dict, nbf, nshells):
    '''uses unique functions '''
    S = np.zeros((nbf,nbf))
    for i in range(nshells):
        # Load data for this contracted integral
        c1 =    onp.asarray(basis_dict[i]['coef'])
        exp1 =  onp.asarray(basis_dict[i]['exp'])
        atom1 = basis_dict[i]['atom']
        row_idx = basis_dict[i]['idx']
        row_idx_stride = basis_dict[i]['idx_stride']
        am_bra = basis_dict[i]['am'] 

        for j in range(nshells):
            c2 =    onp.asarray(basis_dict[j]['coef'])
            exp2 =  onp.asarray(basis_dict[j]['exp'])
            atom2 = basis_dict[j]['atom']
            col_idx = basis_dict[j]['idx']
            col_idx_stride = basis_dict[j]['idx_stride']
            am_ket = basis_dict[j]['am'] 
            Ax,Ay,Az = geom[atom1]
            Cx,Cy,Cz = geom[atom2]
            #Ax,Ay,Az = 0.0,0.0,0.0
            #Cx,Cy,Cz = 0.0,0.0,0.0
            # number of primitive components in this shell pair
            size = ((am_bra + 1) * (am_bra + 2) // 2) *  ((am_ket + 1) * (am_ket + 2) // 2) 
    
            exp_combos = old_cartesian_product(exp1,exp2)
            coeff_combos = old_cartesian_product(c1,c2)
            #M = exp_combos.shape[0] #contraction order
            contracted = vectorized_contracted_overlap(Ax,Ay,Az,Cx,Cy,Cz,exp_combos[:,0],exp_combos[:,1],coeff_combos[:,0],coeff_combos[:,1],am_bra,am_ket)
                
            # METHOD 1, vectorize primitives
            #primitives = vectorized_primitive_overlaps(Ax,Ay,Az,Cx,Cy,Cz,exp_combos[:,0],exp_combos[:,1],coeff_combos[:,0],coeff_combos[:,1],am_bra,am_ket)
            #contracted = np.sum(primitives[:,:size], axis=0)
            
            # METHOD 2, naive loop with jitted function calls 
            #for k in range(exp_combos.shape[0]):
            #    args = (Ax, Ay, Az, Cx, Cy, Cz,exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1])
            #    sgra = (Cx, Cy, Cz, Ax, Ay, Az,exp_combos[k,1],exp_combos[k,0],coeff_combos[k,1], coeff_combos[k,0])
            #    primitives = primitive_overlap(args,sgra,am_bra,am_ket)
            #    #contracted = np.sum(primitives[:size], axis=0)

            # METHOD 3, replace innermost loop lax.map, really bad since it compiles over and over again
            #primitives = jax.lax.map(map_primitive_overlap, (np.repeat(Ax, M), np.repeat(Ay, M),np.repeat(Az, M),np.repeat(Cx, M),np.repeat(Cy, M),np.repeat(Cz, M), exp_combos[:,0],exp_combos[:,1],coeff_combos[:,0],coeff_combos[:,1],np.repeat(am_bra,M),np.repeat(am_ket,M)))
    
            # METHOD 4 fully vectorize
            #primitives = vectorized_primitive_overlaps(np.repeat(Ax, M), np.repeat(Ay, M),np.repeat(Az, M),np.repeat(Cx, M),np.repeat(Cy, M),np.repeat(Cz, M), exp_combos[:,0],exp_combos[:,1],coeff_combos[:,0],coeff_combos[:,1],np.repeat(am_bra,M),np.repeat(am_ket,M))

            # METHOD 5 vectorize primitive_overlap(args, sgra, am_bra, am_ket. Slow as mehtod 1
            #args = (Ax, Ay, Az, Cx, Cy, Cz,exp_combos[:,0],exp_combos[:,1],coeff_combos[:,0], coeff_combos[:,1])
            #sgra = (Cx, Cy, Cz, Ax, Ay, Az,exp_combos[:,1],exp_combos[:,0],coeff_combos[:,1], coeff_combos[:,0])
            #primitives = vec_prim_overlap(args, sgra, am_bra, am_ket)
            #print(primitives)


            #print(primitives[:size])
            #if int(bra) < int(ket):
            #    lookup = str(basis_dict[j]['am']) +  str(basis_dict[i]['am'])
            #    primitives = overlap_funcs[lookup](Bx,By,Bz,Ax,Ay,Az,exp_combos[:,1],exp_combos[:,0],coeff_combos[:,1],coeff_combos[:,0])
            #else:
            #    lookup = str(basis_dict[i]['am']) +  str(basis_dict[j]['am'])
            #    primitives = overlap_funcs[lookup](Ax,Ay,Az,Bx,By,Bz,exp_combos[:,0],exp_combos[:,1],coeff_combos[:,0],coeff_combos[:,1])

            ## This fixes shaping error for dp vs pd, etc
            #if int(bra) < int(ket):
            #    contracted = np.sum(primitives, axis=0).reshape(-1)
            #else:
            #    contracted = np.sum(primitives, axis=0).T.reshape(-1)

            row_indices = np.repeat(row_idx, row_idx_stride) + np.arange(row_idx_stride)
            col_indices = np.repeat(col_idx, col_idx_stride) + np.arange(col_idx_stride)
            indices = old_cartesian_product(row_indices,col_indices)
            print(contracted, indices)

            S = jax.ops.index_update(S, (indices[:,0],indices[:,1]), contracted)
    return S

    
my_S = build_overlap(geom, basis_dict, nbf, nshells)
print(my_S)

mints = psi4.core.MintsHelper(basis_set)
psi_S = np.asarray(onp.asarray(mints.ao_overlap()))
print(np.allclose(my_S, psi_S))

#print(np.equal(my_S, psi_S))
