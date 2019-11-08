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

    #Ax, Ay, Az = geom[0]
    #Cx, Cy, Cz = geom[1]

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

    tmp_primitives = jax.lax.map(compute, (centers_bra, centers_ket, basis_data))
    contracted = jax.ops.segment_sum(tmp_primitives, sid)

    mask = (contracted >= -99)
    # Final primitive values
    contracted = contracted[mask]
    return contracted

    #print(centers_bra)

print(build_overlap(geom, centers1, centers2, basis_data, sid))


# Confirm gradients work
#overlap_gradient = jax.jacfwd(build_overlap, 0)  
#print(overlap_gradient(geom, centers1, centers2, basis_data, sid))

##print(data)
#print("All primitives")
#print(data.shape)
#print("SID")
##print(sid)
#print(sid.shape)
#
#
#
# Maps over primitive overlap computations (s|s) to (p|p)
#def mapper(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2, bra_am, ket_am):
def mapper(data):
    Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = data
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

##overlap_jax = jax.vmap(mapper, (0,0,0,0,0,0,0,0,0,0,0,0))
##overlap_jax = jax.jit(jax.vmap(mapper, (0,)))
#tmp_primitives = jax.lax.map(mapper, data)
#print(tmp_primitives)
##tmp_primitives = overlap_jax(data[:5])
##tmp_primitives = overlap_jax(data)
#print(tmp_primitives.shape)
#
#contracted = jax.ops.segment_sum(tmp_primitives, sid)
#print(contracted.shape)
#print(contracted)


#
#d = time.time()
#print("time to contract: ",d-c)
#
### remove all the junk from padding:
##mask = (tmp_primitives != -100)
##print(mask)
## Final primitive values
##final = tmp_primitives[mask]
##print(final)
##print(final.shape)
#
##Ax = data[:,0]
##Ay = data[:,1]
##Az = data[:,2]
##Cx = data[:,3]
##Cy = data[:,4]
##Cz = data[:,5]
##alpha_bra = data[:,6]
##alpha_ket = data[:,7]
##c1 = data[:,8]
##c2 = data[:,9]
##bra_am = data[:,10]
##ket_am = data[:,11]
##primitives = overlap_jax(Ax,Ay,Az,Cx,Cy,Cz,alpha_bra,alpha_ket,c1,c2,bra_am,ket_am)
##print(primitives)
#
#
## Use segment sum 
#
### Maps over primitive computations (s|s) to (d|d)
##def mapper(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2, which):
##    args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
##    val = np.where(which == 0, np.pad(overlap_ss(*args).reshape(-1), (0,35),constant_values=-100),
##          np.where(which == 1, np.pad(overlap_ps(*args).reshape(-1), (0,33),constant_values=-100),
##          np.where(which == 2, np.pad(overlap_ds(*args).reshape(-1), (0,30),constant_values=-100),
##          np.where(which == 3, np.pad(overlap_pp(*args).reshape(-1), (0,27),constant_values=-100),
##          np.where(which == 4, np.pad(overlap_dp(*args).reshape(-1), (0,18),constant_values=-100),
##          np.where(which == 5, overlap_dd(*args).reshape(-1), np.zeros(36)))))))
##    return val
#
##dope = jax.vmap(mapper, (0,0,0,0,0,0,0,0,0,0,0))
##dummy = np.repeat(0.1,120000)
##which = np.repeat(np.array([0,1,2,3,4,5]), 20000)
##result = dope(dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, which)
##print(result)
##print(result.shape)
### How to remove all the junk from padding:
##mask = (result != -100)
##print(mask)
### Final primitive values
##final = result[mask].reshape(-1)
##print(final)
##print(final.shape)
#
#
