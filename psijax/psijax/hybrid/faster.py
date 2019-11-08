import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product
np.set_printoptions(linewidth=800, suppress=True, threshold=3000)
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
basis_name = 'cc-pvdz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
pprint(basis_dict)
# Number of basis functions, number of shells
nbf = basis_set.nbf()
print("number of basis functions", nbf)
nshells = len(basis_dict)

def preprocess(geom, basis_dict, nshells):
    segment_id = 0
    data = []
    segment = []
    for i in range(nshells):
        for j in range(nshells):
            # Load data for this contracted integral
            c1 =    np.asarray(basis_dict[i]['coef'])
            c2 =    np.asarray(basis_dict[j]['coef'])
            exp1 =  np.asarray(basis_dict[i]['exp'])
            exp2 =  np.asarray(basis_dict[j]['exp'])
            atom1 = basis_dict[i]['atom']
            atom2 = basis_dict[j]['atom']
            #row_idx = basis_dict[i]['idx']
            #col_idx = basis_dict[j]['idx']
            #row_idx_stride = basis_dict[i]['idx_stride']
            #col_idx_stride = basis_dict[j]['idx_stride']
            Ax,Ay,Az = geom[atom1]
            Bx,By,Bz = geom[atom2]
            bra = basis_dict[i]['am']
            ket = basis_dict[j]['am']
            
            exp_combos = old_cartesian_product(exp1,exp2)
            coeff_combos = old_cartesian_product(c1,c2)
            for k in range(exp_combos.shape[0]):
                data.append(np.array([Ax,Ay,Az,Bx,By,Bz,exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],int(bra),int(ket)]))
                segment.append(segment_id)
            segment_id += 1
    return np.asarray(data), np.asarray(segment)

data, sid = preprocess(geom, basis_dict, nshells)

#print(data)
print("All primitives")
print(data.shape)
print("SID")
#print(sid)
print(sid.shape)

# Maps over primitive overlap computations (s|s) to (p|p)
#def mapper(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2, bra_am, ket_am):
def mapper(data):
    Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2, bra_am, ket_am = data
    args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    rev_args = (Cx, Cy, Cz, Ax, Ay, Az, alpha_ket, alpha_bra, c2, c1)
    val = np.where((bra_am == 0) & (ket_am == 0), np.pad(overlap_ss(*args).reshape(-1), (0,8),constant_values=-100),
          np.where((bra_am == 1) & (ket_am == 0), np.pad(overlap_ps(*args).reshape(-1), (0,6),constant_values=-100),
          np.where((bra_am == 0) & (ket_am == 1), np.pad(overlap_ps(*rev_args).reshape(-1), (0,6),constant_values=-100),
          np.where((bra_am == 1) & (ket_am == 1), overlap_pp(*args).reshape(-1), np.zeros(9))))) 
    return val

#overlap_jax = jax.vmap(mapper, (0,0,0,0,0,0,0,0,0,0,0,0))
overlap_jax = jax.vmap(mapper, (0,))
tmp_primitives = overlap_jax(data)
#print(tmp_primitives.shape)


contracted = jax.ops.segment_sum(tmp_primitives, sid)
print(contracted.shape)
print(contracted)
## remove all the junk from padding:
#mask = (tmp_primitives != -100)
#print(mask)
# Final primitive values
#final = tmp_primitives[mask]
#print(final)
#print(final.shape)

#Ax = data[:,0]
#Ay = data[:,1]
#Az = data[:,2]
#Cx = data[:,3]
#Cy = data[:,4]
#Cz = data[:,5]
#alpha_bra = data[:,6]
#alpha_ket = data[:,7]
#c1 = data[:,8]
#c2 = data[:,9]
#bra_am = data[:,10]
#ket_am = data[:,11]
#primitives = overlap_jax(Ax,Ay,Az,Cx,Cy,Cz,alpha_bra,alpha_ket,c1,c2,bra_am,ket_am)
#print(primitives)


# Use segment sum 

## Maps over primitive computations (s|s) to (d|d)
#def mapper(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2, which):
#    args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
#    val = np.where(which == 0, np.pad(overlap_ss(*args).reshape(-1), (0,35),constant_values=-100),
#          np.where(which == 1, np.pad(overlap_ps(*args).reshape(-1), (0,33),constant_values=-100),
#          np.where(which == 2, np.pad(overlap_ds(*args).reshape(-1), (0,30),constant_values=-100),
#          np.where(which == 3, np.pad(overlap_pp(*args).reshape(-1), (0,27),constant_values=-100),
#          np.where(which == 4, np.pad(overlap_dp(*args).reshape(-1), (0,18),constant_values=-100),
#          np.where(which == 5, overlap_dd(*args).reshape(-1), np.zeros(36)))))))
#    return val

#dope = jax.vmap(mapper, (0,0,0,0,0,0,0,0,0,0,0))
#dummy = np.repeat(0.1,120000)
#which = np.repeat(np.array([0,1,2,3,4,5]), 20000)
#result = dope(dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, which)
#print(result)
#print(result.shape)
## How to remove all the junk from padding:
#mask = (result != -100)
#print(mask)
## Final primitive values
#final = result[mask].reshape(-1)
#print(final)
#print(final.shape)


