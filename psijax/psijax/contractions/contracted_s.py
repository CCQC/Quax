import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)

def cartesian_product(*arrays):
    '''Generalized cartesian product of any number of arrays'''
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * np.dot(A-C, A-C)) / (aa + bb)))
    return ss

# Vectorized version of overlap_ss
vectorized_overlap_ss = jax.jit(jax.vmap(overlap_ss, (None,None,None,None,None,None,0,0)))


from basis import basis_dict,geom
nshells = len(basis_dict)
S = np.zeros((nshells,nshells))

for i in range(nshells):
    for j in range(nshells):
        # Load data for this contracted integral
        c1 =    np.asarray(basis_dict[i]['coef'])
        c2 =    np.asarray(basis_dict[j]['coef'])
        exp1 =  np.asarray(basis_dict[i]['exp'])
        exp2 =  np.asarray(basis_dict[j]['exp'])
        atom1 = basis_dict[i]['atom']
        atom2 = basis_dict[j]['atom']
        Ax,Ay,Az = geom[atom1]
        Bx,By,Bz = geom[atom2]

        # Expand exponent data to compute all primitive combinations with vectorized overlap function
        exp_combos = cartesian_product(exp1,exp2)
        primitives = vectorized_overlap_ss(Ax,Ay,Az,Bx,By,Bz,exp_combos[:,0],exp_combos[:,1])
        # Build coefficients products for contraction
        coefficients = np.einsum('i,j->ij', c1, c2).flatten()
        # Contract
        result = np.sum(primitives * coefficients)
        S = jax.ops.index_update(S, jax.ops.index[i,j], result)

print(S)

