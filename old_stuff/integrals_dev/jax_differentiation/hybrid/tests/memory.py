import psi4
import jax.numpy as np
import jax
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)

def cartesian_product(*arrays):
    '''Find all indices of ERI tensor given 4 arrays 
       (np.arange(nbf), np.arange(nbf), np.arange(nbf), np.arange(nbf)) '''
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns a (s|s) overlap integral
    """
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    ss = ((np.pi / (alpha_bra + alpha_ket))**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(A-C, A-C)) / (alpha_bra + alpha_ket)))
    return ss * c1 * c2

def oei(geom,basis):
    nbf = basis.shape[0]
    nbf_per_atom = int(nbf / 2)
    centers = np.repeat(geom, nbf_per_atom, axis=0) 
    indices = cartesian_product(np.arange(nbf),np.arange(nbf))
    def compute_overlap(idx):
        i,j = idx
        Ax, Ay, Az = centers[i]
        Cx, Cy, Cz = centers[j]
        oei = overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, basis[i], basis[j], basis[i], basis[j])
        return oei
    overlap = jax.lax.map(compute_overlap, indices)
    print(overlap)
    return overlap

geom = np.array([0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]).reshape(-1,3)
atom1_basis = np.repeat(np.array([0.5, 0.4, 0.3, 0.2]),8)
atom2_basis = np.repeat(np.array([0.5, 0.4, 0.3, 0.2]),7)
basis = np.concatenate((atom1_basis, atom2_basis))
print(basis.shape)
nbf_per_atom = np.array([atom1_basis.shape[0],atom2_basis.shape[0]])

S = oei(geom,basis)
print(S.shape)

quar = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(oei))))(geom, basis)
print(quar.shape)




