import jax 
from jax.config import config; config.update("jax_enable_x64", True)
config.enable_omnistaging()
import jax.numpy as np
import psi4
import numpy as onp

import tei
import oei 
from energy_utils import nuclear_repulsion, cholesky_orthogonalization
from basis_utils import build_basis_set



def preamble(geom,basis,nuclear_charges,charge):
    STV = oei.oei_arrays(geom,basis,nuclear_charges)
    G = tei.tei_array(geom,basis)
    Enuc = nuclear_repulsion(geom,nuclear_charges)
    return STV, G, Enuc

def hf(STV,G,Enuc): 
    S = STV[0]
    T = STV[1]
    V = STV[2]

    # Hardcoded
    ndocc = 1
    SCF_MAX_ITER=10
    # For slightly shifting eigenspectrum of Fp for degenerate eigenvalues 
    # (JAX cannot differentiate degenerate eigenvalue eigh) 
    seed = jax.random.PRNGKey(0)
    epsilon = 1e-9
    fudge = jax.random.uniform(seed, (S.shape[0],), minval=0.1, maxval=1.0) * epsilon
    fudge_factor = np.diag(fudge)


    A = cholesky_orthogonalization(S)
    H = T + V
    D = np.zeros_like(H)

    @jax.jit
    def rhf_iter(H,A,G,D,Enuc):
        J = np.einsum('pqrs,rs->pq', G, D)
        K = np.einsum('prqs,rs->pq', G, D)
        F = H + J * 2 - K
        E_scf = np.einsum('pq,pq->', F + H, D) + Enuc
        Fp = np.linalg.multi_dot((A.T, F, A))
        Fp = Fp + fudge_factor
        
        eps, C2 = np.linalg.eigh(Fp)
        C = np.dot(A,C2)
        Cocc = C[:, :ndocc]
        D = np.einsum('pi,qi->pq', Cocc, Cocc)
        return E_scf, D, C, eps 

    iteration = 0
    E_scf = 1.0
    E_old = 0.0
    while abs(E_scf - E_old) > 1e-12:
        E_old = E_scf * 1
        E_scf, D, C, eps = rhf_iter(H,A,G,D,Enuc)
        iteration += 1
        if iteration == SCF_MAX_ITER:
            break
    return E_scf


molecule = psi4.geometry("""
                         0 1
                         N 0.0 0.0 -0.8
                         N 0.0 0.0  0.8
                         units bohr
                         """)

basis_name = 'cc-pvdz'
basis_dict = build_basis_set(molecule, basis_name)
geom = np.asarray(onp.asarray(molecule.geometry()))
mult = molecule.multiplicity()
charge = molecule.molecular_charge()
nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])

STV, G, Enuc = preamble(geom,basis_dict,nuclear_charges,charge)
#E_scf = hf(STV,G,Enuc)
#print(E_scf)

# Compute dInts/dGeom
dSTV_dgeom, dG_dgeom, dEnuc_dgeom = jax.jacfwd(preamble, (0))(geom,basis_dict,nuclear_charges,charge)
# Compute dE/dInts
dE_dSTV, dE_dG, dE_dEnuc = jax.jacfwd(hf, (0,1,2))(STV,G,Enuc)
print("dSTV_dgeom  ",dSTV_dgeom.shape)      #(3, 2, 2, 2, 3)         
print("dE_dSTV     ",dE_dSTV.shape)         #(3, 2, 2)
print("dG_dgeom    ",dG_dgeom.shape)        #(2, 2, 2, 2, 2, 3)
print("dE_dG       ",dE_dG.shape)           #(2, 2, 2, 2)
print("dEnuc_dgeom ",dEnuc_dgeom.shape)     #(2, 3)
print("dE_dEnuc    ",dE_dEnuc.shape)        #()                 
dHS =  np.einsum('ijklm,ijk->lm', dSTV_dgeom, dE_dSTV)
dG = np.einsum('ijklmn,ijkl->mn', dG_dgeom, dE_dG)
print("Nuclear gradient from decomposition")
print(dHS + dG + dEnuc_dgeom)


