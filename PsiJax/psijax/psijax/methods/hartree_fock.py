import jax 
from jax.config import config; config.update("jax_enable_x64", True)
config.enable_omnistaging()
import jax.numpy as np
import psi4
import numpy as onp
import time
np.set_printoptions(linewidth=500)

from ..integrals import tei

from ..external_integrals import external_tei
from ..external_integrals import external_oei
from ..integrals import oei 
from .energy_utils import nuclear_repulsion, cholesky_orthogonalization
from functools import partial

@partial(jax.jit, static_argnums=(5,))
def rhf_iter(H,A,G,D,Enuc,ndocc,fudge_factor):
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

def restricted_hartree_fock(geom, basis, mints, nuclear_charges, charge, SCF_MAX_ITER=50, return_aux_data=True):
#def restricted_hartree_fock(geom, basis, nuclear_charges, charge, SCF_MAX_ITER=50, return_aux_data=True):
    nelectrons = int(np.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2

    #S, T, V = oei.oei_arrays(geom,basis,nuclear_charges)
    #G = tei.tei_array(geom,basis)
    #S, T, V = oei.oei_arrays(geom.reshape(-1,3),basis,nuclear_charges)
    S = external_oei.psi_overlap(geom,mints=mints) 
    T = external_oei.psi_kinetic(geom,mints=mints) 
    V = external_oei.psi_potential(geom,mints=mints) 
    G = external_tei.psi_tei(geom,mints=mints)

    # Canonical orthogonalization via cholesky decomposition
    A = cholesky_orthogonalization(S)

    # For slightly shifting eigenspectrum of Fp for degenerate eigenvalues 
    # (JAX cannot differentiate degenerate eigenvalue eigh) 
    seed = jax.random.PRNGKey(0)
    epsilon = 1e-9
    fudge = jax.random.uniform(seed, (S.shape[0],), minval=0.1, maxval=1.0) * epsilon
    fudge_factor = np.diag(fudge)

    H = T + V
    Enuc = nuclear_repulsion(geom.reshape(-1,3),nuclear_charges)
    D = np.zeros_like(H)

    iteration = 0
    E_scf = 1.0
    E_old = 0.0
   
    #TODO make epsilon fudge factor relate to energy convergence criteria.
    # Not sure how, but they should probably depend on each other. 
    # TODO maybe noise could be reduced if you subsequently add and subtract fudge factor
    while abs(E_scf - E_old) > 1e-12:
        E_old = E_scf * 1
        E_scf, D, C, eps = rhf_iter(H,A,G,D,Enuc,ndocc,fudge_factor)

        #print("RHF Total Energy:          ", E_scf)
        iteration += 1
        if iteration == SCF_MAX_ITER:
            break
    print("RHF Total Energy:          ", E_scf)
    if not return_aux_data:
        return E_scf
    else:
        return E_scf, C, eps, G

