import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
import psi4
import numpy as onp
from tei import tei_array 
from oei import oei_arrays
from energy_utils import nuclear_repulsion, cholesky_orthogonalization

def restricted_hartree_fock(geom, basis, nuclear_charges, charge, SCF_MAX_ITER=30, return_mo_data=True):
    nelectrons = int(np.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2

    S, T, V = oei_arrays(geom,basis,nuclear_charges)
    # Canonical orthogonalization via cholesky decomposition
    A = cholesky_orthogonalization(S)
    G = tei_array(geom,basis)

    H = T + V
    Enuc = nuclear_repulsion(geom,nuclear_charges)
    D = np.zeros_like(H)

    iteration = 0
    E_scf = 1.0
    E_old = 0.0
    while abs(E_scf - E_old) > 1e-12:
        E_old = E_scf * 1
        J = np.einsum('pqrs,rs->pq', G, D)
        K = np.einsum('prqs,rs->pq', G, D)
        F = H + J * 2 - K
        E_scf = np.einsum('pq,pq->', F + H, D) + Enuc
        Fp = np.linalg.multi_dot((A.T, F, A))
        # Slightly shift eigenspectrum of Fp for degenerate eigenvalues 
        # (JAX cannot differentiate degenerate eigenvalue eigh) 
        # TODO are there consequences to doing this for correlated methods?
        seed = jax.random.PRNGKey(0)
        eps = 1e-12
        fudge = jax.random.uniform(seed, (Fp.shape[0],)) * eps
        Fp = Fp + np.diag(fudge)

        eps, C2 = np.linalg.eigh(Fp)
        C = np.dot(A,C2)
        Cocc = C[:, :ndocc]
        D = np.einsum('pi,qi->pq', Cocc, Cocc)
        iteration += 1
        if iteration == SCF_MAX_ITER:
            break
    if not return_mo_data:
        return E_scf
    else:
        return E_scf, C, eps, G
    

    
