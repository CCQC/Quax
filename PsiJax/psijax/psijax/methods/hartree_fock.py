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

def restricted_hartree_fock(geom, basis_name, xyz_path, nuclear_charges, charge, SCF_MAX_ITER=50, return_aux_data=True):
    nelectrons = int(np.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2

    # Requires basis_dict, buildable from basis dict and psi4 molecule
    #basis_dict = build_basis_set(molecule, basis)
    #S, T, V = oei.oei_arrays(geom,basis_dict,nuclear_charges)
    #G = tei.tei_array(geom,basis_dict)

    # Use Libint2 directly
    external_oei.libint_init(xyz_path, basis_name)
    S = external_oei.psi_overlap(geom)
    T = external_oei.psi_kinetic(geom) 
    V = external_oei.psi_potential(geom) 
    G = external_tei.psi_tei(geom)
    external_oei.libint_finalize()

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
    #TODO make epsilon fudge factor relate to energy convergence criteria.
    # Not sure how, but they should probably depend on each other. 
    # TODO maybe noise could be reduced if you subsequently add and subtract fudge factor
    while abs(E_scf - E_old) > 1e-12:
        E_old = E_scf * 1
        E_scf, D, C, eps = rhf_iter(H,A,G,D,Enuc)
        iteration += 1
        if iteration == SCF_MAX_ITER:
            break
    print(iteration, " RHF iterations performed")
    #print("RHF Total Energy:          ", E_scf)
    if not return_aux_data:
        return E_scf
    else:
        return E_scf, C, eps, G

