import jax 
from jax.config import config; config.update("jax_enable_x64", True)
config.enable_omnistaging()
import jax.numpy as jnp
import psi4

from ..integrals.basis_utils import build_basis_set

from ..integrals import tei as og_tei
from ..integrals import oei 

from ..external_integrals import overlap
from ..external_integrals import kinetic
from ..external_integrals import potential
from ..external_integrals import tei
#from ..external_integrals import tmp_potential

from ..external_integrals import libint_initialize
from ..external_integrals import libint_finalize

from .energy_utils import nuclear_repulsion, cholesky_orthogonalization
from functools import partial

def restricted_hartree_fock(geom, basis_name, xyz_path, nuclear_charges, charge, SCF_MAX_ITER=50, return_aux_data=True):
    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2

    ## Use local JAX implementation of integrals
    #with open(xyz_path, 'r') as f:
    #    tmp = f.read()
    #molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
    #basis_dict = build_basis_set(molecule, basis_name)
    #S, T, V = oei.oei_arrays(geom.reshape(-1,3),basis_dict,nuclear_charges)
    #G = og_tei.tei_array(geom.reshape(-1,3),basis_dict)

    # Use Libint2 for all integrals 
    libint_initialize(xyz_path, basis_name)
    S = overlap(geom)
    T = kinetic(geom) 
    V = potential(geom) 
    G = tei(geom)
    libint_finalize()

    # TEMP TODO do not use libint for potential
    # Have to build Molecule object and basis dictionary
    #print("Using slow potential integrals")
    #with open(xyz_path, 'r') as f:
    #    tmp = f.read()
    #molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
    #basis_dict = build_basis_set(molecule, basis_name)
    #V = tmp_potential(jnp.asarray(geom.reshape(-1,3)), basis_dict, nuclear_charges)
    ## TEMP TODO

    # Canonical orthogonalization via cholesky decomposition
    A = cholesky_orthogonalization(S)

    # For slightly shifting eigenspectrum of transformed Fock for degenerate eigenvalues 
    # (JAX cannot differentiate degenerate eigenvalue eigh) 
    seed = jax.random.PRNGKey(0)
    epsilon = 1e-9
    fudge = jax.random.uniform(seed, (S.shape[0],), minval=0.1, maxval=1.0) * epsilon
    fudge_factor = jnp.diag(fudge)

    H = T + V
    Enuc = nuclear_repulsion(geom.reshape(-1,3),nuclear_charges)
    D = jnp.zeros_like(H)

    @jax.jit
    def rhf_iter(H,A,G,D,Enuc):
        J = jnp.einsum('pqrs,rs->pq', G, D)
        K = jnp.einsum('prqs,rs->pq', G, D)
        F = H + J * 2 - K
        E_scf = jnp.einsum('pq,pq->', F + H, D) + Enuc
        Fp = jnp.linalg.multi_dot((A.T, F, A))
        Fp = Fp + fudge_factor
        
        eps, C2 = jnp.linalg.eigh(Fp)
        C = jnp.dot(A,C2)
        Cocc = C[:, :ndocc]
        D = jnp.einsum('pi,qi->pq', Cocc, Cocc)
        return E_scf, D, C, eps 

    iteration = 0
    E_scf = 1.0
    E_old = 0.0
    #TODO make epsilon fudge factor relate to energy convergence criteria.
    #Not sure how, but they should probably depend on each other. 
    #TODO maybe noise could be reduced if you subsequently add and subtract fudge factor
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

