import jax 
from jax.config import config; config.update("jax_enable_x64", True)
config.enable_omnistaging()
import jax.numpy as jnp
import numpy as np
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

def restricted_hartree_fock(geom, basis_name, xyz_path, nuclear_charges, charge, SCF_MAX_ITER=100, return_aux_data=True):
    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2

    # Use local JAX implementation of integrals
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

    # Use psi4 for integrals
    #with open(xyz_path, 'r') as f:
    #    tmp = f.read()
    #molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
    #basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
    #mints = psi4.core.MintsHelper(basis_set)
    #S = jnp.round(jnp.asarray(mints.ao_overlap()), 12)
    #T = jnp.round(jnp.asarray(mints.ao_kinetic()), 12)
    #V = jnp.round(jnp.asarray(mints.ao_potential()), 12)
    #G = jnp.round(jnp.asarray(mints.ao_eri()), 12)


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
    nbf = S.shape[0]
    epsilon = 1e-9 # This epsilon yields the most stable and most precise higher order derivatives.
    fudge = jnp.asarray(np.linspace(0, 1, nbf)) * epsilon
    fudge_factor = jnp.diag(fudge)

    H = T + V
    Enuc = nuclear_repulsion(geom.reshape(-1,3),nuclear_charges)
    D = jnp.zeros_like(H)
    

    @jax.jit
    def rhf_iter(H,A,G,D,Enuc):
        #TODO Einsums + JIT + higher order autodiff = big memory footprint. Use tensordot whenever possible 
        #J = jnp.einsum('pqrs,rs->pq', G, D)
        #K = jnp.einsum('prqs,rs->pq', G, D)
        #F = H + J * 2 - K
        JK = 2 * jnp.tensordot(G, D, axes=[(2,3), (0,1)]) # 2 * J
        JK -= jnp.tensordot(G, D, axes=[(1,3), (0,1)])    # - K
        F = H + JK

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
    while abs(E_scf - E_old) > 1e-12:
        E_old = E_scf * 1
        E_scf, D, C, eps = rhf_iter(H,A,G,D,Enuc)
        iteration += 1
        if iteration == SCF_MAX_ITER:
            break
    print(iteration, " RHF iterations performed")

    # If many orbitals are degenerate, warn that higher order derivatives may be unstable 
    tmp = jnp.round(eps,10)
    ndegen_orbs = jnp.unique(tmp).shape[0] - tmp.shape[0]
    if (ndegen_orbs / nbf) > 0.10:
        print("Hartree-Fock warning: More than 10% of orbitals have degenerate counterparts. Higher order derivatives may be unstable due to eigendecomposition AD rule")

    if not return_aux_data:
        return E_scf
    else:
        return E_scf, C, eps, G

