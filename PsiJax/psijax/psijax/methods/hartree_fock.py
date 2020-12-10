import jax 
jax.config.update("jax_enable_x64", True)
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

# Jitting any subfunction of this will double memory use... hrmm
#def tmp(x,y):
#    return jnp.tensordot(x, y, axes=[(0,1),(0,1)])
# Memory efficient JK contraction
#jk_build = jax.vmap(jax.vmap(tmp, in_axes=(0,None)), in_axes=(0,None))
#jk_build = jax.jit(jax.vmap(jax.vmap(tmp, in_axes=(0,None)), in_axes=(0,None)))

# Jitting any subfunction of this will double memory use... hrmm
jk_build = jax.jit(jax.vmap(jax.vmap(lambda x,y: jnp.tensordot(x, y, axes=[(0,1),(0,1)]), in_axes=(0,None)), in_axes=(0,None)))

#j_build = jax.jit(jax.vmap(jax.vmap(lambda x,y: jnp.tensordot(x, y, axes=[(0,1),(0,1)]), in_axes=(0,None)), in_axes=(0,None)))
#k_build = jax.jit(jax.vmap(lambda x,y: jnp.tensordot(x,y, [(0,2),(0,1)]), (0,None)))
#j_build = jax.vmap(jax.vmap(lambda x,y: jnp.tensordot(x, y, axes=[(0,1),(0,1)]), in_axes=(0,None)), in_axes=(0,None))
#k_build = jax.vmap(lambda x,y: jnp.tensordot(x,y, [(0,2),(0,1)]), (0,None))

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
    
    # At high differentiation orders, jitting this increases memory by A LOT, but makes it much faster...
    # N2 cc-pvtz Hartree fock partial quartic, preloaded ints: NO JIT: 3m31s, 5 GB ||  WITH JIT: 1m13s 10.3 GB
    # Hope: recomp of JAX will fix?
    # TODO Can possibly move JK build OUTSIDE of JIT. This is likely where the memory blow up occurs.
    # If you move JK outside of jitted function, 3m37s, 5 GB 
    # No omnistaging, with JIT, new version of JAX

    # No JIT's anywhere, double vmap algo: 2m24s, 4.7 GB

    #@jax.jit
    def rhf_iter(H,A,G,D,Enuc):
        #JK = 2 * jnp.tensordot(G, D, axes=[(2,3), (0,1)]) - jnp.tensordot(G, D, axes=[(1,3), (0,1)]) # 2 * J

        # This causes increased memory use
        #JK = 2 * jnp.tensordot(G, D, axes=[(2,3), (0,1)]) # 2 * J
        #JK -= jnp.tensordot(G, D, axes=[(1,3), (0,1)])  # - K
        #F = H + JK

        #JK = 2 * jk_build(G,D)

        #JK = 2 * jnp.tensordot(G, D, axes=[(2,3), (0,1)]) # 2 * J
        #JK = 2 * jax.lax.map(lambda x: jnp.sum(jnp.dot(x, D)), G.reshape(-1,nbf,nbf)).reshape(nbf,nbf)
        #JK = 2 * jax.lax.map(lambda x: jnp.einsum('qrs,rs->q', x, D), G).reshape(nbf,nbf) # this works.

        # Map over leading axis of TEI's and contract to reduce memory use
        #JK = 2 * jax.lax.map(lambda x: jnp.tensordot(x, D, axes=[(1,2),(0,1)]), G)
        #JK -= jax.lax.map(lambda x: jnp.tensordot(x, D, axes=[(0,2),(0,1)]), G)

        # This part is the only expensive part that benefits from JIT
        JK = 2 * jk_build(G, D)
        JK -= jk_build(G.transpose((0,2,1,3)), D)
        F = H + JK

        E_scf = jnp.einsum('pq,pq->', F + H, D) + Enuc
        Fp = jnp.dot(A.T, jnp.dot(F, A))
        Fp = Fp + fudge_factor
        
        eps, C2 = jnp.linalg.eigh(Fp)
        C = jnp.dot(A,C2)
        Cocc = C[:, :ndocc]
        D = jnp.dot(Cocc, Cocc.T)
        return E_scf, D, C, eps

    #@jax.jit
    def new_rhf_iter(H,A,JK,D,Enuc):
        F = H + JK
        E_scf = jnp.einsum('pq,pq->', F + H, D) + Enuc
        Fp = jnp.dot(A.T, jnp.dot(F, A))
        Fp = Fp + fudge_factor
        eps, C2 = jnp.linalg.eigh(Fp)
        C = jnp.dot(A,C2)
        Cocc = C[:, :ndocc]
        D = jnp.dot(Cocc, Cocc.T)
        return E_scf, D, C, eps

    iteration = 0
    E_scf = 1.0
    E_old = 0.0
    while abs(E_scf - E_old) > 1e-12:
        E_old = E_scf * 1
        #E_scf, D, C, eps = rhf_iter(H,A,G,D,Enuc)

        JK = 2 * jk_build(G, D)
        JK -= jk_build(G.transpose((0,2,1,3)), D)
        E_scf, D, C, eps = new_rhf_iter(H,A,JK,D,Enuc)

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

