import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import fori_loop
import psi4

from .ints import compute_f12_oeints, compute_f12_teints
from .energy_utils import nuclear_repulsion, tei_transformation
from .mp2 import restricted_mp2

def restricted_mp2_f12(geom, basis_set, xyz_path, nuclear_charges, charge, options, cabs_space, deriv_order=0):
    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2
    E_mp2, C_obs, eps = restricted_mp2(geom, basis_set, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order, return_aux_data=True)

    print("Running MP2-F12 Computation...")
    cabs_set = cabs_space.basisset()
    C_cabs = jnp.array(cabs_space.C().to_array())
    nobs = C_obs.shape[0]
    nri = C_cabs.shape[0]

    f, fk = form_Fock(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)

    return f

def form_h(geom, basis_set, cabs_set, C_obs, C_cabs, nobs, nri, xyz_path, deriv_order, options):
    h = jnp.empty((nri, nri))

    h_tmp = compute_f12_oeints(geom, basis_set, basis_set, xyz_path, deriv_order, options)
    h_tmp = jnp.einsum('pP,qQ,pq->PQ', C_obs, C_obs, h_tmp, optimize='optimal')
    h = h.at[:nobs, :nobs].set(h_tmp) # <O|O>

    h_tmp = compute_f12_oeints(geom, basis_set, cabs_set, xyz_path, deriv_order, options)
    h_tmp = jnp.einsum('pP,qQ,pq->PQ', C_obs, C_cabs, h_tmp, optimize='optimal')
    h = h.at[:nobs, nobs:nri].set(h_tmp) # <O|C>
    h = h.at[nobs:nri, :nobs].set(jnp.transpose(h_tmp)) # <C|O>

    h_tmp = compute_f12_oeints(geom, cabs_set, cabs_set, xyz_path, deriv_order, options)
    h_tmp = jnp.einsum('pP,qQ,pq->PQ', C_cabs, C_cabs, h_tmp, optimize='optimal')
    h = h.at[nobs:nri, nobs:nri].set(h_tmp) # <C|C>
    del h_tmp

    return h

def form_Fock(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options):
    # OEINTS
    f = form_h(geom, basis_set, cabs_set, C_obs, C_cabs, nobs, nri, xyz_path, deriv_order, options)

    # TEINTS
    G = jnp.empty((nri, nobs, nri, nri))

    G_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "eri", xyz_path, deriv_order, options)
    G_tmp = jnp.einsum('pP,rR,qQ,sS,prqs->PQRS', C_obs, C_obs, C_obs[:, :ndocc], C_obs, G_tmp, optimize='optimal')
    G = G.at[:nobs, :ndocc, :nobs, :nobs].set(G_tmp) # <Oo|OO>

    G_tmp = compute_f12_teints(geom, cabs_set, basis_set, basis_set, basis_set, "eri", xyz_path, deriv_order, options)
    G_tmp = jnp.einsum('pP,rR,qQ,sS,prqs->PQRS', C_cabs, C_obs, C_obs, C_obs, G_tmp, optimize='optimal')
    G = G.at[nobs:nri, :nobs, :nobs, :nobs].set(G_tmp) # <CO|OO>
    G = G.at[:nobs, :nobs, nobs:nri, :nobs].set(jnp.transpose(G_tmp, (2,1,0,3))) # <OO|CO>
    G = G.at[:nobs, :nobs, :nobs, nobs:nri].set(jnp.transpose(G_tmp, (3,2,1,0))) # <OO|OC>

    G_tmp = compute_f12_teints(geom, cabs_set, basis_set, basis_set, cabs_set, "eri", xyz_path, deriv_order, options)
    G_tmp = jnp.einsum('pP,rR,qQ,sS,prqs->PQRS', C_cabs, C_obs, C_obs[:, :ndocc], C_cabs, G_tmp, optimize='optimal')
    G = G.at[nobs:nri, :ndocc, :nobs, nobs:nri].set(G_tmp) # <Co|OC>

    G_tmp = compute_f12_teints(geom, cabs_set, cabs_set, basis_set, basis_set, "eri", xyz_path, deriv_order, options)
    G_tmp = jnp.einsum('pP,rR,qQ,sS,prqs->PQRS', C_cabs, C_cabs, C_obs[:, :ndocc], C_obs, G_tmp, optimize='optimal')
    G = G.at[nobs:nri, :ndocc, nobs:nri, :nobs].set(G_tmp) # <Co|CO>
    del G_tmp

    # Fill Fock Matrix
    f = f.at[:, :].add(2.0 * jnp.einsum('PIQI->PQ', G[:, :ndocc, :, :ndocc], optimize='optimal'))
    fk = f # Fock Matrix without Exchange
    f = f.at[:, :].add(-1.0 * jnp.einsum('PIIQ->PQ', G[:, :ndocc, :ndocc, :], optimize='optimal'))

    return f, fk

#def form_V(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options):
    
