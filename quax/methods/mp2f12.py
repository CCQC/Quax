import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import fori_loop
import psi4

from .ints import compute_f12_oeints, compute_f12_teints
from .energy_utils import nuclear_repulsion, tei_transformation
from .mp2 import restricted_mp2

def restricted_mp2_f12(geom, basis_name, xyz_path, nuclear_charges, charge, options, cabs_space, deriv_order=0):
    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2
    E_mp2, C_obs, eps = restricted_mp2(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order, return_aux_data=True)
    cabs_name = cabs_space.name()
    C_cabs = jnp.array(cabs_space.C().to_array())

    f, fk = form_Fock(geom, basis_name, cabs_name, C_obs, C_cabs, ndocc, xyz_path, deriv_order, options)

    return f

def form_h(geom, basis_name, cabs_name, C_obs, C_cabs, xyz_path, deriv_order, options):
    nobs = C_obs.shape[0]
    nri = C_cabs.shape[0]

    h = jnp.empty((nri, nri))

    # <O|O>
    h_tmp = compute_f12_oeints(geom, basis_name, basis_name, xyz_path, deriv_order, options)
    h_tmp = jnp.einsum('pP,qQ,pq->PQ', C_obs, C_obs, h_tmp, optimize='optimal')
    h = h.at[:nobs, :nobs].set(h_tmp)

    # <O|C> and <C|O>
    h_tmp = compute_f12_oeints(geom, basis_name, cabs_name, xyz_path, deriv_order, options)
    h_tmp = jnp.einsum('pP,qQ,pq->PQ', C_obs, C_cabs, h_tmp, optimize='optimal')
    h = h.at[:nobs, nobs:nri].set(h_tmp)
    h = h.at[nobs:nri, :nobs].set(jnp.transpose(h_tmp))

    # <C|C>
    h_tmp = compute_f12_oeints(geom, cabs_name, cabs_name, xyz_path, deriv_order, options)
    h_tmp = jnp.einsum('pP,qQ,pq->PQ', C_cabs, C_cabs, h_tmp, optimize='optimal')
    h = h.at[nobs:nri, nobs:nri].set(h_tmp)

    return h

def form_Fock(geom, basis_name, cabs_name, C_obs, C_cabs, nocc, xyz_path, deriv_order, options):
    nobs = C_obs.shape[0]
    nri = C_cabs.shape[0]

    f = jnp.empty((nri, nri))
    fk = jnp.empty((nri, nri))

    # OEINTS
    h = form_h(geom, basis_name, cabs_name, C_obs, C_cabs, xyz_path, deriv_order, options)
    f.at[:, :].set(h)

    # TEINTS
    G = jnp.empty((nri, nobs, nri, nri))

    G_tmp = compute_f12_teints(geom, basis_name, basis_name, basis_name, basis_name, "eri", xyz_path, deriv_order, options)
    G_tmp = jnp.einsum('pP,qQ,rR,sS,pqrs->PRQS', C_obs, C_obs, C_obs, C_obs, G_tmp, optimize='optimal')
    G = G.at[:nobs, :nocc, :nobs, :nobs].set(G_tmp) # <OO|OO>

    G_tmp = compute_f12_teints(geom, cabs_name, basis_name, basis_name, basis_name, "eri", xyz_path, deriv_order, options)
    G_tmp = jnp.einsum('pP,qQ,rR,sS,pqrs->PRQS', C_cabs, C_obs, C_obs, C_obs, G_tmp, optimize='optimal')
    G = G.at[nobs:nri, :nocc, :nobs, :nobs].set(G_tmp) # <CO|OO>
    G = G.at[:nocc, :nobs, nobs:nri, :nobs].set(jnp.transpose(G_tmp, (2,3,1,0))) # <OO|CO>
    G = G.at[:nocc, :nobs, :nobs, nobs:nri].set(jnp.transpose(G_tmp, (3,2,1,0))) # <OO|OC>

    G_tmp = compute_f12_teints(geom, cabs_name, basis_name, basis_name, cabs_name, "eri", xyz_path, deriv_order, options)
    G_tmp = jnp.einsum('pP,qQ,rR,sS,pqrs->PRQS', C_cabs, C_obs, C_obs, C_cabs, G_tmp, optimize='optimal')
    G = G.at[nobs:nri, :nocc, :nobs, nobs:nri].set(G_tmp) # <CO|OC>

    G_tmp = compute_f12_teints(geom, cabs_name, cabs_name, basis_name, basis_name, "eri", xyz_path, deriv_order, options)
    G_tmp = jnp.einsum('pP,qQ,rR,sS,pqrs->PRQS', C_cabs, C_cabs, C_obs, C_obs, G_tmp, optimize='optimal')
    G = G.at[nobs:nri, :nocc, nobs:nri, :nobs].set(G_tmp) # <CO|CO>

    # Fill Fock Matrix
    f.at[:, :].set(2.0 * jnp.einsum('PIQI->PQ', G[:, :nocc, :, nocc], optimize='optimal'))
    fk.at[:, :].set(f)      
    f.at[:, :].add(-1.0 * jnp.einsum('PIIQ->PQ', G[:, :nocc, :nocc, :], optimize='optimal'))

    return f, fk

    




    
