import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import fori_loop
import psi4

from ..integrals.basis_utils import build_CABS
from .ints import compute_f12_oeints, compute_f12_teints
from .energy_utils import nuclear_repulsion, tei_transformation
from .mp2 import restricted_mp2

def restricted_mp2_f12(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=0):
    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2
    E_mp2, C_obs, eps = restricted_mp2(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order, return_aux_data=True)

    # Force to use Dunning basis sets with associated CABS
    # Libint has a limited number of basis sets available
    if 'cc-pv' in basis_name.lower():
        cabs_name = basis_name + "-cabs"
    C_cabs = jnp.array(build_CABS(geom, basis_name, cabs_name))

    h = form_h(geom, basis_name, cabs_name, C_obs, C_cabs, xyz_path, deriv_order, options)

    f, fk = form_Fock(geom, basis_name, cabs_name, C_obs, C_cabs, ndocc, xyz_path, deriv_order, options)

def form_h(geom, basis_name, cabs_name, C_obs, C_cabs, xyz_path, deriv_order, options):
    nobs = C_obs.shape[0]
    nri = C_cabs.shape[0]

    h = jnp.zeros((nri, nri))

    h_tmp = compute_f12_oeints(geom, basis_name, basis_name, xyz_path, deriv_order, options)
    h[:nobs, :nobs] = jnp.dot(C_obs, jnp.dot(h_tmp, C_obs))

    h_tmp = compute_f12_oeints(geom, basis_name, cabs_name, xyz_path, deriv_order, options)
    h[:nobs, nobs:nri] = jnp.dot(C_obs, jnp.dot(h_tmp, C_cabs))
    h[nobs:nri, :nobs] = h[:nobs, nobs:nri].T

    h_tmp = compute_f12_oeints(geom, cabs_name, cabs_name, xyz_path, deriv_order, options)
    h[nobs:nri, nobs:nri] = jnp.dot(C_cabs, jnp.dot(h_tmp, C_cabs))

    return h

def form_Fock(geom, basis_name, cabs_name, C_obs, C_cabs, nocc, xyz_path, deriv_order, options):
    nobs = C_obs.shape[0]
    nri = C_cabs.shape[0]

    f = np.zeros((nri, nri))
    fk = np.zeros((nri, nri))

    J_tmp = compute_f12_teints(geom, basis_name, basis_name, basis_name, basis_name, "eri", xyz_path, deriv_order, options)




    
