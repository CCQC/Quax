import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import fori_loop

from .energy_utils import partial_tei_transformation, cartesian_product
from .hartree_fock import restricted_hartree_fock

def restricted_mp2(*args, options, deriv_order=0, return_aux_data=False):
    if options['electric_field']:
        electric_field, geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path = args
        scf_args = electric_field, geom, basis_set, nelectrons, nuclear_charges, xyz_path
    else:
        geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path = args
        scf_args = (geom, basis_set, nelectrons, nuclear_charges, xyz_path)

    E_scf, C, eps, G = restricted_hartree_fock(*scf_args, options=options, deriv_order=deriv_order, return_aux_data=True)

    # Load keyword options
    ndocc = nelectrons // 2
    ncore = nfrzn // 2

    print("Running MP2 Computation...")
    nvirt = G.shape[0] - ndocc

    G = partial_tei_transformation(G, C[:,ncore:ndocc], C[:,ndocc:], C[:,ncore:ndocc], C[:,ndocc:])

    # Create tensor dim (occ,vir,occ,vir) of all possible orbital energy denominators
    eps_occ, eps_vir = eps[ncore:ndocc], eps[ndocc:]
    e_denom = jnp.reciprocal(eps_occ.reshape(-1, 1, 1, 1) - eps_vir.reshape(-1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir)

    # Loop algo (lower memory, but tei transform is the memory bottleneck)
    # Create all combinations of four loop variables to make XLA compilation easier
    indices = cartesian_product(jnp.arange(ndocc-ncore), jnp.arange(ndocc-ncore), jnp.arange(nvirt), jnp.arange(nvirt))

    def loop_mp2(idx, mp2_corr):
        i,j,a,b = indices[idx]
        mp2_corr += G[i, a, j, b] * (2 * G[i, a, j, b] - G[i, b, j, a]) * e_denom[i, a, j, b]
        return mp2_corr

    dE_mp2 = fori_loop(0, indices.shape[0], loop_mp2, 0.0) # MP2 correlation

    if return_aux_data:
        #print("MP2 Energy:                ", E_scf + dE_mp2)
        return E_scf + dE_mp2, C, eps, G
    else:
        return E_scf + dE_mp2

