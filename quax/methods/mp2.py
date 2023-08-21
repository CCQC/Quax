import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import fori_loop
import psi4

from .energy_utils import nuclear_repulsion, partial_tei_transformation, tei_transformation, cartesian_product
from .hartree_fock import restricted_hartree_fock

def restricted_mp2(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=0):
    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2
    E_scf, C, eps, G = restricted_hartree_fock(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order, return_aux_data=True)

    nvirt = G.shape[0] - ndocc
    nbf = G.shape[0]

    G = partial_tei_transformation(G, C[:,:ndocc],C[:,ndocc:],C[:,:ndocc],C[:,ndocc:])

    # Create tensor dim (occ,vir,occ,vir) of all possible orbital energy denominators
    eps_occ, eps_vir = eps[:ndocc], eps[ndocc:]
    e_denom = jnp.reciprocal(eps_occ.reshape(-1, 1, 1, 1) - eps_vir.reshape(-1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir)

    # Tensor contraction algo 
    #mp2_correlation = jnp.einsum('iajb,iajb,iajb->', G, G, e_denom) +\
    #                  jnp.einsum('iajb,iajb,iajb->', G - jnp.transpose(G, (0,3,2,1)), G, e_denom)
    #mp2_total_energy = mp2_correlation + E_scf
    #return E_scf + mp2_correlation

    # Loop algo (lower memory, but tei transform is the memory bottleneck)
    # Create all combinations of four loop variables to make XLA compilation easier
    indices = cartesian_product(jnp.arange(ndocc),jnp.arange(ndocc),jnp.arange(nvirt),jnp.arange(nvirt))

    mp2_correlation = 0.0
    def loop_mp2(idx, mp2_corr):
        i,j,a,b = indices[idx]
        mp2_corr += G[i, a, j, b] * (2 * G[i, a, j, b] - G[i, b, j, a]) * e_denom[i,a,j,b]
        return mp2_corr

    dE_mp2 = fori_loop(0, indices.shape[0], loop_mp2, mp2_correlation)

    return E_scf + dE_mp2

