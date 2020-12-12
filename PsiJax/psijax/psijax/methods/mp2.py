import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental import loops
import psi4

from .energy_utils import nuclear_repulsion, partial_tei_transformation, tei_transformation, cartesian_product
from .hartree_fock import restricted_hartree_fock

def restricted_mp2(geom, basis_name, xyz_path, nuclear_charges, charge):
    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2
    E_scf, C, eps, G = restricted_hartree_fock(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=True)

    nvirt = G.shape[0] - ndocc
    nbf = G.shape[0]

    G = partial_tei_transformation(G, C[:,:ndocc],C[:,ndocc:],C[:,:ndocc],C[:,ndocc:])

    # Create tensor dim (occ,vir,occ,vir) of all possible orbital energy denominators
    # Partial tei transformation is super efficient, it is this part that is bad.
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
    with loops.Scope() as s:
      s.mp2_correlation = 0.
      for idx in s.range(indices.shape[0]):
        i,j,a,b = indices[idx]
        s.mp2_correlation += G[i, a, j, b] * (2 * G[i, a, j, b] - G[i, b, j, a]) * e_denom[i,a,j,b]
      return E_scf + s.mp2_correlation

