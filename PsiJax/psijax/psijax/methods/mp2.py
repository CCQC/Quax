import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
from jax.experimental import loops
import psi4
import numpy as onp

from .energy_utils import nuclear_repulsion, partial_tei_transformation, tei_transformation
from .hartree_fock import restricted_hartree_fock

def restricted_mp2(geom, basis, nuclear_charges, charge):
    nelectrons = int(np.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2

    E_scf, C, eps, G = restricted_hartree_fock(geom, basis, nuclear_charges, charge, SCF_MAX_ITER=50, return_aux_data=True)

    eps_occ, eps_vir = eps[:ndocc], eps[ndocc:]
    e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) - eps_vir.reshape(-1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir)

    # MO basis integrals
    G = partial_tei_transformation(G, C[:,ndocc:], C[:,:ndocc], C[:,ndocc:], C[:,:ndocc])

    mp2_correlation = np.einsum('iajb,iajb,iajb->', G, G, e_denom) +\
                      np.einsum('iajb,iajb,iajb->', G - np.transpose(G, (0,3,2,1)), G, e_denom)

    mp2_total_energy = mp2_correlation + E_scf
    #print("MP2 Correlation Energy:    ", mp2_correlation)
    #print("MP2 Total Energy:          ", mp2_correlation)
    return E_scf + mp2_correlation

def restricted_mp2_lowmem(geom, basis, nuclear_charges, charge):
    nelectrons = int(np.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2
    E_scf, C, eps, G, H = restricted_hartree_fock(geom, basis, nuclear_charges, charge, SCF_MAX_ITER=50, return_aux_data=True)

    nvirt = G.shape[0] - ndocc
    nbf = G.shape[0]
    G = tei_transformation(G, C) # have to do full transform for loop algo for some reason? am i dumb? 
    
    with loops.Scope() as s:
      s.mp2_correlation = 0.
      for i in s.range(ndocc):
        for j in s.range(ndocc):
          for a in s.range(ndocc, nbf):
            for b in s.range(ndocc, nbf):
              s.mp2_correlation += G[i, a, j, b] * (2 * G[i, a, j, b] - G[i, b, j, a]) / (eps[i] + eps[j] - eps[a] - eps[b])
      return E_scf + s.mp2_correlation


