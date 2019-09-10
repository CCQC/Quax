import torch
import numpy as np
import opt_einsum

from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion
from differentiate import differentiate 
from hartree_fock import hartree_fock

def mp2_corr(eps, C, G, ndocc):
    '''
    Computes MP2 correlation energy
    '''
    o = slice(None, ndocc)
    v = slice(ndocc, None)
    eps_occ, eps_vir = eps[:ndocc], eps[ndocc:]
    e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) - eps_vir.reshape(-1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir)
    Gmo = torch.einsum('pqrs,sl,rk,qj,pi->ijkl', G, C[:,ndocc:],C[:,:ndocc],C[:,ndocc:],C[:,:ndocc])
    mp2_correlation_e = torch.einsum('iajb,iajb,iajb->', Gmo, Gmo, e_denom) +\
                        torch.einsum('iajb,iajb,iajb->', Gmo - Gmo.permute(0,3,2,1), Gmo, e_denom)
    return mp2_correlation_e

