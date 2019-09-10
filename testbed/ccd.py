import torch
import numpy as np
from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion
from differentiate import differentiate 
from hartree_fock import hartree_fock

# Define coordinates in Bohr as Torch tensors, turn on gradient tracking.  
tmpgeom = [0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]
geomlist = [torch.tensor(i, requires_grad=True) for i in tmpgeom]
geom = torch.stack(geomlist).reshape(2,3)
# Define some basis function exponents
basis0 = torch.tensor([0.5], requires_grad=False)
basis1 = torch.tensor([0.5, 0.4], requires_grad=False)
basis2 = torch.tensor([0.5, 0.4, 0.3, 0.2], requires_grad=False)
basis3 = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05], requires_grad=False)
basis4 = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001], requires_grad=False)
basis5 = torch.rand(50)
# Load converged Psi4 Densities for basis sets 1 through 4
F0 = torch.from_numpy(np.load('psi4_fock_matrices/F0.npy'))
F1 = torch.from_numpy(np.load('psi4_fock_matrices/F1.npy'))
F2 = torch.from_numpy(np.load('psi4_fock_matrices/F2.npy'))
F3 = torch.from_numpy(np.load('psi4_fock_matrices/F3.npy'))
F4 = torch.from_numpy(np.load('psi4_fock_matrices/F4.npy'))

@torch.jit.script
def ccd_corr(eps, G, C):
    ndocc = 1 #hard coded
    o = slice(None, ndocc)
    v = slice(ndocc, None)
    eps_occ, eps_vir = eps[:ndocc], eps[ndocc:]
    e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) - eps_vir.reshape(-1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir)
    g_phys = G.permute(0,2,1,3) - G.permute(0,2,3,1)

    Gmo = torch.einsum('pqrs,sl,rk,qj,pi->ijkl', G, C[:,ndocc:],C[:,:ndocc],C[:,ndocc:],C[:,:ndocc])
    mp2_correlation_e = torch.einsum('iajb,iajb,iajb->', Gmo, Gmo, e_denom) +\
                        torch.einsum('iajb,iajb,iajb->', Gmo - Gmo.permute(0,3,2,1), Gmo, e_denom)
    return mp2_correlation_e
