import torch
import numpy as np
import opt_einsum

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

#@torch.jit.script
def mp2_corr(eps, G, C):
    ndocc = 1 #hard coded
    o = slice(None, ndocc)
    v = slice(ndocc, None)
    eps_occ, eps_vir = eps[:ndocc], eps[ndocc:]
    e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) - eps_vir.reshape(-1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir)

    Gmo = torch.einsum('pqrs,sl,rk,qj,pi->ijkl', G, C[:,ndocc:],C[:,:ndocc],C[:,ndocc:],C[:,:ndocc])
    mp2_correlation_e = torch.einsum('iajb,iajb,iajb->', Gmo, Gmo, e_denom) +\
                        torch.einsum('iajb,iajb,iajb->', Gmo - Gmo.permute(0,3,2,1), Gmo, e_denom)
    return mp2_correlation_e

def mp2(basis,geom,exact_hf_energy,convergence=1e-9):
    """
    Takes basis, geometry, and converged Psi4 Hartree Fock energy.
    In order to get exact analytic hessians, for some reason,
    you MUST iterate, even if the energy is converged on the first iteration.
    All higher order derivatives are exact (to within 'convergence') when the energy naturally reaches the exact energy.
    """
    E_scf, eps, C, G = hartree_fock(basis,geom,exact_hf_energy,convergence=convergence)
    E_mp2 = mp2_corr(eps, G, C) + E_scf
    return E_mp2


def benchmark(basis, geom, exact):
    E_mp2 = mp2(basis,geom,exact)
    #grad = torch.autograd.grad(E_mp2,geom,create_graph=True)
    #grad, hess = differentiate(E_mp2, geomlist, order=2)

exact0 = torch.tensor(-0.931283011458994) 
exact1 = torch.tensor(-0.971685591404988)
exact2 = torch.tensor(-1.060859783988007)
exact3 = torch.tensor(-1.0687016869345958)
exact4 = torch.tensor(-1.06888692908086)

#E_mp2 = mp2(basis4, geom, exact4)
#grad, hess = differentiate(E_mp2, geomlist, order=2)
#print(E_mp2)
#print(grad)
#print(hess)


