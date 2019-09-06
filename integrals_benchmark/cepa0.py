import torch
import numpy as np
import opt_einsum
torch.set_default_dtype(torch.float64)

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


#def spin_block_tei(I):
#    identity = torch.tensor([[1.0,0.0],[0.0,1.0]])
#    I = kronecker(identity, I)
#    return kronecker(identity, I.T)
def spin_block_tei(I):
    """  
    Function that spin blocks two-electron integrals
    Using np.kron, we project I into the space of the 2x2 identity, tranpose the result
    and project into the space of the 2x2 identity again. This doubles the size of each axis.
    The result is our two electron integral tensor in the spin orbital form.
    """
    identity = np.eye(2)
    I = np.kron(identity, I)
    return np.kron(identity, I.T)


def cepa_corr(eps, G, C):
    eps = torch.cat((eps,eps))
    nocc = 2 #hard coded
    nvirt = torch.numel(eps) - nocc
    Czero = torch.zeros_like(C)
    Ctmp = torch.stack((C, Czero),dim=0).reshape(8,4)
    Ctmp2 = torch.stack((Czero, C),dim=0).reshape(8,4)
    Cs = torch.cat((Ctmp,Ctmp2),dim=1)
    ndocc = 1 #hard coded
    Cs = Cs[:, torch.argsort(eps)]
    eps = torch.sort(eps)[0]
    # Is this the same as spin block?
    G2 = torch.from_numpy(spin_block_tei(G.detach().numpy()))
    tmp = G2.permute(0, 2, 1, 3)
    gao = tmp - tmp.permute(0,1,3,2)

    o = slice(None, nocc)
    v = slice(nocc, None)
    eps_occ, eps_vir = eps[:nocc], eps[nocc:]
    # construct full transformed I matrix since we are lazy
    gmo = torch.einsum('pjkl, pi -> ijkl',
          torch.einsum('pqkl, qj -> pjkl',
          torch.einsum('pqrl, rk -> pqkl',
          torch.einsum('pqrs, sl -> pqrl', gao, Cs), Cs), Cs), Cs)
    e_denom = 1 / (-eps_vir.reshape(-1, 1, 1, 1) - eps_vir.reshape(-1, 1, 1) + eps_occ.reshape(-1, 1) + eps_occ)

    t_amp = torch.zeros((nvirt, nvirt, nocc, nocc))

    for i in range(20):
        # Collect terms
        mp2    = gmo[v, v, o, o]
        cepa1  = (1 / 2) * torch.einsum('abcd, cdij -> abij', gmo[v, v, v, v], t_amp)
        cepa2  = (1 / 2) * torch.einsum('klij, abkl -> abij', gmo[o, o, o, o], t_amp)
        cepa3a = torch.einsum('akic, bcjk -> abij', gmo[v, o, o, v], t_amp)
        cepa3b = -cepa3a.permute(1, 0, 2, 3)
        cepa3c = -cepa3a.permute(0, 1, 3, 2)
        cepa3d =  cepa3a.permute(1, 0, 3, 2)
        cepa3  =  cepa3a + cepa3b + cepa3c + cepa3d
        # Update t amplitude
        t_amp_new = e_denom * (mp2 + cepa1 + cepa2 + cepa3)
        # Evaluate Energy
        cepa_correlation_e = (1 / 4) * torch.einsum('ijab, abij ->', gmo[o, o, v, v], t_amp_new)
        print(cepa_correlation_e)
        t_amp = t_amp_new

    return cepa_correlation_e


def cepa(basis,geom,exact_hf_energy,convergence=1e-9):
    """
    Takes basis, geometry, and converged Psi4 Hartree Fock energy.
    In order to get exact analytic hessians, for some reason,
    you MUST iterate, even if the energy is converged on the first iteration.
    All higher order derivatives are exact (to within 'convergence') when the energy naturally reaches the exact energy.
    """
    E_scf, eps, C, G = hartree_fock(basis,geom,exact_hf_energy,convergence=convergence)
    E_cepa = cepa_corr(eps, G, C) + E_scf
    return E_cepa

exact0 = torch.tensor(-0.931283011458994) 
exact1 = torch.tensor(-0.971685591404988)
exact2 = torch.tensor(-1.060859783988007)
exact3 = torch.tensor(-1.0687016869345958)
exact4 = torch.tensor(-1.06888692908086)

E_cepa = cepa(basis1, geom, exact1)
print(E_cepa)

#grad, hess = differentiate(E_mp2, geomlist, order=2)
#print(E_mp2)
#print(grad)
#print(hess)


