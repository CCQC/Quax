import torch
import numpy as np
from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion
from differentiate import differentiate 

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
def mp2_corr(eps, G, C):
    ndocc = 1 #hard coded
    eps_occ, eps_vir = eps[:ndocc], eps[ndocc:]
    e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) - eps_vir.reshape(-1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir)
    Gmo = torch.einsum('pqrs,sl,rk,qj,pi->ijkl', G, C[:,ndocc:],C[:,:ndocc],C[:,ndocc:],C[:,:ndocc])
    mp2_correlation_e = torch.einsum('iajb,iajb,iajb->', Gmo, Gmo, e_denom) +\
                        torch.einsum('iajb,iajb,iajb->', Gmo - Gmo.permute(0,3,2,1), Gmo, e_denom)
    return mp2_correlation_e

#@torch.jit.script
def mp2(basis,geom,exact_hf_energy,convergence=1e-10):
    """
    Takes basis, geometry, and converged Psi4 hartree fock energy.
    In order to get exact analytic hessians, for some reason,
    you MUST iterate, even if the energy is converged on the first iteration.
    All higher order derivatives are exact (to within 'convergence') when the energy naturally reaches the exact energy.
    """
    ndocc = 1   #hard coded
    full_basis = torch.cat((basis,basis))
    nbf_per_atom = torch.tensor([basis.size()[0],basis.size()[0]])
    charge_per_atom = torch.tensor([1.0,1.0])
    Enuc = nuclear_repulsion(geom[0], geom[1])
    S, T, V = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
    A = orthogonalizer(S)
    G = vectorized_tei(full_basis,geom,nbf_per_atom)
    H = T + V
    # CORE GUESS
    Hp = torch.chain_matmul(A,H,A)
    e, C2 = torch.symeig(Hp, eigenvectors=True)
    C = torch.matmul(A,C2)
    Cocc = C[:, :ndocc]
    D = torch.einsum('pi,qi->pq', Cocc, Cocc)
    for i in range(50):
        J = torch.einsum('pqrs,rs->pq', G, D)
        K = torch.einsum('prqs,rs->pq', G, D)
        F = H + J * 2 - K
        Fp = torch.chain_matmul(A, F, A)
        eps, C2 = torch.symeig(Fp, eigenvectors=True)       
        C = torch.matmul(A, C2)
        Cocc = C[:, :ndocc]
        D = torch.einsum('pi,qi->pq', Cocc, Cocc)
        E_scf = torch.einsum('pq,pq->', F + H, D) + Enuc
        print(E_scf)
        if torch.allclose(E_scf, exact_hf_energy, rtol=convergence, atol=convergence):
            break
    E_mp2 = mp2_corr(eps, G, C) + E_scf
    return E_mp2
    
exact0 = torch.tensor(-0.931283011458994) 
exact1 = torch.tensor(-0.971685591404988)
exact2 = torch.tensor(-1.060859783988007)

E_mp2 = mp2(basis1, geom, exact1)
grad, hess = differentiate(E_mp2, geomlist, order=2)
print(E_mp2)
print(grad)
print(hess)


