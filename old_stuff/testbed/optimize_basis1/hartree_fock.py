import torch
import numpy as np
from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion

# Define coordinates in Bohr as Torch tensors, turn on gradient tracking.  
#tmpgeom1 = [0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]
tmpgeom1 = [0.000000000000,0.000000000000,-0.743649338313,0.000000000000,0.000000000000,0.743649338313]

tmpgeom2 = [torch.tensor(i, requires_grad=True) for i in tmpgeom1]
geom = torch.stack(tmpgeom2).reshape(2,3)
# Define some basis function exponents
basis0 = torch.tensor([0.5], requires_grad=False)
basis1 = torch.tensor([0.5, 0.4], requires_grad=False)
basis2 = torch.tensor([0.5, 0.4, 0.3, 0.2], requires_grad=False)
basis3 = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05], requires_grad=False)
basis4 = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001], requires_grad=False)
basis5 = torch.rand(50)
# Load converged Psi4 Densities for basis sets 1 through 4
#F0 = torch.from_numpy(np.load('psi4_fock_matrices/F0.npy'))
#F1 = torch.from_numpy(np.load('psi4_fock_matrices/F1.npy'))
#F2 = torch.from_numpy(np.load('psi4_fock_matrices/F2.npy'))
#F3 = torch.from_numpy(np.load('psi4_fock_matrices/F3.npy'))
#F4 = torch.from_numpy(np.load('psi4_fock_matrices/F4.npy'))
#
@torch.jit.script
def hartree_fock_old(basis,geom,F):
    """Takes basis, geometry, converged Psi4 Fock matrix wfn.Fa()"""
    ndocc = 1                 #hard coded
    full_basis = torch.cat((basis,basis))
    nbf_per_atom = torch.tensor([basis.size()[0],basis.size()[0]])
    charge_per_atom = torch.tensor([1.0,1.0])
    Enuc = nuclear_repulsion(geom[0], geom[1])
    S, T, V = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
    A = orthogonalizer(S)
    G = vectorized_tei(full_basis,geom,nbf_per_atom)
    H = T + V
    Fp = torch.chain_matmul(A, F, A)
    eps, C2 = torch.symeig(Fp, eigenvectors=True)       
    C = torch.matmul(A, C2)
    Cocc = C[:, :ndocc]
    # This density is now 'connected' in a computation graph to the input geometry, compute energy
    D = torch.einsum('pi,qi->pq', Cocc, Cocc)
    J = torch.einsum('pqrs,rs->pq', G, D)
    K = torch.einsum('prqs,rs->pq', G, D)
    F = H + J * 2 - K
    E_scf = torch.einsum('pq,pq->', F + H, D) + Enuc
    return E_scf

#@torch.jit.script
def hartree_fock_iterative(basis,geom,exact_energy):
    """
    Takes basis, geometry, converged Psi4 hartree fock energy.
    In order to get exact analytic hessians, for some reason,
    you MUST iterate, even if the energy is converged on the first iteration.
    The hessian is exact when the energy naturally reaches the exact energy.
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
    #Hp = torch.chain_matmul(A,H,A)
    #e, C2 = torch.symeig(Hp, eigenvectors=True)
    #C = torch.matmul(A,C2)
    #Cocc = C[:, :ndocc]
    #D = torch.einsum('pi,qi->pq', Cocc, Cocc)
    # ZERO GUESS
    #D = torch.zeros_like(H) 
    # TORCH CONVERGED D GUESS
    D = torch.load('D.pt') * 2
    for i in range(50):
        J = torch.einsum('pqrs,rs->pq', G, D)
        K = torch.einsum('prqs,rs->pq', G, D)
        F = H + J * 2 - K
        Fp = torch.chain_matmul(A, F, A)
        eps, C2 = torch.symeig(Fp, eigenvectors=True)       
        C = torch.matmul(A, C2)
        Cocc = C[:, :ndocc]
        D = torch.einsum('pi,qi->pq', Cocc, Cocc)
        #torch.save(D, 'D.pt')
        E_scf = torch.einsum('pq,pq->', F + H, D) + Enuc
        print(E_scf)
        if torch.allclose(E_scf, exact_energy, rtol=1e-10, atol=1e-10):
            return E_scf

def hartree_fock_derivatives(E,geom):
    grad = torch.autograd.grad(E, geom, create_graph=True)[0]
    h1 = torch.autograd.grad(grad[0,0],geom,create_graph=True)[0]
    h2 = torch.autograd.grad(grad[0,1],geom,create_graph=True)[0]
    h3 = torch.autograd.grad(grad[0,2],geom,create_graph=True)[0]
    h4 = torch.autograd.grad(grad[1,0],geom,create_graph=True)[0]
    h5 = torch.autograd.grad(grad[1,1],geom,create_graph=True)[0]
    h6 = torch.autograd.grad(grad[1,2],geom,create_graph=True)[0]
    hess = torch.stack([h1,h2,h3,h4,h5,h6]).reshape(6,6)
    return grad, hess

#exact = torch.tensor(-0.9716855914049)
exact = torch.tensor(-0.979758321007 )

E = hartree_fock_iterative(basis1,geom,exact)
grad, hess = hartree_fock_derivatives(E,geom)
print(E)
print(grad)
print(hess)

# For arbitrary

