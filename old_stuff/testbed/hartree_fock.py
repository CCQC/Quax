import torch 
import numpy as np

from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion
from differentiate import differentiate, jacobian

# Define coordinates in Bohr as Torch tensors, turn on gradient tracking.  
tmpgeom = [0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]
geomlist = [torch.tensor(i, requires_grad=True) for i in tmpgeom]
#geomlist = [torch.tensor(i, requires_grad=False) for i in tmpgeom]
geom = torch.stack(geomlist).reshape(2,3)
# Define some basis function exponents
basis0 = torch.tensor([0.5], requires_grad=False)
basis1 = torch.tensor([0.5, 0.4], requires_grad=False)
basis2 = torch.tensor([0.5, 0.4, 0.3, 0.2], requires_grad=False)
basis3 = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05], requires_grad=False)
basis4 = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001], requires_grad=False)
basis5 = torch.rand(50)

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
    print(torch.autograd.grad(E_scf,geom,create_graph=True))
    print(torch.autograd.grad(D,geom,grad_outputs=torch.ones_like(D),create_graph=True))
    return E_scf

def hartree_fock(basis,geom,convergence=torch.tensor(1e-9)):
    """
    Takes basis, geometry, and converged Psi4 hartree fock energy.
    In order to get exact analytic hessians, for some reason,
    you MUST iterate, even if the energy is converged on the first iteration.
    All higher order derivatives are exact when the energy naturally reaches the exact energy.
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
    eps, C2 = torch.symeig(Hp, eigenvectors=True)
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
        #print(jacobian(Cocc,geom).size())
        # nbf, ndocc, natom, 3
        print(jacobian(Cocc,geom))
        print(jacobian(E_scf,Cocc).size())

        if i>1:
            if torch.allclose(E_scf, old_E_scf, rtol=convergence, atol=convergence):
                print("{} Iterations Required".format(i))
                break
        old_E_scf = E_scf
    return E_scf, eps, C, G


def benchmark(basis, geom):
    E, eps, C, G = hartree_fock(basis,geom)
    #grad = torch.autograd.grad(E,geom,create_graph=True)
    #grad, hess = differentiate(E, geomlist, order=2)
    #print(grad)
    #print(hess)

benchmark(basis2,geom)
#F2 = torch.from_numpy(np.load('psi4_data/F2.npy'))
#F3 = torch.from_numpy(np.load('psi4_data/F3.npy'))
#hartree_fock_old(basis3,geom,F3)


#E, eps, C, G = hartree_fock(basis0,geom,exact0)
##E, eps, C, G = hartree_fock(torch.rand(25),geom,exact2)
#grad, hess = differentiate(E, geomlist, order=2)
#print(E)
#print(grad)
#print(hess)


