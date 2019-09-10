import torch 
import numpy as np
from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion
from differentiate import differentiate, jacobian

def hartree_fock(geom,ndocc,basis,nbf_per_atom,charge_per_atom,convergence=1e-8):
    """
    Roothan-Hall Restricted Hartree Fock using symmetric orthogonalization.
    All higher order derivatives are exact to the precision the energy is converged to. 

    Parameters
    ----------
    geom : torch.tensor
        A Natom x 3 tensor of cartesian coordinates in Bohr
    ndocc : int
        Number of doubly occupied orbitals (rather than charge and multiplicity definitions)
    basis : torch.tensor()
        A 1-dim torch tensor containing all basis functions of all atoms, in the order as given by cartesian coordinates ('geom' arg)
    nbf_per_atom : torch.tensor()
        A 1-dim torch tensor of size 'natoms' defining how many basis functions for each atom in the order as given by cartesian coordinates ('geom' arg)
    charge_per_atom : torch.tensor()
        A 1-dim torch tensor of size 'natoms' containing the nuclear charge of each atom 
    convergence : float
        Convergence criteria of Hartree Fock energy 
    """
    #ndocc = 1   #hard coded
    #full_basis = torch.cat((basis,basis))
    #nbf_per_atom = torch.tensor([basis.size()[0],basis.size()[0]])
    #charge_per_atom = torch.tensor([1.0,1.0])
    #TODO generalize nuclear_repulsion()
    Enuc = nuclear_repulsion(geom[0], geom[1])
    S, T, V = vectorized_oei(basis, geom, nbf_per_atom, charge_per_atom)
    A = orthogonalizer(S)
    G = vectorized_tei(basis,geom,nbf_per_atom)
    H = T + V
    # CORE GUESS
    Hp = torch.chain_matmul(A,H,A)
    eps, C2 = torch.symeig(Hp, eigenvectors=True)
    C = torch.matmul(A,C2)
    Cocc = C[:, :ndocc]
    D = torch.einsum('pi,qi->pq', Cocc, Cocc)

    for i in range(20):
        J = torch.einsum('pqrs,rs->pq', G, D)
        K = torch.einsum('prqs,rs->pq', G, D)
        F = H + J * 2 - K
        Fp = torch.chain_matmul(A, F, A)
        eps, C2 = torch.symeig(Fp, eigenvectors=True)       
        C = torch.matmul(A, C2)
        Cocc = C[:, :ndocc]
        D = torch.einsum('pi,qi->pq', Cocc, Cocc)
        E_scf = torch.einsum('pq,pq->', F + H, D) + Enuc
        if i>1:
            if torch.allclose(E_scf, old_E_scf, rtol=convergence, atol=convergence):
                print("{} Iterations Required".format(i))
                return E_scf, eps, C, G
        old_E_scf = E_scf
    print("Did not converge.")
    return E_scf, eps, C, G

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

# To use:
# Define coordinates with gradient tracking turned on
tmpgeom = [0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]
geomlist = [torch.tensor(i, requires_grad=True) for i in tmpgeom]
geom = torch.stack(geomlist).reshape(2,3)
# Define basis set (no coefficients, just exponents) and charge info
basis = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001], requires_grad=False) # Hydrogen atom basis set
full_basis = torch.cat((basis,basis))                                                   # H2 total basis set
nbf_per_atom = torch.tensor([basis.size()[0],basis.size()[0]])
charge_per_atom = torch.tensor([1.0,1.0])
E, eps, C, G = hartree_fock(geom,1,full_basis,nbf_per_atom,charge_per_atom,convergence=1e-8)

# To differentiate:
# Just gradient:
'''
grad = torch.autograd.grad(E, geom)
'''
# Gradient and Hessian: (cubic,quartic,quintic,sextic, just modify 'order', but be careful, very expensive, high RAM)
'''
grad, hess = differentiate(E, geomlist, order=2)
'''
# A single higher order derivative (Example: d^3E/dz2 dy2 dx2:)
'''
g = torch.autograd.grad(E, geomlist[-1], create_graph=True)
h = torch.autograd.grad(g, geomlist[-2], create_graph=True)
c = torch.autograd.grad(h, geomlist[-3], create_graph=True)
'''

