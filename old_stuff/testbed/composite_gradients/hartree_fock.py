import torch 
import numpy as np

from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion
from differentiate import differentiate, jacobian

torch.set_printoptions(precision=7)

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

def hartree_fock2(basis,geom,D):
    ndocc = 1   #hard coded
    full_basis = torch.cat((basis,basis))
    nbf = torch.numel(full_basis)
    nbf_per_atom = torch.tensor([basis.size()[0],basis.size()[0]])
    charge_per_atom = torch.tensor([1.0,1.0])

    Enuc = nuclear_repulsion(geom[0], geom[1])
    S, T, V = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
    A = orthogonalizer(S)
    G = vectorized_tei(full_basis,geom,nbf_per_atom)
    #DETACH
    Enuc = Enuc.detach()
    S = S.detach()
    T = T.detach()
    V = V.detach()
    A = A.detach()
    #G = G.detach()
    # Energy computation
    H = T + V
    J = torch.tensordot(G, D, dims=([2,3], [0,1])) 
    K = torch.tensordot(G, D, dims=([1,3], [0,1])) 
    F = H + J * 2 - K
    #E_scf = torch.tensordot(F + H, D, dims=([0,1],[0,1])) + Enuc 
    #print(E_scf)
    ##grad = torch.autograd.grad(E_scf,D,create_graph=True)[0]
    #grad = torch.autograd.grad(E_scf,geom,create_graph=True)[0]
    #print(grad)

    Fp = torch.chain_matmul(A, F, A)
    eps, Cp = torch.symeig(Fp, eigenvectors=True)       
    C = torch.matmul(A, Cp)
    Cocc = C[:, :ndocc]
    D = torch.matmul(Cocc, Cocc.t())
    #TODO
    D = D.detach() 
    #TODO
    for i in range(10):
        J = torch.tensordot(G, D, dims=([2,3], [0,1])) 
        K = torch.tensordot(G, D, dims=([1,3], [0,1])) 
        F = H + J * 2 - K
        E_scf = torch.tensordot(F + H, D, dims=([0,1],[0,1])) + Enuc 
        #print(E_scf)
        #grad = torch.autograd.grad(E_scf,D,create_graph=True)[0]
        grad,hess = differentiate(E_scf, geomlist, order=2) 
        print(grad[-1])
        print(hess[2,-1])
        Fp = torch.chain_matmul(A, F, A)
        eps, Cp = torch.symeig(Fp, eigenvectors=True)       
        C = torch.matmul(A, Cp)
        Cocc = C[:, :ndocc]
        D = torch.matmul(Cocc, Cocc.t())
        #TODO
        D = D.detach()
        #TODO


def hartree_fock(basis,geom,C,iterate=False):
    """Takes basis, geometry, converged Psi4 MO coefficients"""
    ndocc = 1   #hard coded
    full_basis = torch.cat((basis,basis))
    nbf = torch.numel(full_basis)
    nbf_per_atom = torch.tensor([basis.size()[0],basis.size()[0]])
    charge_per_atom = torch.tensor([1.0,1.0])
    Enuc = nuclear_repulsion(geom[0], geom[1])

    S, T, V = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
    A = orthogonalizer(S)
    G = vectorized_tei(full_basis,geom,nbf_per_atom)
    #DETACH
    Enuc = Enuc.detach()
    #S = S.detach()
    T = T.detach()
    V = V.detach()
    #A = A.detach()
    G = G.detach()

    H = T + V
    Cocc = C[:, :ndocc]
    D = torch.matmul(Cocc, Cocc.T)
    # Start
    J = torch.tensordot(G, D, dims=([2,3], [0,1])) 
    K = torch.tensordot(G, D, dims=([1,3], [0,1])) 
    F = H + J * 2 - K
    E_scf = torch.tensordot(F + H, D, dims=([0,1],[0,1])) + Enuc 
    #print(E_scf)
    #grad, hess = differentiate(E_scf, geomlist, order=2)
    #print(grad)
    #print(hess)

    Fp = torch.chain_matmul(A, F, A)
    eps, Cp = torch.symeig(Fp, eigenvectors=True)       
    C = torch.matmul(A, Cp)
    #TODO
    #C = C.detach()
    #TODO
    Cocc = C[:, :ndocc]
    D = torch.matmul(Cocc, Cocc.t())
    J = torch.tensordot(G, D, dims=([2,3], [0,1])) 
    K = torch.tensordot(G, D, dims=([1,3], [0,1])) 
    F = H + J * 2 - K
    E_scf = torch.tensordot(F + H, D, dims=([0,1],[0,1])) + Enuc 
    #jac = jacobian(E_scf,C)
    #print(jac)
    #print(E_scf)
    #grad, hess = differentiate(E_scf, geomlist, order=2)
    #print(grad)
    #print(hess)

    for i in range(10):
        J = torch.tensordot(G, D, dims=([2,3], [0,1])) 
        K = torch.tensordot(G, D, dims=([1,3], [0,1])) 
        F = H + J * 2 - K
        E_scf = torch.tensordot(F + H, D, dims=([0,1],[0,1])) + Enuc 
        jac = jacobian(C,geom)
        #print(jac)
        print(jac.size())
        #grad, hess = differentiate(E_scf, geomlist, order=2)
        #print(grad)
        #print(hess)
        Fp = torch.chain_matmul(A, F, A)
        eps, Cp = torch.symeig(Fp, eigenvectors=True)       
        C = torch.matmul(A, Cp)
        #C = C.detach()
        #print(torch.autograd.grad(C2, geom, grad_outputs=torch.ones_like(C2), create_graph=True))
        Cocc = C[:, :ndocc]
        D = torch.matmul(Cocc, Cocc.t())






#C2 = torch.from_numpy(np.load('C2.npy'))
#C4 = np.load('C4.npy')
#C4 = C4.tolist()
#Clist = [torch.tensor(i, requires_grad=True) for i in C4]
##Clist = [torch.tensor(i, requires_grad=False) for i in C4]
D4 = np.load('D4.npy')
D4 = D4.tolist()
Dlist = [torch.tensor(i, requires_grad=True) for i in D4]
D4 = torch.stack(Dlist).reshape(16,16)
hartree_fock2(basis4,geom,D4)
