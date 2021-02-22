import torch
import numpy as np
import opt_einsum

from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion
from differentiate import differentiate

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
# Load converged Psi4 Densities for basis sets 0 through 4
F0 = torch.from_numpy(np.load('psi4_fock_matrices/F0.npy'))
F1 = torch.from_numpy(np.load('psi4_fock_matrices/F1.npy'))
F2 = torch.from_numpy(np.load('psi4_fock_matrices/F2.npy'))
F3 = torch.from_numpy(np.load('psi4_fock_matrices/F3.npy'))
F4 = torch.from_numpy(np.load('psi4_fock_matrices/F4.npy'))

#@torch.jit.script
def hartree_fock_old(basis,geom,Finit):
    """Takes basis, geometry, converged Psi4 Fock matrix wfn.Fa()"""
    ndocc = 1                 #hard coded
    full_basis = torch.cat((basis,basis))
    nbf = torch.numel(full_basis)
    nbf_per_atom = torch.tensor([basis.size()[0],basis.size()[0]])
    charge_per_atom = torch.tensor([1.0,1.0])
    Enuc = nuclear_repulsion(geom[0], geom[1])
    S, T, V = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
    A = orthogonalizer(S)
    G = vectorized_tei(full_basis,geom,nbf_per_atom)
    H = T + V
    Fp = torch.chain_matmul(A, Finit, A)
    eps, C2 = torch.symeig(Fp, eigenvectors=True)       
    C = torch.matmul(A, C2)
    Cocc = C[:, :ndocc]

    # This density is now 'connected' in a computation graph to the input geometry, compute energy
    D = torch.einsum('pi,qi->pq', Cocc, Cocc)
    G2 = vectorized_tei(full_basis,geom,nbf_per_atom)
    S2, T2, V2 = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
    H2 = T2 + V2
    J = torch.einsum('pqrs,rs->pq', G2, D)
    K = torch.einsum('prqs,rs->pq', G2, D)
    F = H2 + J * 2 - K
    E_scf = torch.einsum('pq,pq->', F + H2, D) + Enuc #OLD END
    #C = C.flatten()
    #print(C)
    #print(torch.autograd.grad(E_scf,C))
    #print(E_scf)

    ### Just to be safe:
    #Fp = torch.chain_matmul(A, F, A)
    #eps, C2 = torch.symeig(Fp, eigenvectors=True)       
    #C = torch.matmul(A, C2)
    #Cocc = C[:, :ndocc]
    #D = torch.einsum('pi,qi->pq', Cocc, Cocc)

    # Do SOSCF? I guess?
    #for i in range(20):
    #    moF = torch.einsum('ui,vj,uv', C, C, F)
    #    # electronic gradient
    #    gn = -4 * moF[:ndocc,ndocc:]
    #    #MO = torch.einsum('pqrs,sl,rk,qj,pi->ijkl',G, C[:, :ndocc], C, C, C)
    #    MO = torch.einsum('pqrs,pi,qj,rk,sl->ijkl',G, C[:, :ndocc], C, C, C)

    #    nvirt = C.size()[0] - ndocc
    #    # electronic Hessian
    #    Biajb = 4*(torch.einsum('ab,ij->iajb', moF[ndocc:, ndocc:], torch.diag(torch.ones(ndocc))) -\
    #            torch.einsum('ij,ab->iajb', moF[:ndocc:, :ndocc], torch.diag(torch.ones(nvirt))) +\
    #            4 * MO[:, ndocc:, :ndocc, ndocc:] -\
    #            MO[:, ndocc:, :ndocc, ndocc:].permute(2,1,0,3) -\
    #            MO[:, :ndocc, ndocc:, ndocc:].permute(0,2,1,3))
    #    Binv = torch.inverse(Biajb.reshape(ndocc * nvirt,-1)).reshape(ndocc, nvirt, ndocc, nvirt)

    #    x = torch.einsum('iajb,ia->jb', Binv, gn)
    #    U = torch.zeros_like(C)
    #    U[:ndocc, ndocc:] = x
    #    U[ndocc:, :ndocc] = -x.T
    #    U = U + 0.5 * torch.matmul(U, U)
    #    U = U + torch.diag(torch.ones(nbf))
    #    U, r = torch.qr(U.T)

    #    C = torch.matmul(C,U)
    #Cocc = C[:, :ndocc]

    #D = torch.einsum('pi,qi->pq', Cocc, Cocc)
    #J = torch.einsum('pqrs,rs->pq', G, D)
    #K = torch.einsum('prqs,rs->pq', G, D)
    #F = H + J * 2 - K
    #E_scf = torch.einsum('pq,pq->', F + H, D) + Enuc #OLD END
    return E_scf

#@torch.jit.script
def hartree_fock(basis,geom,exact_energy,convergence=torch.tensor(1e-9)):
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
    #E_scf = torch.tensor(0)
    #real_D = torch.from_numpy(np.load("psi4_fock_matrices/test/D2.npy"))
    #print(core_D)
    #print(real_D)
    #factor = real_D / core_D
    #D = factor * core_D

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

        #Ctmp = C.flatten()
        #print(torch.autograd.grad(E_scf,C,create_graph=True))
        print(torch.autograd.grad(eps[0],F,create_graph=True))

        print(E_scf)
        if torch.allclose(E_scf, exact_energy, rtol=convergence, atol=convergence):
            break
        #    return E_scf, eps, C, G
    return E_scf, eps, C, G


exact0 = torch.tensor(-0.931283011458994) 
exact1 = torch.tensor(-0.971685591404988)
exact2 = torch.tensor(-1.060859783988007)
exact3 = torch.tensor(-1.0687016869345958)
exact4 = torch.tensor(-1.06888692908086)

#F2.requires_grad = True
E = hartree_fock_old(basis2,geom,F2)
#E, eps, C, G = hartree_fock(basis2,geom,exact2)


grad, hess = differentiate(E, geomlist, order=2)
print(E)
print(grad)
print(hess)


