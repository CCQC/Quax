import torch 
import torch.nn as nn
from vanilla_checkpoint import checkpoint
import torch.utils.checkpoint as CP

from my_checkpoint import hesscheckpoint as cp
import numpy as np
from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion
from differentiate import differentiate

mygeom = torch.tensor([[0.000000000000,0.000000000000,-0.849220457955],[0.000000000000,0.000000000000,0.849220457955]], requires_grad=True)

# Define some basis function exponents
basis0 = torch.tensor([0.5], requires_grad=False)                                        # 2 total basis functions for H2
basis1 = torch.tensor([0.5, 0.4], requires_grad=False)                                   # 4 total basis functions for H2
basis2 = torch.tensor([0.5, 0.4, 0.3, 0.2], requires_grad=False)                         # 8 total basis functions for H2
basis3 = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05], requires_grad=False)              #12 total basis functions for H2
basis4 = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001], requires_grad=False) #16 total basis functions for H2

class RHF(nn.Module):
    def __init__(self):
        super(RHF, self).__init__()

    def hf_iter(self, D, H, A, G, Enuc):
        ndocc=torch.tensor(1)
        J = torch.einsum('pqrs,rs->pq', G, D)
        K = torch.einsum('prqs,rs->pq', G, D)
        F = H + J * 2 - K
        Fp = torch.chain_matmul(A, F, A)
        eps, C2 = torch.symeig(Fp, eigenvectors=True)
        C = torch.matmul(A, C2)
        Cocc = C[:, :ndocc]
        D = torch.einsum('pi,qi->pq', Cocc, Cocc)
        E_scf = torch.einsum('pq,pq->', F + H, D)
        return E_scf, D, eps, C

    def compute_tei(self, *inputs):
        G = vectorized_tei(*inputs)
        return G

    def full_hartree_fock(self, D, H, A,G, Enuc):
        ndocc=torch.tensor(1)
        convergence=1e-8
        full_basis = torch.cat((basis2,basis2))
        nbf = torch.numel(full_basis)
        for i in range(20):
            E_tmp, D, eps, C = self.hf_iter(D, H, A, G, Enuc)
            #E_tmp, D, eps, C = cp.checkpoint(self.hf_iter, D, H, G, A, ndocc)
            #E_tmp, D, eps, C = cp(self.hf_iter, D, H, G, A, ndocc)
            #E_tmp, D, eps, C = cp(self.hf_iter, D, H, G, A, ndocc)
            #E_scf = E_tmp + Enuc 
            E_scf = E_tmp + Enuc
            print(E_scf)
            if i>1:
                if torch.allclose(E_scf, old_E_scf, rtol=convergence, atol=convergence):
                    print("{} Iterations Required".format(i))
                    #return E_scf, eps, C
                    return E_scf 
            old_E_scf = E_scf

    def everything(self, geom):
        ndocc=torch.tensor(1)
        convergence=1e-8
        full_basis = torch.cat((basis2,basis2))
        nbf = torch.numel(full_basis)
        nbf_per_atom = torch.tensor([basis2.size()[0],basis2.size()[0]])
        charge_per_atom = torch.tensor([1.0,1.0])
        Enuc = nuclear_repulsion(geom[0], geom[1])
        S, T, V = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
        A = orthogonalizer(S)
        G = vectorized_tei(full_basis,geom,nbf_per_atom)
        H = T + V
        D = torch.zeros((nbf,nbf), requires_grad=True)
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
        return E_scf

    def forward(self, geom):
        #E_scf = self.everything(geom)
        #E_scf = CP.checkpoint(self.everything,geom)
        #E_scf = checkpoint(self.everything,geom)
        E_scf = cp(self.everything, geom)
        return E_scf

model = RHF()
E = model(mygeom)
print(E)
#E, eps, C = model(mygeom)

#grad = torch.autograd.grad(E,geom,create_graph=True)[0]
#hess = torch.autograd.grad(grad[0,2],geom,create_graph=True)[0]
#print(grad)
#print(hess)

#print(geom.is_leaf)
#print(basis4.is_leaf)
#E.backward(create_graph=True, retain_graph=True)
#gradient = geom.grad.clone()
#print(geom.grad)

#gradient = geom.grad.clone().flatten()
#z = gradient @ v
#z.backward()
#print(geom.grad)
#print(gradient.grad)

# TODO This works without checkpoint
E.backward(create_graph=True, retain_graph=True)
gradient = mygeom.grad.clone().flatten()
print(gradient)

#print('gradient')
#print(gradient)
#h1 = torch.autograd.grad(gradient[2], mygeom, create_graph=True)[0]
#print(h1)
#for g in gradient:
#    mygeom.grad.zero_()
#    g.backward(create_graph=True)
#    print(mygeom.grad)



