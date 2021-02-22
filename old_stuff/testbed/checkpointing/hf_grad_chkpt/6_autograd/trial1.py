import torch 
import torch.nn as nn
import torch.utils.checkpoint as CP

from my_checkpoint import hesscheckpoint as cp
import numpy as np
from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion

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

    def integrals(self,geom):
        self.ndocc=torch.tensor(1)
        self.convergence=1e-8
        full_basis = torch.cat((basis2,basis2))
        self.nbf = torch.numel(full_basis)
        nbf_per_atom = torch.tensor([basis2.size()[0],basis2.size()[0]])
        charge_per_atom = torch.tensor([1.0,1.0])
        S, T, V = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
        A = orthogonalizer(S)
        G = vectorized_tei(full_basis,geom,nbf_per_atom)
        H = T + V
        Enuc = nuclear_repulsion(geom[0], geom[1])
        return H, A, G, Enuc
        
    def the_rest(self, H, A, G, Enuc):
        D = torch.zeros((self.nbf,self.nbf), requires_grad=True)
        for i in range(20):
            J = torch.einsum('pqrs,rs->pq', G, D)
            K = torch.einsum('prqs,rs->pq', G, D)
            F = H + J * 2 - K
            Fp = torch.chain_matmul(A, F, A)
            eps, C2 = torch.symeig(Fp, eigenvectors=True)
            C = torch.matmul(A, C2)
            Cocc = C[:, :self.ndocc]
            D = torch.einsum('pi,qi->pq', Cocc, Cocc)
            E_scf = torch.einsum('pq,pq->', F + H, D) + Enuc
        return E_scf

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


    def forward(self, geom, row_idx):
        #E_scf = cp(self.everything, geom, row_idx=row_idx, preserve_rng_state=False)

        H, A, G, Enuc = cp(self.integrals, geom, row_idx=row_idx, preserve_rng_state=False)
        E_scf = self.the_rest(H, A, G, Enuc)

        #H, A, G, Enuc = cp(self.integrals, geom, row_idx=row_idx, preserve_rng_state=False)
        #E_scf = cp(self.the_rest, H, A, G, Enuc, row_idx=row_idx, preserve_rng_state=False)

        #H, A, G, Enuc = self.integrals(geom) 
        #E_scf = cp(self.the_rest, H, A, G, Enuc, row_idx=row_idx, preserve_rng_state=False)
        return E_scf


hessian = RHF()
#E = hessian(mygeom, 5)
E = hessian(mygeom, 2)

def hesshook(inp_grad):
    print("Input to hook:", inp_grad)
    #hess = torch.autograd.grad(inp_grad.flatten()[2], mygeom, retain_graph=True)[0]
    hess = torch.autograd.grad(inp_grad.flatten()[2], mygeom, create_graph=True)[0]
    #hess = torch.autograd.grad(inp_grad.flatten()[2], mygeom, create_graph=True)[0]
    #hess = torch.autograd.grad(inp_grad.flatten()[2], , retain_graph=True)[0]
    print("Output of hook:", hess)
    return hess
mygeom.register_hook(hesshook)


#print(E)
#E.backward(create_graph=True, retain_graph=True)
E.backward(create_graph=True)


#g = torch.load('grads.pt')
#h = torch.load('hess.pt')
#print(g)
#print(h)
#print("HESSIAN ROW")
#print(g + h)

#print("DING DING DING")
#print(mygeom.grad)
#hessian_row = mygeom.grad.clone().flatten()
#print(hessian_row)
#mygeom.grad.zero_()


