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
        self.ndocc=torch.tensor(1)
        convergence=1e-8
        self.full_basis = torch.cat((basis2,basis2))
        self.nbf = torch.numel(self.full_basis)
        self.nbf_per_atom = torch.tensor([basis2.size()[0],basis2.size()[0]])
        self.charge_per_atom = torch.tensor([1.0,1.0])

    def integrals(self,geom):
        S, T, V = vectorized_oei(self.full_basis, geom, self.nbf_per_atom, self.charge_per_atom)
        A = orthogonalizer(S)
        G = vectorized_tei(self.full_basis,geom,self.nbf_per_atom)
        H = T + V
        Enuc = nuclear_repulsion(geom[0], geom[1])
        return H, A, G, Enuc

    def hf_iter(self, H, A, G, Enuc, D):
        J = torch.einsum('pqrs,rs->pq', G, D)
        K = torch.einsum('prqs,rs->pq', G, D)
        F = H + J * 2 - K
        Fp = torch.chain_matmul(A, F, A)
        eps, C2 = torch.symeig(Fp, eigenvectors=True)
        C = torch.matmul(A, C2)
        Cocc = C[:, :self.ndocc]
        D = torch.einsum('pi,qi->pq', Cocc, Cocc)
        E_scf = torch.einsum('pq,pq->', F + H, D) + Enuc
        return E_scf, D

    def everything(self, geom):
        #H, A, G, Enuc = self.integrals(geom)
        H, A, G, Enuc = cp(self.integrals, geom, row_idx=self.row_idx, preserve_rng_state=False)
        D = torch.zeros((self.nbf,self.nbf), requires_grad=True)
        for i in range(20):
            E_scf, D = self.hf_iter(H,A,G,Enuc,D)
            print(E_scf)
        return E_scf

    def forward(self, geom, row_idx):
        self.row_idx = row_idx
        E_scf = cp(self.everything, geom, row_idx=row_idx, preserve_rng_state=False)
        #E_scf = self.everything(geom) 
        return E_scf

hessian = RHF()
E = hessian(mygeom, 2)
E.backward(create_graph=True, retain_graph=True)
#hessian_row = mygeom.grad.clone().flatten()
#print(hessian_row)
#mygeom.grad.zero_()


