import torch 
import torch.nn as nn
#import torch.utils.checkpoint as CP
#import torch.utils.checkpoint as cp
#from torch.utils.checkpoint import checkpoint as cp
from vanilla_checkpoint import checkpoint as cp
torch.set_printoptions(edgeitems=1)

#from my_checkpoint import hesscheckpoint as cp
import numpy as np
from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion

mygeom = torch.tensor([[0.000000000000,0.000000000000,-0.849220457955],[0.000000000000,0.000000000000,0.849220457955]], requires_grad=True)

# Define some basis function exponents
basis0 = torch.tensor([0.5], requires_grad=False)                                        # 2 total basis functions for H2
basis1 = torch.tensor([0.5, 0.4], requires_grad=False)                                   # 4 total basis functions for H2
basis2 = torch.tensor([0.5, 0.4, 0.3, 0.2], requires_grad=False)                         # 8 total basis functions for H2
basis3 = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05], requires_grad=False)              #12 total basis functions for H2
basis4 = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001], requires_grad=False) #16 total basis functions for H2


def hook(inp_grad):
   print("Gradient received by hook:")
   print(inp_grad)
   #print("updating hook gradient")
   #inp_grad.backward(gradient=torch.ones_like(inp_grad))
   #print(inp_grad)

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
        
    def forward(self, geom, row_idx):
        #H, A, G, Enuc = cp(self.integrals, geom, row_idx=row_idx, preserve_rng_state=False)
        H, A, G, Enuc = cp(self.integrals, geom)
        D = torch.zeros((self.nbf,self.nbf), requires_grad=True)
        D.register_hook(hook)
        for i in range(20):
            #E_scf = self.hf_iter(H, A, G, Enuc, D)
            #E_scf, D = cp(self.hf_iter, H, A, G, Enuc, D, row_idx=row_idx, preserve_rng_state=False)
            E_scf, D = cp(self.hf_iter, H, A, G, Enuc, D)
            D.register_hook(hook)

        #H, A, G, Enuc = cp(self.integrals, geom, row_idx=row_idx, preserve_rng_state=False)
        #E_scf = cp(self.the_rest, H, A, G, Enuc, row_idx=row_idx, preserve_rng_state=False)

        #H, A, G, Enuc = self.integrals(geom) 
        #E_scf = cp(self.the_rest, H, A, G, Enuc, row_idx=row_idx, preserve_rng_state=False)
        return E_scf


hessian = RHF()
E = hessian(mygeom, 2)
E.backward(create_graph=True, retain_graph=True)
print(E)
print(mygeom.grad)
