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

    def oei(self,geom):
        ndocc=torch.tensor(1)
        convergence=1e-8
        full_basis = torch.cat((basis2,basis2))
        nbf = torch.numel(full_basis)
        nbf_per_atom = torch.tensor([basis2.size()[0],basis2.size()[0]])
        charge_per_atom = torch.tensor([1.0,1.0])
        S, T, V = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
        return T + V

    def the_rest(self, geom, H):
        ndocc=torch.tensor(1)
        convergence=1e-8
        full_basis = torch.cat((basis2,basis2))
        nbf = torch.numel(full_basis)
        nbf_per_atom = torch.tensor([basis2.size()[0],basis2.size()[0]])
        charge_per_atom = torch.tensor([1.0,1.0])
        S, T2, V2 = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
        A = orthogonalizer(S)
        G = vectorized_tei(full_basis,geom,nbf_per_atom)
        Enuc = nuclear_repulsion(geom[0], geom[1])
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
        #return E_scf, eps, C
        return E_scf

    def forward(self, geom, row_idx):
        #H = self.oei(geom)
        H = cp(self.oei, geom, row_idx=row_idx, preserve_rng_state=False)
        E_scf = self.the_rest(geom, H)

        #H = self.oei(geom)
        #E_scf = cp(self.the_rest, geom, H, row_idx=row_idx, preserve_rng_state=False)

        #H = cp(self.oei, geom, row_idx=row_idx, preserve_rng_state=False)
        #E_scf = cp(self.the_rest, geom, H, row_idx=row_idx, preserve_rng_state=False)
        return E_scf

    #def compute_integrals(self, geom):
    #    ndocc=torch.tensor(1)
    #    convergence=1e-8
    #    full_basis = torch.cat((basis2,basis2))
    #    nbf = torch.numel(full_basis)
    #    nbf_per_atom = torch.tensor([basis2.size()[0],basis2.size()[0]])
    #    charge_per_atom = torch.tensor([1.0,1.0])
    #    S, T, V = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
    #    A = orthogonalizer(S)
    #    H = T + V
    #    G = vectorized_tei(full_basis,geom,nbf_per_atom)
    #    return H,A,G

    #def forward(self, geom, row_idx):
    #    ndocc=torch.tensor(1)
    #    convergence=1e-8
    #    full_basis = torch.cat((basis2,basis2))
    #    nbf = torch.numel(full_basis)
    #    nbf_per_atom = torch.tensor([basis2.size()[0],basis2.size()[0]])
    #    charge_per_atom = torch.tensor([1.0,1.0])
    #    H = cp(self.oei, geom, row_idx=row_idx, preserve_rng_state=False)
    #    #H, A, G = cp(self.compute_integrals, geom, row_idx=row_idx, preserve_rng_state=False)


    #    S, T2, V2 = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
    #    A = orthogonalizer(S)
    #    G = vectorized_tei(full_basis,geom,nbf_per_atom)


    #    Enuc = nuclear_repulsion(geom[0], geom[1])
    #    D = torch.zeros((nbf,nbf), requires_grad=True)
    #    for i in range(20):
    #        J = torch.einsum('pqrs,rs->pq', G, D)
    #        K = torch.einsum('prqs,rs->pq', G, D)
    #        F = H + J * 2 - K
    #        Fp = torch.chain_matmul(A, F, A)
    #        eps, C2 = torch.symeig(Fp, eigenvectors=True)
    #        C = torch.matmul(A, C2)
    #        Cocc = C[:, :ndocc]
    #        D = torch.einsum('pi,qi->pq', Cocc, Cocc)
    #        E_scf = torch.einsum('pq,pq->', F + H, D) + Enuc
    #    #return E_scf, eps, C
    #    return E_scf

hessian = RHF()
E = hessian(mygeom, 2)
#print(E)
E.backward(create_graph=True, retain_graph=True)


#print("DING DING DING")
print(mygeom.grad)
#hessian_row = mygeom.grad.clone().flatten()
#print(hessian_row)
#mygeom.grad.zero_()


