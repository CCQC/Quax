import torch
import numpy as np
from hartree_fock import hartree_fock
from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion
from differentiate import differentiate 
torch.set_default_dtype(torch.float64)

def spin_block_tei(I):
    identity = np.eye(2)
    I = np.kron(identity, I)
    return np.kron(identity, I.T)

def cepa_corr(eps, G, C, ndocc, convergence=1e-9):
    eps = torch.cat((eps,eps))
    nbf = C.size()[0]
    nocc = 2*ndocc 
    nvirt = torch.numel(eps) - nocc
    Czero = torch.zeros_like(C)
    Ctmp = torch.stack((C, Czero),dim=0).reshape(2*nbf,nbf)
    Ctmp2 = torch.stack((Czero, C),dim=0).reshape(2*nbf,nbf)
    Cs = torch.cat((Ctmp,Ctmp2),dim=1)
    Cs = Cs[:, torch.argsort(eps)]
    eps = torch.sort(eps)[0]
    # New way: Generate a spin block of 1's and 0's and multiply by 2x expanded G
    dum = torch.ones_like(G).numpy()
    dum2 = torch.tensor(spin_block_tei(dum))
    Gexpanded = G.repeat(2,2,2,2)
    G2 = Gexpanded * dum2
    
    tmp = G2.permute(0, 2, 1, 3)
    #gao = tmp - tmp.permute(0,1,3,2)
    G2 = tmp - tmp.permute(0,1,3,2)

    o = slice(None, nocc)
    v = slice(nocc, None)
    eps_occ, eps_vir = eps[:nocc], eps[nocc:]
    # construct full transformed ERI matrix since we are lazy
    G2 = torch.einsum('pjkl, pi -> ijkl',
         torch.einsum('pqkl, qj -> pjkl',
         torch.einsum('pqrl, rk -> pqkl',
         torch.einsum('pqrs, sl -> pqrl', G2, Cs), Cs), Cs), Cs)
    e_denom = 1 / (-eps_vir.reshape(-1, 1, 1, 1) - eps_vir.reshape(-1, 1, 1) + eps_occ.reshape(-1, 1) + eps_occ)

    # New algorithm: Stores 4 4-index intermediates. 
    # G2 is of dimension (2*nbf,2*nbf,2*nbf,2*nbf) 
    # e_denom is of dimension (2*nbf - nocc, 2*nbf - nocc, nocc, nocc) 
    # t_amp is of dimension (2*nbf - nocc, 2*nbf - nocc, nocc, nocc) 
    # new is of dimension (2*nbf - nocc, 2*nbf - nocc, nocc, nocc) 
    # Each iteration thus incurs appoximately [(2*nbf)**4 + 3*(2*nbf)**2] * 64 bits for smallish system (nocc is small)
    # For 100 basis functions, 1.6 GB per iteration
    # For 200 basis functions, 25.6 GB per iteration
    t_amp = torch.zeros((nvirt, nvirt, nocc, nocc))
    for i in range(20):
        new = G2[v, v, o, o].clone()
        new.add_(0.5 * torch.einsum('abcd, cdij -> abij', G2[v, v, v, v], t_amp))
        new.add_(0.5 * torch.einsum('klij, abkl -> abij', G2[o, o, o, o], t_amp))
        cepa3a = torch.einsum('akic, bcjk -> abij', G2[v, o, o, v], t_amp)
        new.add_(cepa3a)
        new.add_(-cepa3a.permute(1, 0, 2, 3))
        new.add_(-cepa3a.permute(0, 1, 3, 2))
        new.add_(cepa3a.permute(1, 0, 3, 2))
        new.mul_(e_denom)
        # Evaluate Energy
        cepa_correlation_e = 0.25 * torch.einsum('ijab, abij ->', G2[o, o, v, v], new)
        t_amp = new
        print("CEPA0 Energy:", cepa_correlation_e.item())
        if i>1:
            if torch.allclose(cepa_correlation_e, old_cepa_correlation_e, rtol=convergence, atol=convergence):
                print("{} Iterations Required".format(i))
                break
        old_cepa_correlation_e = cepa_correlation_e
    return cepa_correlation_e


tmpgeom = [0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]
geomlist = [torch.tensor(i, requires_grad=True) for i in tmpgeom]
geom = torch.stack(geomlist).reshape(2,3)
# Define basis set (no coefficients, just exponents) and charge info
basis = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001], requires_grad=False) # Hydrogen atom basis set
full_basis = torch.cat((basis,basis))                                                   # H2 total basis set
nbf_per_atom = torch.tensor([basis.size()[0],basis.size()[0]])
charge_per_atom = torch.tensor([1.0,1.0])
ndocc = 1

E, eps, C, G = hartree_fock(geom,ndocc,full_basis,nbf_per_atom,charge_per_atom,convergence=1e-8)
cepa_energy = cepa_corr(eps, G, C, ndocc)



