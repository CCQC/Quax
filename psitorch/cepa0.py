import torch
import numpy as np
np.set_printoptions(linewidth=200)
import opt_einsum
torch.set_default_dtype(torch.float64)

from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion
from differentiate import differentiate 

def spin_block_tei(I):
    identity = np.eye(2)
    I = np.kron(identity, I)
    return np.kron(identity, I.T)

def cepa_corr(eps, G, C,ndocc):
    eps = torch.cat((eps,eps))
    nocc = 2*ndocc 
    nvirt = torch.numel(eps) - nocc
    nbf = C.size()[0]
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

    t_amp = torch.zeros((nvirt, nvirt, nocc, nocc))

    for i in range(20):
        # Collect terms
        mp2    = G2[v, v, o, o]
        cepa1  = (1 / 2) * torch.einsum('abcd, cdij -> abij', G2[v, v, v, v], t_amp)
        cepa2  = (1 / 2) * torch.einsum('klij, abkl -> abij', G2[o, o, o, o], t_amp)
        cepa3a = torch.einsum('akic, bcjk -> abij', G2[v, o, o, v], t_amp)
        cepa3b = -cepa3a.permute(1, 0, 2, 3)
        cepa3c = -cepa3a.permute(0, 1, 3, 2)
        cepa3d =  cepa3a.permute(1, 0, 3, 2)
        cepa3  =  cepa3a + cepa3b + cepa3c + cepa3d
        t_amp = e_denom * (mp2 + cepa1 + cepa2 + cepa3)
        # Evaluate Energy
        cepa_correlation_e = (1 / 4) * torch.einsum('ijab, abij ->', G2[o, o, v, v], t_amp)
        if i>1:
            if torch.allclose(cepa_correlation_e, old_cepa_correlation_e, rtol=convergence, atol=convergence):
                print("{} Iterations Required".format(i))
                break
        old_cepa_correlation_e = cepa_correlation_e
    return cepa_correlation_e


