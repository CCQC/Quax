import torch
import numpy as np
np.set_printoptions(linewidth=200)
import opt_einsum
torch.set_default_dtype(torch.float64)

from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion
from differentiate import differentiate 
from hartree_fock import hartree_fock

# Define coordinates in Bohr as Torch tensors, turn on gradient tracking.  
tmpgeom = [0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]
geomlist = [torch.tensor(i, requires_grad=True) for i in tmpgeom]
#geomlist = [torch.tensor(i, requires_grad=False) for i in tmpgeom]
geom = torch.stack(geomlist).reshape(2,3)
# Define some basis function exponents
basis0 = torch.tensor([0.5], requires_grad=False)                                        # 2 total basis functions for H2
basis1 = torch.tensor([0.5, 0.4], requires_grad=False)                                   # 4 total basis functions for H2
basis2 = torch.tensor([0.5, 0.4, 0.3, 0.2], requires_grad=False)                         # 8 total basis functions for H2
basis3 = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05], requires_grad=False)              #12 total basis functions for H2
basis4 = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001], requires_grad=False) #16 total basis functions for H2
basis5 = basis4.repeat(2)                                                                #32 total basis functions for H2
basis6 = basis5.repeat(2)                                                                #64 total basis functions for H2
print(basis5)
#basis5 = torch.rand(50)

def spin_block_tei(I):
    identity = np.eye(2)
    I = np.kron(identity, I)
    return np.kron(identity, I.T)

def cepa_corr(eps, G, C, exact_energy,convergence=1e-10):
    print(G.size())
    eps = torch.cat((eps,eps))
    nocc = 2 #hard coded
    nvirt = torch.numel(eps) - nocc
    nbf = C.size()[0]

    Czero = torch.zeros_like(C)
    Ctmp = torch.stack((C, Czero),dim=0).reshape(2*nbf,nbf)
    Ctmp2 = torch.stack((Czero, C),dim=0).reshape(2*nbf,nbf)
    Cs = torch.cat((Ctmp,Ctmp2),dim=1)

    Cs = Cs[:, torch.argsort(eps)]
    eps = torch.sort(eps)[0]

    # Old way to spin block- no gradients
    #G2 = torch.from_numpy(spin_block_tei(G.detach().numpy()))
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
          #torch.einsum('pqrs, sl -> pqrl', gao, Cs), Cs), Cs), Cs)
          torch.einsum('pqrs, sl -> pqrl', G2, Cs), Cs), Cs), Cs)
    e_denom = 1 / (-eps_vir.reshape(-1, 1, 1, 1) - eps_vir.reshape(-1, 1, 1) + eps_occ.reshape(-1, 1) + eps_occ)

    t_amp = torch.zeros((nvirt, nvirt, nocc, nocc))
    #t_amp = torch.load('tamp4.pt')
    #oldg = torch.zeros_like(gmo)

    for i in range(30):
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
        print(cepa_correlation_e)

        ## Track stationarity of CEPA energy derivatives 
        #var_to_track = t_amp
        #newg = torch.autograd.grad(cepa_correlation_e,var_to_track,create_graph=True)[0]
        #if i > 0:
        #    print('stationary?',torch.allclose(newg,oldg))
        #oldg = torch.autograd.grad(cepa_correlation_e,var_to_track,create_graph=True)[0]
        #if i == 0:
        #    firstg = torch.autograd.grad(cepa_correlation_e,var_to_track,create_graph=True)[0]
        #if i == 45:
        #    laterg = torch.autograd.grad(cepa_correlation_e,var_to_track,create_graph=True)[0]
        #    print("long term stationary?", torch.allclose(firstg,laterg))
        # Check nuclear gradients
        #grad = torch.autograd.grad(eps, geom, grad_outputs=torch.ones_like(eps),create_graph=True)
        #grad = torch.autograd.grad(G, geom, grad_outputs=torch.ones_like(G),create_graph=True)[0]
        #grad = torch.autograd.grad(gmo, geom, grad_outputs=torch.ones_like(gmo),create_graph=True)[0]
     
        #if torch.allclose(cepa_correlation_e, exact_energy, rtol=convergence, atol=convergence):
        #    print("{} Iterations Required".format(i))
        #    break
        if i>1:
            if torch.allclose(cepa_correlation_e, old_cepa_correlation_e, rtol=convergence, atol=convergence):
                print("{} Iterations Required".format(i))
                break
        old_cepa_correlation_e = cepa_correlation_e

    torch.save(t_amp, 'tamp2.pt')
    return cepa_correlation_e


def cepa(basis,geom,exact_hf_energy,exact_cepa_energy,convergence=1e-9):
    """
    Takes basis, geometry, and converged Psi4 Hartree Fock energy.
    In order to get exact analytic hessians, for some reason,
    you MUST iterate, even if the energy is converged on the first iteration.
    All higher order derivatives are exact (to within 'convergence') when the energy naturally reaches the exact energy.
    """
    E_scf, eps, C, G = hartree_fock(basis,geom,exact_hf_energy,convergence=convergence)
    E_cepa = cepa_corr(eps, G, C, exact_cepa_energy, convergence) + E_scf
    return E_cepa

exact0 = torch.tensor(-0.931283011458994) 
exact1 = torch.tensor(-0.971685591404988)
exact2 = torch.tensor(-1.060859783988007)
exact3 = torch.tensor(-1.0687016869345958)
exact4 = torch.tensor(-1.06888692908086)
exact00 = torch.tensor(-0.02648863499316)
exact11 = torch.tensor(-0.02473604307415)
exact22 = torch.tensor(-0.02684340602564)
exact33 = torch.tensor(-0.02711468024242)
exact44 = torch.tensor(-0.02713049375575)


def benchmark(basis, geom, exacthf, exactcepa):
    E_cepa = cepa(basis,geom,exacthf,exactcepa,convergence=1e-8)
    #grad = torch.autograd.grad(E_cepa,geom,create_graph=True)
    #print(grad)
    #grad, hess = differentiate(E_cepa, geomlist, order=2)


#e = benchmark(torch.rand(30),geom,exact4,exact44)
#e = benchmark(basis4,geom,exact4,exact44)
#e = benchmark(basis2,geom,exact2,exact22)
e = benchmark(basis6,geom,exact4,exact44)
#e = benchmark(basis4,geom,exact4,exact33)
#print(e)

#E_cepa = cepa(basis1, geom, exact1)
#print(E_cepa)

#grad, hess = differentiate(E_cepa, geomlist, order=2)
#print(E_mp2)
#print(grad)
#print(hess)


