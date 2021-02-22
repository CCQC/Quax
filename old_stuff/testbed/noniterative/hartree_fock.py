import torch
import numpy as np
import opt_einsum

from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion
from differentiate import differentiate,jacobian

torch.set_printoptions(precision=5)
torch.set_default_dtype(torch.float64)

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

#C2 = np.load('C2.npy').flatten().tolist()
#C2 = np.load('C2.npy')
# ndocc hard coded
#C2 = C2[:, :1].flatten().tolist()

def hook(inp_grad):
    print("Backward hook")
    print(inp_grad)
    #real_grad = torch.zeros_like(inp_grad)
    #print("Backward hook changed grad to")
    #print(real_grad)
    #return real_grad
    

#C4 = np.load('C4.npy').flatten().tolist()
C = torch.from_numpy(np.load('C4.npy'))
#C.requires_grad_()
#C.register_hook(hook)

#Cgrad = torch.load('Cgrad')
#C.grad = Cgrad
#Clist = [torch.tensor(i, requires_grad=True) for i in C4]
##Clist = [torch.tensor(i, requires_grad=False) for i in C2]
#for c in Clist:
#    c.grad = torch.tensor(2.0)
#C = torch.stack(Clist).reshape(16,16)
##C = torch.stack(Clist).reshape(8,8)


#@torch.jit.script
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
    #TODO
    H = T + V
    Cocc = C[:, :ndocc]
    D = torch.matmul(Cocc, Cocc.T)
    D.requires_grad_()
    D.register_hook(hook)
    # Start
    J = torch.tensordot(G, D, dims=([2,3], [0,1])) 
    K = torch.tensordot(G, D, dims=([1,3], [0,1])) 
    F = H + J * 2 - K
    Fp = torch.chain_matmul(A, F, A)
    eps, Cp = torch.symeig(Fp, eigenvectors=True)       
    C = torch.matmul(A, Cp)
    Cocc = C[:, :ndocc]
    D = torch.matmul(Cocc, Cocc.t())

    #D.register_hook(hook)
    if iterate == True:
        for i in range(10):
            J = torch.tensordot(G, D, dims=([2,3], [0,1])) 
            K = torch.tensordot(G, D, dims=([1,3], [0,1])) 
            F = H + J * 2 - K
            E_scf = torch.tensordot(F + H, D, dims=([0,1],[0,1])) + Enuc 
            print(i)
            # Backward calls
            grad = torch.autograd.grad(E_scf, geom, create_graph=True)[0]
            one_hess = torch.autograd.grad(grad[0][2], geom, create_graph=True)
            Fp = torch.chain_matmul(A, F, A)
            eps, Cp = torch.symeig(Fp, eigenvectors=True)       
            C = torch.matmul(A, Cp)
            Cocc = C[:, :ndocc]
            D = torch.matmul(Cocc, Cocc.t())
            #D.register_hook(hook)
        E_scf = torch.tensordot(F + H, D, dims=([0,1],[0,1])) + Enuc 
        G.register_hook(hook)
        #Enuc.register_hook(hook)


        grad = torch.autograd.grad(E_scf, geom, create_graph=True)[0]
        #one_hess = torch.autograd.grad(grad[0][2], geom, create_graph=True)
        #grad, hess = differentiate(E_scf, geomlist, order=2)
        #print(grad[-1])
        #print(hess[2,2])

        #Cjac = jacobian(E_scf,C)
        #torch.save(Cjac,'Cgrad')
        #print(Cjac)
        #geomjac = jacobian(C,geom)
        #contract = torch.einsum('ij,ijkl->kl',Cjac,geomjac)
        #print('C contribution to gradient after')
        #print(contract)
            
    #track_var = C
    #newg = torch.autograd.grad(E_scf, track_var, create_graph=True)[0]
    #if i > 0:
    #    print('stationary?',torch.allclose(newg,oldg))
    #oldg = torch.autograd.grad(E_scf, track_var, create_graph=True)[0]
    #if i == 0:
    #    firstg = torch.autograd.grad(E_scf, track_var, create_graph=True)[0]
    #if i == 8:
    #    laterg = torch.autograd.grad(E_scf, track_var, create_graph=True)[0]
    #    print("long term stationary?", torch.allclose(firstg,laterg))

    #Fp = torch.chain_matmul(A, F, A)
    #eps, C2 = torch.symeig(Fp, eigenvectors=True)       
    #C = torch.matmul(A, C2)
    #return E_scf, eps, G, C
    return E_scf

#E_scf, eps, G, C = hartree_fock(basis4,geom,C, iterate=False)
E_scf = hartree_fock(basis4,geom,C, iterate=True)
print(E_scf)
#grad, hess = differentiate(E_scf, geomlist, order=2)
#print(grad)
#print(hess)


