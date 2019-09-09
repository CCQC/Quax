import torch
import numpy as np
from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion

class RHF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, geom, basis, C, ndocc):
        #ctx.save_for_backward(geom,basis,C,ndocc)
        full_basis = torch.cat((basis,basis))
        nbf = torch.numel(full_basis)
        nbf_per_atom = torch.tensor([basis.size()[0],basis.size()[0]])
        charge_per_atom = torch.tensor([1.0,1.0])
        Enuc = nuclear_repulsion(geom[0], geom[1])
        S, T, V = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
        A = orthogonalizer(S)
        G = vectorized_tei(full_basis,geom,nbf_per_atom)
        H = T + V
        Cocc = C[:, :ndocc]
        D = torch.matmul(Cocc, Cocc.T)
        J = torch.tensordot(G, D, dims=([2,3], [0,1])) 
        K = torch.tensordot(G, D, dims=([1,3], [0,1])) 
        F = H + J * 2 - K
        E_scf = torch.tensordot(F + H, D, dims=([0,1],[0,1])) + Enuc 
        ## warning: no epsilon computation
        ##Fp = torch.chain_matmul(A, F, A)
        ##eps1, Cp1 = torch.symeig(Fp1, eigenvectors=True)       
        #with torch.enable_grad():
        #    for i in range(5):
        #        J = torch.tensordot(G, D, dims=([2,3], [0,1])) 
        #        K = torch.tensordot(G, D, dims=([1,3], [0,1])) 
        #        F = H + J * 2 - K
        #        Fp = torch.chain_matmul(A, F, A)
        #        eps, Cp = torch.symeig(Fp, eigenvectors=True)       
        #        C = torch.matmul(A,Cp) 
        #        Cocc = C[:, :ndocc]
        #        D = torch.matmul(Cocc, Cocc.T)
        ctx.save_for_backward(geom,basis,C,ndocc)
        E_scf = torch.tensordot(F + H, D, dims=([0,1],[0,1])) + Enuc 
        return E_scf

    @staticmethod
    def backward(ctx, grad_output):
        geom, basis, C, ndocc = ctx.saved_tensors
        #"if backward is implemented with differentiable operations, 
        # then higher order derivatives will work."
        # Implement RHF gradients by hand
        # Compute HF nuclear gradient 
        # (Pseudocode for now)
        grad_geom = grad_output.expand(2,3)

        full_basis = torch.cat((basis,basis))
        nbf = torch.numel(full_basis)
        nbf_per_atom = torch.tensor([basis.size()[0],basis.size()[0]])
        charge_per_atom = torch.tensor([1.0,1.0])
        # It would be better to hand-implement autograd.Functions for each integral type
        with torch.enable_grad():  # Get automatic gradients of integrals
            Enuc = nuclear_repulsion(geom[0], geom[1])
            S, T, V = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
            G = vectorized_tei(full_basis,geom,nbf_per_atom)
            s_mo = torch.einsum('pq,pi,qj->ij', S, C, C)
            t_mo = torch.einsum('pq,pi,qj->ij', T, C, C)
            v_mo = torch.einsum('pq,pi,qj->ij', V, C, C)
            gmo = torch.einsum('pjkl, pi -> ijkl',
                  torch.einsum('pqkl, qj -> pjkl',
                  torch.einsum('pqrl, rk -> pqkl',
                  torch.einsum('pqrs, sl -> pqrl', G, C), C), C), C)

        Hao = T + V
        H = torch.einsum('uj,vi,uv',C,C,Hao)
        #gmo = torch.einsum('pjkl, pi -> ijkl',
        #      torch.einsum('pqkl, qj -> pjkl',
        #      torch.einsum('pqrl, rk -> pqkl',
        #      torch.einsum('pqrs, sl -> pqrl', G, C), C), C), C)
        # Physicist notation    
        gmophys = gmo.permute(0,2,1,3)
        F = H + 2 * torch.einsum('pmqm->pq', gmophys[:, :ndocc, :, :ndocc]) -\
            torch.einsum('pmmq->pq', gmophys[:, :ndocc, :ndocc, :])


        # A single cartesian derivative of a single OEI
        #2.0 * torch.einsum("ii->", deriv1_np[map_key][:occ,:occ])
        # These gradients CHANGE if you iterate
        nuc_grad = torch.autograd.grad(Enuc, geom,create_graph=True)[0]
        print("Nuc grad")
        print(nuc_grad)
        #s_ao_grad = torch.autograd.grad(S[:ndocc,:ndocc], geom, grad_outputs=torch.ones(1,1), create_graph=True)[0] #grad_outputs=torch.ones(nbf,nbf), create_graph=True)[0]
        ##s_ao_grad = torch.autograd.grad(S, geom, grad_outputs=torch.ones(nbf,nbf), create_graph=True)[0] #grad_outputs=torch.ones(nbf,nbf), create_graph=True)[0]
        ##s_ao_grad = torch.autograd.grad(S, geomlist[0], grad_outputs=torch.ones(nbf,nbf), create_graph=True)[0]
        #print('S ao')
        #print(s_ao_grad)
        #t_ao_grad = torch.autograd.grad(T, geom, grad_outputs=torch.ones(nbf,nbf), create_graph=True)[0]
        #print('T ao')
        #print(t_ao_grad)
        #v_ao_grad = torch.autograd.grad(V, geom, grad_outputs=torch.ones(nbf,nbf), create_graph=True)[0]
        #print('V ao'),
        #print(v_ao_grad)
        #print('oei ao')
        #print(s_ao_grad + t_ao_grad + v_ao_grad)
        #g_ao_grad = torch.autograd.grad(G, geom, grad_outputs=torch.ones(nbf,nbf,nbf,nbf), create_graph=True)[0]
        #print('tei ao')
        #print(g_ao_grad)
        

        # Warning: really inefficient
        s_mo_grad = jacobian(s_mo, geom)
        t_mo_grad = jacobian(t_mo, geom)
        v_mo_grad = jacobian(v_mo, geom)
        s_final = -2.0 * torch.einsum("ii,iijk->jk", F[:ndocc,:ndocc], s_mo_grad[:ndocc,:ndocc])
        print(s_final)
        #v_mo_grad = jacobian(v_mo, geom)
        #print(jac)
        #print(jac.size())
        #s_mo_grad = torch.autograd.grad(s_mo, geom, grad_outputs=torch.ones(nbf,nbf), create_graph=True)[0]
        # You really just need the JACOBIAN of each integral array and geom parameter d(integral)/dx for all integral and all x
        #tmpgeom = geom.flatten().repeat(nbf*nbf,1)
        #print(tmpgeom)
        #print(s_mo.size())
        #s_mo = s_mo.expand(6,-1,-1)
        #print(s_mo.size())
    
        #s_mo_grad = torch.autograd.grad(s_mo[:,:ndocc,:ndocc], geom, grad_outputs=torch.ones(6,1,1), create_graph=True)[0]
        #s_mo_grad = torch.autograd.grad(s_mo[:,:2,:2], geom, grad_outputs=torch.ones(6,1,1), create_graph=True)[0]
        #s_mo_grad = torch.autograd.grad(s_mo[:ndocc,:ndocc], geom, grad_outputs=torch.ones(1,1), create_graph=True)[0]
        #s_mo_grad = torch.autograd.grad(s_mo[:2,:2], geom, grad_outputs=torch.ones(2,2), create_graph=True)[0]
        #t_mo_grad = torch.autograd.grad(t_mo[:ndocc,:ndocc], geom, grad_outputs=torch.ones(1,1), create_graph=True)[0]
        #v_mo_grad = torch.autograd.grad(v_mo[:ndocc,:ndocc], geom, grad_outputs=torch.ones(1,1), create_graph=True)[0]
        #print(s_mo_grad * trace + v_mo_grad + t_mo_grad)
        #v_mo_grad = torch.autograd.grad(v_mo[:2,:2], geom, grad_outputs=torch.ones(2,2), create_graph=True)[0]
        #print(v_mo_grad)
        #print("OEI")
        #print(s_mo_grad + t_mo_grad + v_mo_grad)

        # This is correct, matches CFOUR TEI grad for some reason accident maybe? since its just one element
        g_mo_grad = torch.autograd.grad(gmo[:ndocc,:ndocc,:ndocc,:ndocc], geom, grad_outputs=torch.ones(1,1,1,1), create_graph=True)[0]
        print(g_mo_grad)

        #print(s_mo_grad + t_mo_grad + v_mo_grad + g_mo_grad + nuc_grad)
        #print(s_mo_grad + t_mo_grad + v_mo_grad + g_mo_grad + nuc_grad)
        #print(s_mo_grad + t_mo_grad + v_mo_grad + g_mo_grad )

        #t_ao_grad = torch.autograd.grad(T, geom, grad_outputs=torch.ones((nbf,nbf), create_graph=True))
        #v_ao_grad = torch.autograd.grad(V, geom, grad_outputs=torch.ones((nbf,nbf), create_graph=True))
            
        # Must return same number number of inputs
        return grad_geom, None, None, None

def jacobian(outputs, inputs, create_graph=True):
    """Computes the jacobian of outputs with respect to inputs

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """
    jac = outputs.new_zeros(outputs.size() + inputs.size()
                            ).view((-1,) + inputs.size())
    for i, out in enumerate(outputs.view(-1)):
        col_i = torch.autograd.grad(out, inputs, retain_graph=True,
                              create_graph=create_graph, allow_unused=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()

    return jac.view(outputs.size() + inputs.size())

#Testbed

# Alias the custom op
hartree_fock = RHF.apply
# Load basis and geometry, and MO coefficients
basis2 = torch.tensor([0.5, 0.4, 0.3, 0.2], requires_grad=False)
tmpgeom = [0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]
geomlist = [torch.tensor(i, requires_grad=True) for i in tmpgeom]
geom = torch.stack(geomlist).reshape(2,3)
C2 = np.load('C2.npy').flatten().tolist()
Clist = [torch.tensor(i, requires_grad=True) for i in C2]
#Clist = [torch.tensor(i, requires_grad=False) for i in C2]
C = torch.stack(Clist).reshape(8,8)

E_scf = hartree_fock(geom,basis2,C,torch.tensor(1))
print(E_scf)
grad = torch.autograd.grad(E_scf, geom)
print(grad)
