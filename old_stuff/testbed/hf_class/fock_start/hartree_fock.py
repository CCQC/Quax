import torch
import numpy as np
from integrals import orthogonalizer, vectorized_oei, vectorized_tei, nuclear_repulsion

class RHF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, geom, basis, F, ndocc):
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
        Fp = torch.chain_matmul(A, F, A)
        eps, C2 = torch.symeig(Fp, eigenvectors=True)       
        C = torch.matmul(A, C2)
        Cocc = C[:, :ndocc]
        # This density is now 'connected' in a computation graph to the input geometry, compute energy
        D = torch.einsum('pi,qi->pq', Cocc, Cocc)
        J = torch.einsum('pqrs,rs->pq', G, D)
        K = torch.einsum('prqs,rs->pq', G, D)
        F = H + J * 2 - K
        E_scf = torch.einsum('pq,pq->', F + H, D) + Enuc
        Fp = torch.chain_matmul(A, F, A)
        eps, C2 = torch.symeig(Fp, eigenvectors=True)       
        C = torch.matmul(A, C2)
        ## warning: no epsilon computation
        ##Fp = torch.chain_matmul(A, F, A)
        ##eps1, Cp1 = torch.symeig(Fp1, eigenvectors=True)       
        #for i in range(5):
        #    J = torch.tensordot(G, D, dims=([2,3], [0,1])) 
        #    K = torch.tensordot(G, D, dims=([1,3], [0,1])) 
        #    F = H + J * 2 - K
        #    Fp = torch.chain_matmul(A, F, A)
        #    eps, Cp = torch.symeig(Fp, eigenvectors=True)       
        #    C = torch.matmul(A,Cp) 
        #    Cocc = C[:, :ndocc]
        #    D = torch.matmul(Cocc, Cocc.T)
        ctx.save_for_backward(geom,basis,C,ndocc)
        #E_scf = torch.tensordot(F + H, D, dims=([0,1],[0,1])) + Enuc 
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
        #print(grad_output)

        full_basis = torch.cat((basis,basis))
        nbf = torch.numel(full_basis)
        nbf_per_atom = torch.tensor([basis.size()[0],basis.size()[0]])
        charge_per_atom = torch.tensor([1.0,1.0])
        # It would be better to hand-implement autograd.Functions for each integral type
        with torch.enable_grad():  # Get automatic gradients of integrals
            Enuc = nuclear_repulsion(geom[0], geom[1])
            S, T, V = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
            G = vectorized_tei(full_basis,geom,nbf_per_atom)
            # Transform first, then differentiate
            s_mo = torch.einsum('pq,pi,qj->ij', S, C, C)
            t_mo = torch.einsum('pq,pi,qj->ij', T, C, C)
            v_mo = torch.einsum('pq,pi,qj->ij', V, C, C)
            gmo = torch.einsum('pjkl, pi -> ijkl',
                  torch.einsum('pqkl, qj -> pjkl',
                  torch.einsum('pqrl, rk -> pqkl',
                  torch.einsum('pqrs, sl -> pqrl', G, C), C), C), C)

        Hao = T + V
        H = torch.einsum('uj,vi,uv',C,C,Hao)
        gmophys = gmo.permute(0,2,1,3)
        F = H + 2 * torch.einsum('pmqm->pq', gmophys[:, :ndocc, :, :ndocc]) -\
            torch.einsum('pmmq->pq', gmophys[:, :ndocc, :ndocc, :])

        nuc_grad = torch.autograd.grad(Enuc, geom,create_graph=True)[0]
            # Warning: really inefficient, high memory consumption. Need a more 'integral direct' approach, coordinate by coordinate mayb
        s_mo_grad = jacobian(s_mo, geom)
        t_mo_grad = jacobian(t_mo, geom)
        v_mo_grad = jacobian(v_mo, geom)
        # YIKES so expensive 
        g_mo_grad = jacobian(gmo, geom)
        # This is correct, matches CFOUR TEI grad for some reason accident maybe? since its just one element
        #TODO TODO TODO this is a cheap trick just for rapid testing. The above is the correct, general code for gradient #TODO #TODO #TODO
        #g_final = torch.autograd.grad(gmo[:ndocc,:ndocc,:ndocc,:ndocc], geom, grad_outputs=torch.ones(1,1,1,1), create_graph=True)[0]

        s_final = -2.0 * torch.einsum("ii,iijk->jk", F[:ndocc,:ndocc], s_mo_grad[:ndocc,:ndocc])
        print(s_final)
        t_final = 2.0 * torch.einsum("iijk->jk", t_mo_grad[:ndocc,:ndocc])  
        print(t_final)
        v_final = 2.0 * torch.einsum("iijk->jk", v_mo_grad[:ndocc,:ndocc])  
        print(v_final)
        g_final =  2.0 * torch.einsum("iijjkl->kl", g_mo_grad[:ndocc,:ndocc,:ndocc,:ndocc]) + -1.0 * torch.einsum("ijijkl->kl", g_mo_grad[:ndocc,:ndocc,:ndocc,:ndocc])
        print(g_final)
        gradient = s_final + t_final + v_final + g_final + nuc_grad

        #s_ao_grad = torch.autograd.grad(S, geom, grad_outputs=torch.ones(nbf,nbf), create_graph=True)[0] 
        #t_ao_grad = torch.autograd.grad(T, geom, grad_outputs=torch.ones(nbf,nbf), create_graph=True)[0]
        #v_ao_grad = torch.autograd.grad(V, geom, grad_outputs=torch.ones(nbf,nbf), create_graph=True)[0]
        #g_ao_grad = torch.autograd.grad(G, geom, grad_outputs=torch.ones(nbf,nbf,nbf,nbf), create_graph=True)[0]

        # Warning: really inefficient, high memory consumption. Need a more 'integral direct' approach, coordinate by coordinate mayb
        #nuc_grad = torch.autograd.grad(Enuc, geom,create_graph=True)[0]
        #s_mo_grad = jacobian(s_mo, geom)
        #t_mo_grad = jacobian(t_mo, geom)
        #v_mo_grad = jacobian(v_mo, geom)
        # CFOUR calls this the 'reorthonormalization gradient'
        #s_final = -2.0 * torch.einsum("ii,iijk->jk", F[:ndocc,:ndocc], s_mo_grad[:ndocc,:ndocc])
        #t_final = 2.0 * torch.einsum("iijk->jk", t_mo_grad[:ndocc,:ndocc])  
        #v_final = 2.0 * torch.einsum("iijk->jk", v_mo_grad[:ndocc,:ndocc])  

        # YIKES so expensive 
        #g_mo_grad = jacobian(gmo, geom)
        #g_final =  2.0 * torch.einsum("iijjkl->kl", g_mo_grad[:ndocc,:ndocc,:ndocc,:ndocc]) + -1.0 * torch.einsum("ijijkl->kl", g_mo_grad[:ndocc,:ndocc,:ndocc,:ndocc])

        # This is correct, matches CFOUR TEI grad for some reason accident maybe? since its just one element
        #TODO TODO TODO this is a cheap trick just for rapid testing. The above is the correct, general code for gradient #TODO #TODO #TODO
        #g_final = torch.autograd.grad(gmo[:ndocc,:ndocc,:ndocc,:ndocc], geom, grad_outputs=torch.ones(1,1,1,1), create_graph=True)[0]
        # Must return same number number of inputs
        return gradient, None, None, None

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
    #for i, out in enumerate(outputs.view(-1)):
    for i, out in enumerate(outputs.reshape(-1)):
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
F2 = np.load('F2.npy').flatten().tolist()
Flist = [torch.tensor(i, requires_grad=True) for i in F2]
#Flist = [torch.tensor(i, requires_grad=False) for i in F2]
F = torch.stack(Flist).reshape(8,8)

E_scf = hartree_fock(geom,basis2,F,torch.tensor(1))
print(E_scf)
grad = torch.autograd.grad(E_scf, geom)[0]
print(grad)
for g in grad.flatten(): 
    h = torch.autograd.grad(g, geom, create_graph=True)[0]
    print(h)
#h1 = torch.autograd.grad(grad[0,0
