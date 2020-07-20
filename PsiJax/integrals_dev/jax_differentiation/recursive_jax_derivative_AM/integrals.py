import jax
from functools import partial
import numpy as onp
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=400)

#@jax.jit
def tei_setup(geom,basis, nbf_per_atom):
    nbf = basis.shape[0]
    nbf_per_atom = int(nbf / 2)
    #TODO
    # NotImplementedError: np.repeat implementation only supports scalar repeats
    #centers = np.repeat(geom, 32, axis=0) # TODO currently can only repeat each center the same number of times => only works for when all atoms have same # of basis functions
    centers = np.repeat(geom, nbf_per_atom, axis=0) # TODO currently can only repeat each center the same number of times => only works for when all atoms have same # of basis functions
    #norm = (2 * basis / np.pi)**(3/4)
    # Obtain miscellaneous terms 
    # (i,l,j,k) + (l,i,j,k) ---> (i+l,i+l,j+j,k+k) ---> (A+D,D+A,C+C,B+B) which is just (A+D,A+D,C+C,B+B)
    tmp1 = np.broadcast_to(basis, (nbf,nbf,nbf,nbf))
    aa_plus_bb = tmp1.transpose((0,3,1,2)) + tmp1.transpose((3,0,1,2))
    aa_times_bb = tmp1.transpose((0,3,1,2)) * tmp1.transpose((3,0,1,2))
    # Obtain gaussian product coefficients
    tmp2 = np.broadcast_to(centers, (nbf,nbf,nbf,nbf,3))
    AminusB = tmp2.transpose((0,3,1,2,4)) - tmp2.transpose((3,0,1,2,4))
    # 'dot' the cartesian dimension
    contract_AminusB = np.einsum('ijklm,ijklm->ijkl', AminusB,AminusB)
    c1 = np.exp(contract_AminusB * -aa_times_bb / aa_plus_bb)

    # Obtain gaussian product centers Rp = (aa * A + bb * B) / (aa + bb);  Rq = (cc * C + dd * D) / (cc + dd)
    weighted_centers = np.einsum('ijkl,ijklm->ijklm', tmp1, tmp2)
    tmpAB = weighted_centers.transpose((0,3,1,2,4)) + weighted_centers.transpose((3,0,1,2,4))
    Rp_minus_Rq = np.einsum('ijklm,ijkl->ijklm', tmpAB, 1/aa_plus_bb) -\
                  np.einsum('ijklm,ijkl->ijklm', tmpAB.transpose((2,3,0,1,4)), 1/aa_plus_bb.transpose((3,2,0,1)))
    boys_arg = np.einsum('ijklm,ijklm->ijkl', Rp_minus_Rq, Rp_minus_Rq) /\
               (1 / (aa_plus_bb) + 1 / (aa_plus_bb.transpose((3,2,0,1))))
    #boys_arg = jax.scipy.special.erf(np.sqrt(boys_arg + 1e-9)) * np.sqrt(np.pi) / (2 * np.sqrt(boys_arg + 1e-9))
    boys_arg = boys_eval(boys_arg)  # jarrett-enhanced boys function eval, reduces memory

    G = np.ones((nbf,nbf,nbf,nbf)) 

    @jax.jarrett
    def tei_finish(G):
        G *= (2 * np.pi**2)
        norm = (2 * basis / np.pi)**(3/4)
        G *= np.einsum('i,j,k,l',norm,norm,norm,norm)
        G *= (1 / (aa_plus_bb * aa_plus_bb.transpose((3,2,0,1))))
        G *= np.sqrt(np.pi / (aa_plus_bb + aa_plus_bb.transpose((3,2,0,1))))
        return G
    G = tei_finish(G)

    G *= c1 * c1.transpose((2,3,0,1)) * boys_arg
    return G

@jax.jarrett
def boys_eval(boys_arg):
    return jax.scipy.special.erf(np.sqrt(boys_arg + 1e-9)) * np.sqrt(np.pi) / (2 * np.sqrt(boys_arg + 1e-9))

#@jax.jit
def oei_setup(geom,basis,nbf_per_atom,charge_per_atom):
    # SETUP AND OVERLAP INTEGRALS
    nbf = basis.shape[0]
    nbf_per_atom = int(nbf / 2)
    # 'centers' are the cartesian centers ((nbf,3) array) corresponding to each basis function, in the same order as the 'basis' vector
    #TODO
    centers = np.repeat(geom, nbf_per_atom, axis=0) # TODO currently can only repeat each center the same number of times => only works for when all atoms have same # of basis functions
    # Construct Normalization constant product array, Na * Nb component
    norm = (2 * basis / np.pi)**(3/4)
    normtensor = np.outer(norm,norm) # outer product => every possible combination of Na * Nb
    # Construct pi / aa + bb ** 3/2 term
    aa_times_bb = np.outer(basis,basis)
    #aa_plus_bb = basis.expand(nbf,-1) + torch.transpose(basis.expand(nbf,-1),0,1) # doesnt copy data, unlike repeat(). may not work, but very efficient
    aa_plus_bb = np.broadcast_to(basis, (nbf,nbf)) + np.transpose(np.broadcast_to(basis, (nbf,nbf)), (1,0))
    term = (np.pi / aa_plus_bb) ** (3/2)
    ## Construct gaussian product coefficient array, c = exp(A-B dot A-B) * ((-aa * bb) / (aa + bb))
    tmpA = np.broadcast_to(centers, (nbf,nbf,3))
    AminusB = tmpA - np.transpose(tmpA, (1,0,2)) #caution: tranpose shares memory with original array. changing one changes the other
    AmBAmB = np.einsum('ijk,ijk->ij', AminusB, AminusB)
    coeff = np.exp(AmBAmB * (-aa_times_bb / aa_plus_bb))
    S = normtensor * coeff * term
    # KINETIC INTEGRALS
    P = aa_times_bb / aa_plus_bb
    T = S * (3 * P + 2 * P * P * -AmBAmB)
    # Construct gaussian product center array, R = (aa * A + bb * B) / (aa + bb)
    # First construct every possible sum of exponential-weighted cartesian centers, aa*A + bb*B 
    aatimesA = np.einsum('i,ij->ij', basis,centers)
    # This is a 3D tensor (nbf,nbf,3), where each row is a unique sum of two exponent-weighted cartesian centers
    numerator = aatimesA[:,None,:] + aatimesA[None,:,:]
    R = np.einsum('ijk,ij->ijk', numerator, 1/aa_plus_bb)
    ## Now we must subtract off the atomic coordinates, for each atom, introducing yet another dimension, where we expand according to number of atoms
    R_per_atom = np.broadcast_to(R, (geom.shape[0],) + R.shape)
    expanded_geom = np.transpose(np.broadcast_to(geom, (nbf,nbf) + geom.shape), (2,1,0,3))
    # Subtract off atom coordinates
    Rminusgeom = R_per_atom - expanded_geom
    # Now contract along the coordinate dimension, and weight by aa_plus_bb. This is the boys function argument.
    contracted = np.einsum('ijkl,ijkl->ijk', Rminusgeom,Rminusgeom)
    boys_arg = np.einsum('ijk,jk->ijk', contracted, aa_plus_bb)
    #Vtmp = normtensor * coeff * 2 * np.pi / aa_plus_bb
    #boys_arg = jax.scipy.special.erf(np.sqrt(boys_arg + 1e-9)) * np.sqrt(np.pi) / (2 * np.sqrt(boys_arg + 1e-9))
    boys_arg = boys_eval(boys_arg) 
    Fcharge = -charge_per_atom[:,None,None] * boys_arg[:,...]
    Ffinal = np.sum(Fcharge, axis=0)
    V = normtensor * coeff * 2 * np.pi / aa_plus_bb  * Ffinal
    return S, T, V

def nuclear_repulsion(atom1, atom2):
    ''' warning : hard coded for H2'''
    Za = 1.0
    Zb = 1.0
    return Za*Zb / np.linalg.norm(atom1-atom2)

def orthogonalizer(S):
    '''Compute overlap to the negative 1/2 power'''
    # STABLE FOR SMALL EIGENVALUES
    #eigval, eigvec = np.linalg.eigh(S)
    #cutoff = 1.0e-12
    #above_cutoff = (abs(eigval) > cutoff * np.max(abs(eigval)))
    #val = 1 / np.sqrt(eigval[above_cutoff])
    #vec = eigvec[:, above_cutoff]
    #A = vec.dot(np.diag(val)).dot(vec.T)

    ## STABLE FOR SMALL EIGENVALUES
    eigval, eigvec = np.linalg.eigh(S)
    #cutoff = 1.0e-12
    #above_cutoff = (abs(eigval) > cutoff * np.max(abs(eigval)))
    #TODO TODO hard coded
    val = 1 / np.sqrt(eigval[-8:])
    vec = eigvec[:, -8:]
    A = vec.dot(np.diag(val)).dot(vec.T)
    return A

geom = np.array([0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]).reshape(-1,3)
#basis2 = np.array([0.5, 0.4, 0.3, 0.2])
#basis = basis2.repeat(12)
basis = np.array([0.5, 0.4, 0.3, 0.2])
#basis = np.array([0.5, 0.4])
full_basis = np.concatenate((basis,basis))
print(full_basis.shape)
#nbf_per_atom = np.array([basis.shape[0],basis.shape[0]])
#nbf_per_atom = np.array([16,16])
#print(nbf_per_atom)
nbf_per_atom = np.array([basis.shape[0],basis.shape[0]])
#nbf_per_atom = int(basis.shape[0])
charge_per_atom = np.array([1.0,1.0])

@jax.jit
def hartree_fock_iter(D, A, H, G, Enuc):
    ndocc = 1
    J = np.einsum('pqrs,rs->pq', G, D)
    K = np.einsum('prqs,rs->pq', G, D)
    F = H + J * 2 - K
    E_scf = np.einsum('pq,pq->', F + H, D) + Enuc
    print(E_scf)
    Fp = A.dot(F).dot(A)
    eps, C2 = np.linalg.eigh(Fp)
    C = np.dot(A, C2)
    Cocc = C[:, :ndocc]
    D = np.einsum('pi,qi->pq', Cocc, Cocc)
    return E_scf, D

def naive(geom):
    S, T, V = oei_setup(geom, full_basis, nbf_per_atom, charge_per_atom)
    H = T + V
    A = orthogonalizer(S)
    Enuc = nuclear_repulsion(geom[0],geom[1])
    D = np.zeros_like(H)
    G = tei_setup(geom,full_basis, nbf_per_atom)

    #fast_hartree_fock_iter = jax.jit(hartree_fock_iter)
    for i in range(12):
        E_scf, D = hartree_fock_iter(D, A, H, G, Enuc)
        #E_scf, D = fast_hartree_fock_iter(D, A, H, G, Enuc)
    return E_scf

def hartree_fock(geom):
    ndocc = 1
    S, T, V = oei_setup(geom, full_basis, nbf_per_atom, charge_per_atom)
    H = T + V
    A = orthogonalizer(S)
    Enuc = nuclear_repulsion(geom[0],geom[1])
    D = np.zeros_like(H)
    G = tei_setup(geom,full_basis, nbf_per_atom)
    D_old = np.ones_like(H)

    def body_func(D, i):
        J = np.einsum('pqrs,rs->pq', G, D)
        K = np.einsum('prqs,rs->pq', G, D)
        F = H + J * 2 - K
        Fp = A.dot(F).dot(A)
        eps, C2 = np.linalg.eigh(Fp)
        C = np.dot(A, C2)
        Cocc = C[:, :ndocc]
        D = np.einsum('pi,qi->pq', Cocc, Cocc)
        return D

    D_final = jax.lax.scan(body_func, D, np.arange(0,12))

    J = np.einsum('pqrs,rs->pq', G, D_final)
    K = np.einsum('prqs,rs->pq', G, D_final)
    F = H + J * 2 - K
    E_scf = np.einsum('pq,pq->', F + H, D_final) + Enuc
    return E_scf


def benchmark(geom):
    G = tei_setup(geom,full_basis, nbf_per_atom)
    fake = np.sum(G)
    return fake
#tei_setup

#G = tei_setup(geom, full_basis, nbf_per_atom)
#gradfunc = jax.jacrev(benchmark)
#hessfunc = jax.jacfwd(gradfunc)
##grad = gradfunc(geom)
#hess = hessfunc(geom)
#print(hess)


#hartree_fock(geom)
#gradfunc = jax.jacrev(hartree_fock)
#print(gradfunc(geom))

gradfunc = jax.jacrev(naive)
#hessfunc = jax.jacfwd(gradfunc)
print(gradfunc(geom))

#METHOD 2
#one_hots = (np.array([[0.0,0.0,0.0],[0.0,0.0,1.0]]), np.array([[0.0,0.0,1.0],[0.0,0.0,0.0]]))
#one_hots = np.array([[[0.0,0.0,0.0],[0.0,0.0,1.0]],[[0.0,0.0,1.0],[0.0,0.0,0.0]]])
#one_hots = np.array([[[1.0,0.0,0.0],[0.0,0.0,0.0]],[[0.0,1.0,0.0],[0.0,0.0,0.0]],[[0.0,0.0,1.0],[0.0,0.0,0.0]],[[0.0,0.0,0.0],[1.0,0.0,0.0]],[[0.0,0.0,0.0],[0.0,1.0,0.0]],[[0.0,0.0,0.0],[0.0,0.0,1.0]]])
#pushfwd = partial(jax.jvp, hartree_fock, (geom,))
#y, out_tangents = jax.vmap(pushfwd, in_axes=(0,), out_axes=(None,1))((one_hots,))
#print(out_tangents)

#METHOD 3
#one_hots = np.array([[[1.0,0.0,0.0],[0.0,0.0,0.0]],[[0.0,1.0,0.0],[0.0,0.0,0.0]],[[0.0,0.0,1.0],[0.0,0.0,0.0]],[[0.0,0.0,0.0],[1.0,0.0,0.0]],[[0.0,0.0,0.0],[0.0,1.0,0.0]],[[0.0,0.0,0.0],[0.0,0.0,1.0]]])
#pushfwd = partial(jax.jvp, hartree_fock, (geom,))
#func = jax.vmap(pushfwd, in_axes=(0,), out_axes=(None,1))
#y, out_tangents = func((one_hots,))
#print(out_tangents)

def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]
    #return jax.jvp(jax.jvp(f, primals, tangents), primals, tangents)[1]
    #return jax.jvp(jax.jacfwd(f), primals, tangents)[1]
    #return jax.jvp(jax.jit(jax.grad(f)), primals, tangents)[1]

#print(hvp(naive, geom))
#test = hvp(hartree_fock, (geom,), (np.array([[0.0,0.0,0.0],[0.0,0.0,1.0]]),))
#print(test)
