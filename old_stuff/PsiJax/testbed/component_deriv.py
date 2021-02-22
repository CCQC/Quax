import jax
import jax.numpy as np
import itertools as it
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=200)

def normalize(aa):
    '''Normalization constant for s primitive basis functions. Argument is orbital exponent coefficient'''
    aa = ((2*aa)/np.pi)**(3/4)
    return aa

def boys_eval(arg):
    return jax.scipy.special.erf(np.sqrt(arg + 1e-9)) * np.sqrt(np.pi) / (2 * np.sqrt(arg + 1e-9))

def eri(aa,bb,cc,dd,A,B,C,D):
    '''Computes a single two electron integral over 4 s-orbital basis functions on 4 centers'''
    g1 = aa + bb
    g2 = cc + dd
    Rp = (aa * A + bb * B) / (aa + bb)
    tmpc1 = np.dot(A-B, A-B) * ((-aa * bb) / (aa + bb))
    c1 = np.exp(tmpc1)
    Rq = (cc * C + dd * D) / (cc + dd)
    tmpc2 = np.dot(C-D, C-D) * ((-cc * dd) / (cc + dd))
    c2 = np.exp(tmpc2)

    Na, Nb, Nc, Nd = normalize(aa), normalize(bb), normalize(cc), normalize(dd)
    delta = 1 / (4 * g1) + 1 / (4 * g2)
    arg = np.dot(Rp - Rq, Rp - Rq) / (4 * delta)
    F = boys_eval(arg)
    G = F * Na * Nb * Nc * Nd * c1 * c2 * 2 * np.pi**2 / (g1 * g2) * np.sqrt(np.pi / (g1 + g2))
    return G

def find_indices(nbf):
    '''Find a set of indices of ERI tensor corresponding to unique two-electron integrals'''
    v = onp.arange(nbf)
    indices = cartesian_product(v,v,v,v)
    cond1 = indices[:,0] >= indices[:,1]
    cond2 = indices[:,2] >= indices[:,3]
    cond3 = indices[:,0] * (indices[:,0] + 1)/2 + indices[:,1] >= indices[:,2]*(indices[:,2]+1)/2 + indices[:,3]
    mask = cond1 & cond2 & cond3

#    cond4 = indices[:,2] >= indices[:,3]
#    cond5 = indices[:,0] >= indices[:,1]
#    cond6 = indices[:,2] * (indices[:,2] + 1)/2 + indices[:,3] >= indices[:,0]*(indices[:,0]+1)/2 + indices[:,1]
#    mask2 = cond4 & cond5 & cond6
#    print(mask2 & mask)

    return np.asarray(indices[mask,:])

def cartesian_product(*arrays):
    '''Find all indices of ERI tensor given 4 arrays 
       (np.arange(nbf), np.arange(nbf), np.arange(nbf), np.arange(nbf)) '''
    la = len(arrays)
    dtype = onp.result_type(*arrays)
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    #arr = onp.empty([len(a) for a in arrays] + [la])
    for i, a in enumerate(onp.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def test_tei(x1,y1,z1,x2,y2,z2,basis):
    geom = np.hstack((x1,y1,z1,x2,y2,z2)).reshape(-1,3)
    nbf = basis.shape[0]
    nbf_per_atom = int(nbf / 2)
    centers = np.repeat(geom, nbf_per_atom, axis=0) # TODO currently can only repeat each center the same number of times => only works for when all atoms have same # of basis functions
    indices = find_indices(nbf)

    # Compute unique ERIs
    #@jax.jit
    def compute_eri(idx):
        i,j,k,l = idx
        tei = eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l])
        return tei
    #vectorized_eri = jax.jit(jax.vmap(compute_eri, (0,)))
    #unique_teis = vectorized_eri(indices)
    unique_teis = jax.lax.map(compute_eri, indices)


    # best so far, though still a lot of redundancy, index 0,0,0,0 for instances gets assigned a value 8 times.
    ##@jax.jit
    #def fill_I():
    #    I = np.empty((nbf,nbf,nbf,nbf))
    #    I = jax.ops.index_update(I, ((indices[:,0],indices[:,2],indices[:,1],indices[:,3],indices[:,1],indices[:,3],indices[:,0],indices[:,2]), 
    #                                 (indices[:,1],indices[:,3],indices[:,0],indices[:,2],indices[:,0],indices[:,2],indices[:,1],indices[:,3]),
    #                                 (indices[:,2],indices[:,0],indices[:,3],indices[:,1],indices[:,2],indices[:,0],indices[:,3],indices[:,1]),
    #                                 (indices[:,3],indices[:,1],indices[:,2],indices[:,0],indices[:,3],indices[:,1],indices[:,2],indices[:,0])),np.broadcast_to(unique_teis, (8,unique_teis.shape[0])))
    #    return I
    #I = fill_I()
    #return I
    I = expand_tei(unique_teis, indices, nbf)
    return I 
    #return unique_teis,indices
    #return np.repeat(unique_teis, 8)


    #@jax.jit
    #def dum():
    #    I = np.empty((nbf,nbf,nbf,nbf))
    #    I = jax.ops.index_update(I, (indices[:,0],indices[:,1],indices[:,2],indices[:,3]), unique_teis)
    #    return I 
    #I = dum()
    #I = fill_I()
    #I = np.broadcast_to(unique_teis, (8,unique_teis.shape[0]))
    #return unique_teis

def fast_tei(geom,basis):
    nbf = basis.shape[0]
    nbf_per_atom = int(nbf / 2)
    centers = np.repeat(geom, nbf_per_atom, axis=0) # TODO currently can only repeat each center the same number of times => only works for when all atoms have same # of basis functions
    indices = find_indices(nbf)

    # Compute unique ERIs
    def compute_eri(idx):
        i,j,k,l = idx
        tei = eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l])
        return tei
    vectorized_eri = jax.jit(jax.vmap(compute_eri, (0,)))
    unique_teis = vectorized_eri(indices)

    # best so far, though still a lot of redundancy, index 0,0,0,0 for instances gets assigned a value 8 times.
    #@jax.jit
    def fill_I():
        I = np.empty((nbf,nbf,nbf,nbf))
        I = jax.ops.index_update(I, ((indices[:,0],indices[:,2],indices[:,1],indices[:,3],indices[:,1],indices[:,3],indices[:,0],indices[:,2]), 
                                     (indices[:,1],indices[:,3],indices[:,0],indices[:,2],indices[:,0],indices[:,2],indices[:,1],indices[:,3]),
                                     (indices[:,2],indices[:,0],indices[:,3],indices[:,1],indices[:,2],indices[:,0],indices[:,3],indices[:,1]),
                                     (indices[:,3],indices[:,1],indices[:,2],indices[:,0],indices[:,3],indices[:,1],indices[:,2],indices[:,0])),np.broadcast_to(unique_teis, (8,unique_teis.shape[0])))
        return I
    I = fill_I()
    return I

def oei(geom,basis,nbf_per_atom,charge_per_atom):
    # SETUP AND OVERLAP INTEGRALS
    nbf = basis.shape[0]
    centers = np.repeat(geom, nbf_per_atom, axis=0) # TODO currently can only repeat each center the same number of times => only works for when all atoms have same # of basis functions
    #centers = np.repeat(geom, [4,4], axis=0) # TODO currently can only repeat each center the same number of times => only works for when all atoms have same # of basis functions
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

    ### STABLE FOR SMALL EIGENVALUES
    eigval, eigvec = np.linalg.eigh(S)
    #cutoff = 1.0e-12
    #above_cutoff = (abs(eigval) > cutoff * np.max(abs(eigval)))
    #TODO TODO hard coded
    val = 1 / np.sqrt(eigval[-8:])
    vec = eigvec[:, -8:]
    A = vec.dot(np.diag(val)).dot(vec.T)
    return A

geom = np.array([0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]).reshape(-1,3)
#geom = np.array([0.000000000000,0.000000000000,-0.8492204,0.000000000000,0.000000000000,0.8492204]).reshape(-1,3)

atom1_basis = np.repeat(np.array([0.5, 0.4, 0.3, 0.2]),8)
atom2_basis = np.repeat(np.array([0.5, 0.4, 0.3, 0.2]),8)
#atom1_basis = np.array([0.5, 0.4, 0.3, 0.2])
#atom2_basis = np.array([0.5, 0.4, 0.3, 0.2])
#atom1_basis = np.array([0.5, 0.4])
#atom2_basis = np.array([0.5, 0.4])
#atom1_basis = np.array([0.5])
#atom2_basis = np.array([0.4])
basis = np.concatenate((atom1_basis, atom2_basis))
print(basis.shape)
#centers = np.concatenate((np.tile(geom[0],atom1_basis.size).reshape(-1,3), np.tile(geom[1],atom2_basis.size).reshape(-1,3)))
nbf_per_atom = np.array([atom1_basis.shape[0],atom2_basis.shape[0]])
charge_per_atom = np.array([1.0,1.0])

#@jax.jit
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

def hartree_fock(x1,y1,z1,x2,y2,z2):
    geom = np.hstack((x1,y1,z1,x2,y2,z2)).reshape(-1,3)
    S,T,V = oei(geom,basis,nbf_per_atom,charge_per_atom)
    G = fast_tei(geom,basis) 
    #G = tei_setup(geom,basis)
    H = T + V
    A = orthogonalizer(S)
    Enuc = nuclear_repulsion(geom[0],geom[1])
    D = np.zeros_like(H)

    #for i in range(12):
    for i in range(1):
        E_scf, D = hartree_fock_iter(D, A, H, G, Enuc)
    return E_scf


def expand_tei(unique_teis, indices, nbf):
    I = np.empty((nbf,nbf,nbf,nbf))
    I = jax.ops.index_update(I, ((indices[:,0],indices[:,2],indices[:,1],indices[:,3],indices[:,1],indices[:,3],indices[:,0],indices[:,2]), 
                                 (indices[:,1],indices[:,3],indices[:,0],indices[:,2],indices[:,0],indices[:,2],indices[:,1],indices[:,3]),
                                 (indices[:,2],indices[:,0],indices[:,3],indices[:,1],indices[:,2],indices[:,0],indices[:,3],indices[:,1]),
                                 (indices[:,3],indices[:,1],indices[:,2],indices[:,0],indices[:,3],indices[:,1],indices[:,2],indices[:,0])),np.broadcast_to(unique_teis, (8,unique_teis.shape[0])))
    return I

#unique_teis, indices = test_tei(0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955, basis)
#G = expand_tei(unique_teis, indices, 64)
G = test_tei(0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955, basis)
hessfunc = jax.jacfwd(jax.jacfwd(test_tei, argnums=2), argnums=2)
cubefunc = jax.jacfwd(jax.jacfwd(jax.jacfwd(test_tei, argnums=2), argnums=2), argnums=2)
quarfunc = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(test_tei, argnums=2), argnums=2), argnums=2), argnums=2)

#hess = hessfunc(0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955, basis)
#q = quarfunc(0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955, basis)
#print(G.shape)
#print(q.shape)

#E = hartree_fock(0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955)
#print(E)

#gradfunc = jax.jacrev(hartree_fock, argnums=(2,))
#gradfunc = jax.jacrev(hartree_fock, argnums=(2,))
#hessfunc = jax.jacfwd(gradfunc, argnums=(2,))
#cubefunc = jax.jacfwd(hessfunc, argnums=(2,))
#quarfunc = jax.jacfwd(cubefunc, argnums=(2,))

#gradfunc = jax.jacrev(hartree_fock, argnums=(0,1,2,3,4,5))
#hessfunc = jax.jacfwd(gradfunc, argnums=(0,1,2,3,4,5))
#cubefunc = jax.jacfwd(hessfunc, argnums=(0,1,2,3,4,5))
#quarfunc = jax.jacfwd(cubefunc, argnums=(0,1,2,3,4,5))



#gradfunc = jax.jacrev(hartree_fock, argnums=(2,))
#hessfunc = jax.jacfwd(jax.jacfwd(hartree_fock, argnums=(2,)), argnums=(2,))
#cubefunc = jax.jacfwd(jax.jacfwd(jax.jacfwd(hartree_fock, argnums=(2,)), argnums=(2,)), argnums=(2,))
#quarfunc = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(hartree_fock, argnums=2), argnums=2), argnums=2), argnums=2)
#quinfunc = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(hartree_fock, argnums=(2,)), argnums=(2,)), argnums=(2,)), argnums=(2,)), argnums=(2,))

#hessfunc = jax.jacfwd(jax.jacfwd(hartree_fock, argnums=(0,1,2,3,4,5)), argnums=(0,1,2,3,4,5))
#grad = gradfunc(0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955)
#print(grad)
#hess = hessfunc(0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955)
#print(hess)
#cube = cubefunc(0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955)
#print(cube)
#quar = quarfunc(0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955)
#print(quar)
#quin = quinfunc(0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955)
#print(quin)

#gradfunc = jax.jacfwd(hartree_fock)
#hessfunc = jax.jacfwd(gradfunc)
#cubefunc = jax.jacfwd(hessfunc)

#G = fast_tei(geom,basis)
#print(G.shape)
#Ggrad = jax.jacfwd(fast_tei)(geom,basis)
#print(Ggrad.shape)
#jax.jacfwd(jax.jacfwd(jax.jacfwd(fast_tei)))(geom,basis)
#print(Gcube.shape)

