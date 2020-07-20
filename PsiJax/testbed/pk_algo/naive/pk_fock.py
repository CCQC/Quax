import jax
import jax.numpy as np
import itertools as it
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
onp.set_printoptions(precision=8,linewidth=200)
np.set_printoptions(precision=8,linewidth=200)

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
#    cond3 = indices[:,0] * (indices[:,0] + 1)/2 + indices[:,1] >= indices[:,2]*(indices[:,2]+1)/2 + indices[:,3]
    condij = indices[:,0] < indices[:,1]
    condkl = indices[:,2] < indices[:,3]
    # all indices where i<j, compute j(j+1) / 2 + i. all indices where i>=j, compute i(i+1) / 2 + j
    tmpIJ = onp.empty(indices.shape[0])
    tmpIJ[condij] = indices[condij, 1] * (indices[condij,1] + 1) / 2 + indices[condij,0]
    tmpIJ[~condij] = indices[~condij, 0] * (indices[~condij,0] + 1) / 2 + indices[~condij,1]
    # all indices where k<l, compute l(l+1) / 2 + k. all indices where k>=l, compute k(k+1) / 2 + l
    tmpKL = onp.empty(indices.shape[0])
    tmpKL[condkl] = indices[condkl, 3] * (indices[condkl,3] + 1) / 2 + indices[condkl,2]
    tmpKL[~condkl] = indices[~condkl, 2] * (indices[~condkl,2] + 1) / 2 + indices[~condkl,3]
    # Bool for IJ>=KL
    cond3 = tmpIJ >= tmpKL

    mask = cond1 & cond2 & cond3
    return np.asarray(indices[mask,:])

def cartesian_product(*arrays):
    '''Find all indices of ERI tensor given 4 arrays 
       (np.arange(nbf), np.arange(nbf), np.arange(nbf), np.arange(nbf)) '''
    la = len(arrays)
    dtype = onp.result_type(*arrays)
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def tei(geom,basis):
    nbf = basis.shape[0]
    nbf_per_atom = int(nbf / 2)
    centers = np.repeat(geom, nbf_per_atom, axis=0) # TODO currently can only repeat each center the same number of times => only works for when all atoms have same # of basis functions
    indices = find_indices(nbf)

    # Compute unique ERIs
    def compute_eri(idx):
        i,j,k,l = idx
        tei = eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l])
        return tei
    unique_teis = jax.lax.map(compute_eri, indices)
    return unique_teis, indices

def oei(geom,basis,nbf_per_atom,charge_per_atom):
    nbf = basis.shape[0]
    centers = np.repeat(geom, nbf_per_atom, axis=0) # TODO currently can only repeat each center the same number of times => only works for when all atoms have same # of basis functions
    norm = (2 * basis / np.pi)**(3/4)
    normtensor = np.outer(norm,norm) # outer product => every possible combination of Na * Nb
    aa_times_bb = np.outer(basis,basis)
    aa_plus_bb = np.broadcast_to(basis, (nbf,nbf)) + np.transpose(np.broadcast_to(basis, (nbf,nbf)), (1,0))
    term = (np.pi / aa_plus_bb) ** (3/2)
    tmpA = np.broadcast_to(centers, (nbf,nbf,3))
    AminusB = tmpA - np.transpose(tmpA, (1,0,2)) #caution: tranpose shares memory with original array. changing one changes the other
    AmBAmB = np.einsum('ijk,ijk->ij', AminusB, AminusB)
    coeff = np.exp(AmBAmB * (-aa_times_bb / aa_plus_bb))
    S = normtensor * coeff * term
    P = aa_times_bb / aa_plus_bb
    T = S * (3 * P + 2 * P * P * -AmBAmB)
    aatimesA = np.einsum('i,ij->ij', basis,centers)
    numerator = aatimesA[:,None,:] + aatimesA[None,:,:]
    R = np.einsum('ijk,ij->ijk', numerator, 1/aa_plus_bb)
    R_per_atom = np.broadcast_to(R, (geom.shape[0],) + R.shape)
    expanded_geom = np.transpose(np.broadcast_to(geom, (nbf,nbf) + geom.shape), (2,1,0,3))
    Rminusgeom = R_per_atom - expanded_geom
    contracted = np.einsum('ijkl,ijkl->ijk', Rminusgeom,Rminusgeom)
    boys_arg = np.einsum('ijk,jk->ijk', contracted, aa_plus_bb)
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
    eigval, eigvec = np.linalg.eigh(S)
    cutoff = 1.0e-12
    above_cutoff = (abs(eigval) > cutoff * np.max(abs(eigval)))
    val = 1 / np.sqrt(eigval[above_cutoff])
    vec = eigvec[:, above_cutoff]
    A = vec.dot(np.diag(val)).dot(vec.T)
    return A

geom = np.array([0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]).reshape(-1,3)

#atom1_basis = np.repeat(np.array([0.5, 0.4, 0.3, 0.2]),1)
#atom2_basis = np.repeat(np.array([0.5, 0.4, 0.3, 0.2]),1)
#atom1_basis = np.array([0.5, 0.4])
#atom2_basis = np.array([0.5, 0.4])
atom1_basis = np.array([0.5])
atom2_basis = np.array([0.5])
basis = np.concatenate((atom1_basis, atom2_basis))
print(basis.shape)
nbf_per_atom = np.array([atom1_basis.shape[0],atom2_basis.shape[0]])
charge_per_atom = np.array([1.0,1.0])

# Cant use raffennetti, is one-based indexing for FORTRAN 
def index2(i,j):
    if (i < j):
        return int(j * (j + 1) / 2 + i)
    else:
        return int(i * (i + 1) / 2 + j)

def index4(i,j,k,l):
    return index2(index2(i,j),index2(k,l))

def build_P(geom, nbf):
    ''' g: unique tei integrals vector. 
        indices: A column vector of 4-index indices, unique set of TEI indices corresponding to unique teis'''
    dim = int((nbf**2 - nbf) / 2 + nbf)
    J = onp.zeros((dim,dim))
    K = onp.zeros((dim,dim))
    nbf_per_atom = int(nbf / 2)
    centers = np.repeat(geom, nbf_per_atom, axis=0) 
    # "Yoshimine describes a procedure for bringing a list of randomly ordered integrals ino a specific order, namely 'canonical' order where i >= j, k>=l, IJ>=KL."
    indices = find_indices(nbf)

    for idx in indices:
        i,j,k,l = idx
        IJ = index2(i,j)
        KL = index2(k,l)
        j_int  = eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l])
        k_int1 = eri(basis[i], basis[k], basis[j], basis[l], centers[i], centers[k], centers[j], centers[l])
        k_int2 = eri(basis[i], basis[l], basis[j], basis[k], centers[i], centers[l], centers[j], centers[k])
        # "In order to obtain proper quantities for J, K/2, a factor must be multiplied into each integral at the first stage of processing.
        #  When the original integral (ij|kl) is being sorted for the J array, there is no factor.
        #  When sorted for the K/2 array, location IKJL, the factor is given by 0.5 if i=k or j=l, 0.25 otherwise."
        if (i == j or k == l):
            if i == k or j == l:
                #k_int1 *= 0.5
                #alpha1 = 0.5
                alpha = 0.5
            else: 
                #k_int1 *= 0.25
                #alpha1 = 0.25
                alpha = 0.25

        # "If i=j or k=l the integral is not sorted again... if it is sorted, the factor alpha is 0.5 if i=l or j=k, 0.25 otherwise."
        alpha2 = 1.0
        if (i != j or k != l):
            if i == l or j == k:
                #k_int2 *= 0.5
                #alpha2 = 0.5
                alpha = 0.5
            else:
                #k_int2 *= 0.5
                #alpha2 = 0.25
                alpha = 0.25

        J[IJ,KL] = j_int
        K[IJ,KL] = (k_int1 + k_int2) / 4
        #K[IJ,KL] = (alpha1 * k_int1 + alpha2 * k_int2) / 4
        #K[IJ,KL] = alpha * (k_int1 + k_int2) / 4

    b = onp.eye(dim, dtype=bool)
    # "The elements of P and K/2 must be set to one-half their value if IJ==KL." Change K/2 before or after forming P? Huh?
    #K[b] *= 0.5
    P = J - K
    P[b] *= 0.5
    print(P)
    return P, indices

def build_G(indices, D, P, nbf):
    D = D[np.tril_indices(nbf)].flatten()
    # Build G
    dim = int((nbf**2 - nbf) / 2 + nbf)
    G = onp.zeros((dim))
    for idx in indices:
        i,j,k,l = idx
        IJ = index2(i,j)
        KL = index2(k,l)
        G[IJ] = G[IJ] + P[IJ,KL] * D[KL]
        G[KL] = G[KL] + P[IJ,KL] * D[IJ]
    newG = onp.zeros((nbf,nbf))
    newG[np.tril_indices(nbf)] = G
    return newG

def hartree_fock(x1,y1,z1,x2,y2,z2):
    geom = onp.hstack((x1,y1,z1,x2,y2,z2)).reshape(-1,3)
    nbf = basis.shape[0]  
    S,T,V = oei(geom,basis,nbf_per_atom,charge_per_atom)
    P,indices = build_P(geom, nbf)
    H = T + V
    A = orthogonalizer(S)
    Enuc = nuclear_repulsion(geom[0],geom[1])
    D = onp.zeros((nbf,nbf))
    ndocc = 1
    
    for i in range(5):
        b = onp.eye(nbf, dtype=bool) # Boolean mask of diagonal
        G = build_G(indices, D, P, nbf)
        G[np.triu_indices(nbf)] = G[np.tril_indices(nbf)]
        F = H + G
        print(F)
        # Reverse effect of doubling density.. "The density matrix D must be stored with all off-diagonal elements doubled."
        D[~b] *= 0.5
        E_scf = onp.einsum('pq,pq->', F + H, D) + Enuc
        print(E_scf)
        Fp = A.dot(F).dot(A)
        eps, C2 = onp.linalg.eigh(Fp)
        C = onp.dot(A, C2)
        Cocc = C[:, :ndocc]
        D = onp.einsum('pi,qi->pq', Cocc, Cocc)
        # "The density matrix D must be stored with all off-diagonal elements doubled."
        D[~b] *= 2

E = hartree_fock(0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955)


