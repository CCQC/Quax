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

def index2(i,j):
    if (i < j):
        return int(j * (j + 1) / 2 + i)
    else:
        return int(i * (i + 1) / 2 + j)

def find_indices(nbf):
    '''Find a set of indices of ERI tensor corresponding to unique two-electron integrals'''
    v = onp.arange(nbf)
    indices = cartesian_product(v,v,v,v)
    # 'Canonical' order, i>=j, k>=l, IJ>=KL
    cond1 = (indices[:,0] >= indices[:,1]) & (indices[:,2] >= indices[:,3])
    cond2 = indices[:,0] * (indices[:,0] + 1)/2 + indices[:,1] >= indices[:,2]*(indices[:,2]+1)/2 + indices[:,3]
    mask = cond1 & cond2 
    return np.asarray(indices[mask,:])

def cartesian_product(*arrays):
    '''Find all indices of ERI tensor given 4 arrays 
       (np.arange(nbf), np.arange(nbf), np.arange(nbf), np.arange(nbf)) '''
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
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



def index2(i,j):
    '''Compute compound index ij given indices i,j'''
    if (i < j):
        return int(j * (j + 1) / 2 + i)
    else:
        return int(i * (i + 1) / 2 + j)

def index4(i,j,k,l):
    '''Compute compound index ijkl given indices i,j,k,l'''
    return index2(index2(i,j),index2(k,l))

def vectorized_index2(indicesI, indicesJ):
    '''Compute compound indices IJ given a vector of many i's, a vector of many j's'''
    indicesI = onp.asarray(indicesI)
    indicesJ = onp.asarray(indicesJ)
    cond = indicesI < indicesJ
    compoundij = onp.zeros((indicesI.shape[0]))
    if np.any(cond): 
        compoundij[cond] = indicesJ[cond] * (indicesJ[cond] + 1) / 2 + indicesI[cond]
    if np.any(~cond): 
        compoundij[~cond] = indicesI[~cond] * (indicesI[~cond] + 1) / 2 + indicesJ[~cond]
    return compoundij

def vectorized_index4(indicesI, indicesJ, indicesK, indicesL):
    '''Compute compound indices IJKL given a vector of i's, j's, k's, l's'''
    compound_IJKL = vectorized_index2(vectorized_index2(indicesI, indicesJ),vectorized_index2(indicesK, indicesL)).astype(int)
    # convert to JAX array
    return np.asarray(compound_IJKL)

def build_c_ikjl(indices):
    '''Compute the coefficients of PK elements at all indices IKJL according to Raffenetti's conditions.'''
    indices = onp.asarray(indices)
    i,j,k,l = indices[:,0], indices[:,1], indices[:,2], indices[:,3]
    cond = (i == k) | (j == l)
    c_ikjl = onp.where(cond, -0.5, -0.25)
    # convert to JAX array
    return np.asarray(c_ikjl)
    
def build_c_iljk(indices):
    '''Compute the coefficients of PK elements at all indices ILJK according to Raffenetti's conditions.'''
    indices = onp.asarray(indices)
    i,j,k,l = indices[:,0], indices[:,1], indices[:,2], indices[:,3]
    cond1 = (i == j) | (k == l) # yield 0.0
    cond2 = (i == l) | (j == k) # yield 0.5 else 0.25
    c_iljk = onp.where(cond1, 0.0, onp.where(cond2, -0.5, -0.25))
    # convert to JAX array
    return np.asarray(c_iljk)

def build_ILJK(indices):
    '''For a given integral index in 'indices', find if it contributes to index ILJK in PK. 
       If it does, determine the index. Else, set the index to -1.'''
    indices = onp.asarray(indices)
    i,j,k,l = indices[:,0], indices[:,1], indices[:,2], indices[:,3]
    cond = (i != j) & (k != l)  # if this, compute index, else set -1
    ILJK = onp.where(cond, vectorized_index4(i,l,j,k), -1).astype(int)
    # convert to JAX array
    return np.asarray(ILJK)

def pk_diagonal(nbf):
    '''Returns the index vector corresponding to locations 'diagonal' entries IJ==KL in the PK vector
       Used to multiply all diagonal elements of PK by 0.5, as instructed in Raffenetti, 1973'''
    dim = int(nbf * (nbf + 1) / 2)
    lower_triangle_vector = onp.arange((dim**2 - dim) / 2 + dim) #includes diagonal
    mask = onp.tri(dim, dtype=bool, k=0)
    out = onp.zeros((dim,dim),dtype=int)
    out[mask] = lower_triangle_vector
    pk_diag = onp.diagonal(out)
    return pk_diag

def build_PK(g, indices, nbf):
    '''Builds the RHF PK 'supermatrix' using JAX/XLA primitives (its really a big vector of length equal to the number of unique two electron integrals) '''
    IJKL = np.arange(g.shape[0])
    IKJL = vectorized_index4(indices[:,0], indices[:,2], indices[:,1], indices[:,3])
    ILJK = build_ILJK(indices)
    C_IKJL = build_c_ikjl(indices)
    C_ILJK = build_c_iljk(indices)

    #Deal with 'diagonal' IJ=IJ of pk being multiplied by 0.5
    pk_diag_indices = pk_diagonal(nbf)
    pk_diag = onp.ones(g.shape[0] + 1)
    pk_diag[pk_diag_indices] = 0.5
    pk_diag = np.asarray(pk_diag)

    # Build PK with a buffer dummy element at the end. This is so ILJK indices which are set to -1 don't actually effect PK values
    # Note this causes some redundant computation, but leads to straight-forward 'jaxification'.
    pk = np.zeros(g.shape[0] + 1)

    # jax.lax.scan super slow if multiple index updates in one function. (race conditions?) Just do it three times I guess?
    # The C_x arrays have the negative sign built in, contain either -0.25 or -0.5.
    def update_ijkl(pk,i):
        pk = jax.ops.index_add(pk, IJKL[i], g[i])
        return pk, ()

    def update_ikjl(pk, i):
        pk = jax.ops.index_add(pk, IKJL[i], C_IKJL[i] * g[i])
        return pk, ()

    def update_iljk(pk, i):
        pk = jax.ops.index_add(pk, ILJK[i], C_ILJK[i] * g[i])
        return pk, ()

    # Scan over function update_x, carry pk object through, update indices [0,1,2,3,4...size(g)] simultaneously
    pk, _ = jax.lax.scan(update_ijkl, pk, IJKL)
    pk, _ = jax.lax.scan(update_ikjl, pk, IJKL)
    pk, _ = jax.lax.scan(update_iljk, pk, IJKL)

    final_pk = pk * pk_diag
    return final_pk

def build_G(indices, tmpD, pk, nbf):
    '''Builds the 'G matrix' using the density and the PK matrix. The G matrix is related to the RHF Fock matrix by F = H + G, where H is the one electron part.'''
    # 'factor' is a matrix of 1's along the diagonal, 2's elsewhere. 
    # Elementwise multiplication is equivalent to doubling off diagonal elements, as instructed in Raffenetti, 1973
    factor = np.eye(nbf)
    off_diag = onp.where(~np.eye(nbf, dtype=bool))
    factor = jax.ops.index_add(factor, off_diag, 2)
    tmpD = tmpD * factor
    D = tmpD[np.tril_indices(nbf)]

    # These index arrays are Numpy by default, switch to jax arrays
    IJ   = np.asarray(vectorized_index2(indices[:,0], indices[:,1]).astype(int))
    KL   = np.asarray(vectorized_index2(indices[:,2], indices[:,3]).astype(int))
    IJKL = np.asarray(vectorized_index2(IJ,KL).astype(int))

    # Slow, naive implementation:
    #for i in range(IJ.shape[0]):
    #    G = jax.ops.index_add(G, IJ[i], pk[IJKL[i]] * D[KL[i]])
    #    G = jax.ops.index_add(G, KL[i], pk[IJKL[i]] * D[IJ[i]])

    # Build G using jax.lax.scan to 'unroll' the loop and simulatenously compute elements
    G = np.zeros_like(D)

    def update_IJ(G, i):
        G = jax.ops.index_add(G, IJ[i], pk[IJKL[i]] * D[KL[i]])
        return G, ()

    def update_KL(G, i):
        G = jax.ops.index_add(G, KL[i], pk[IJKL[i]] * D[IJ[i]])
        return G, ()

    idx = np.arange(IJ.shape[0])
    G, _ = jax.lax.scan(update_IJ, G, idx)
    G, _ = jax.lax.scan(update_KL, G, idx)

    ## Convert to symmeteric matrix... how to do this in JAX?
    fullG = np.zeros((nbf,nbf))
    xs, ys = np.tril_indices(nbf)
    fullG = jax.ops.index_update(fullG, (xs,ys), G)
    fullG = jax.ops.index_update(fullG, (ys,xs), G)

    #fullG = np.ones((nbf,nbf))
    return 2 * fullG
    
def hartree_fock(x1,y1,z1,x2,y2,z2):
    geom = np.array([x1,y1,z1,x2,y2,z2]).reshape(-1,3)
    nbf = basis.shape[0]  
    S,T,V = oei(geom,basis,nbf_per_atom,charge_per_atom)

    g, indices = tei(geom,basis) 
    pk = build_PK(g, indices, nbf)

    H = T + V
    A = orthogonalizer(S)
    Enuc = nuclear_repulsion(geom[0],geom[1])
    D = np.zeros((nbf,nbf))
    ndocc = 1
    
    for i in range(20):
        G = build_G(indices, D, pk, nbf) 
#        G = np.ones((nbf,nbf))
        F = G + H
        E_scf = np.einsum('pq,pq->', F + H, D) + Enuc
        print(E_scf)
        Fp = A.dot(F).dot(A)
        eps, C2 = np.linalg.eigh(Fp)
        C = np.dot(A, C2)
        Cocc = C[:, :ndocc]
        D = np.einsum('pi,qi->pq', Cocc, Cocc)

    return E_scf


geom = np.array([0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]).reshape(-1,3)
atom1_basis = np.repeat(np.array([0.5, 0.4, 0.3, 0.2]),16)
atom2_basis = np.repeat(np.array([0.5, 0.4, 0.3, 0.2]),16)
#atom1_basis = np.array([0.5, 0.4])
#atom2_basis = np.array([0.5, 0.4])
#atom1_basis = np.array([0.5])
#atom2_basis = np.array([0.4])
basis = np.concatenate((atom1_basis, atom2_basis))
print(basis.shape)
nbf_per_atom = np.array([atom1_basis.shape[0],atom2_basis.shape[0]])
charge_per_atom = np.array([1.0,1.0])


E = hartree_fock(0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955)
#e = hartree_fock(geom)
hessfunc = jax.jacfwd(jax.jacfwd(hartree_fock, argnums=2), argnums=2)
cubefunc = jax.jacfwd(jax.jacfwd(jax.jacfwd(hartree_fock, argnums=2), argnums=2), argnums=2)
quarfunc = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(hartree_fock, argnums=2), argnums=2), argnums=2), argnums=2)


#h = quarfunc(0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955)

#h = hessfunc(

#g = jax.jacfwd(hartree_fock)(geom)
#g = jax.jacfwd(jax.jacfwd(hartree_fock))(geom)
#g = jax.jacfwd(jax.jacfwd(jax.jacfwd(hartree_fock)))(geom)
#print(g)







