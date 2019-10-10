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
    #vectorized_eri = jax.jit(jax.vmap(compute_eri, (0,)))
    #unique_teis = vectorized_eri(indices)
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
atom1_basis = np.array([0.5, 0.4])
atom2_basis = np.array([0.5, 0.4])
basis = np.concatenate((atom1_basis, atom2_basis))
print(basis.shape)
nbf_per_atom = np.array([atom1_basis.shape[0],atom2_basis.shape[0]])
charge_per_atom = np.array([1.0,1.0])


def index(a,b):
    return int(a*(a+1)/2 + b if (a > b) else b*(b+1)/2 + a)

def build_pj_pk(g, indices, nbf):
    ''' g: unique tei integrals vector. 
        indices: A column vector of 4-index indices, unique set of TEI indices corresponding to unique teis'''

    pj = onp.zeros((g.shape[0]))
    pk = onp.zeros((g.shape[0]))

    for idx,val in enumerate(g):
        i,j,k,l = indices[idx][0],indices[idx][1],indices[idx][2],indices[idx][3]
        bra = index(i,j)
        ket = index(k,l)
        braket = index(bra,ket)
        pj[braket] += val

        if i != j and k != l:
            bra = index(i,l)
            ket = index(j,k)
            braket = index(bra,ket)
        if i == l or j == k:
            pk[braket] += val
        else:
            pk[braket] += 0.5 * val

        bra = index(i,k)
        ket = index(j,l)
        braket = index(bra,ket)
        if i == k or j == l:
            pk[braket] += val
        else:
            pk[braket] += 0.5 * val

    for ij in range(int(nbf * (nbf + 1) / 2)):
        r = index(ij,ij)
        pj[r] *= 0.5
        pk[r] *= 0.5

    return pj, pk

def build_fock(D, H, p_j, p_k):
    nbf = D.shape[0]
    D = onp.asarray(D)
    # Multiply off diagonal of density by 2
    b = onp.eye(nbf, dtype=bool)
    D[~b] *= 2
    
    D = D.flatten()
    J = onp.zeros_like(D)
    K = onp.zeros_like(D)

    # D_rs is an address in Density vector
    D_rs = 0
    # J_rs is an address in the Coulomb vector J 
    J_rs = 0
    K_rs = 0
    for pq in range(nbf):
        # D_pq is a value   
        D_pq = D[pq]
        # J_pq is a value
        J_pq = 0.0
        K_pq = 0.0
        for rs in range(0, pq):
            J_pq += p_j[J_rs] * D[D_rs]
            J[J_rs] += p_j[J_rs] * D_pq

            K_pq += p_k[K_rs] * D[D_rs]
            K[K_rs] += p_k[K_rs] * D_pq

            D_rs += 1
            J_rs += 1
            K_rs += 1
        J[pq] += J_pq
        K[pq] += K_pq

    print(J)
    J = J.reshape(nbf,nbf)
    K = K.reshape(nbf,nbf)
    F = H + 2 * J + K
    return F 

def hartree_fock(x1,y1,z1,x2,y2,z2):
    geom = onp.hstack((x1,y1,z1,x2,y2,z2)).reshape(-1,3)
    nbf = basis.shape[0]  
    S,T,V = oei(geom,basis,nbf_per_atom,charge_per_atom)
    G, indices = tei(geom,basis) 
    pj, pk = build_pj_pk(G, indices, nbf)  
    H = T + V
    A = orthogonalizer(S)
    Enuc = nuclear_repulsion(geom[0],geom[1])
    D = onp.zeros((nbf,nbf))
    print(type(D))
    ndocc = 1
    
    for i in range(5):
        F = build_fock(D, H, pj, pk)
        E_scf = onp.einsum('pq,pq->', F + H, D) + Enuc
        print(E_scf)
        Fp = A.dot(F).dot(A)
        eps, C2 = onp.linalg.eigh(Fp)
        C = onp.dot(A, C2)
        Cocc = C[:, :ndocc]
        D = onp.einsum('pi,qi->pq', Cocc, Cocc)


E = hartree_fock(0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955)
#gradfunc = jax.jacrev(hartree_fock, argnums=(2,))
#gradfunc = jax.jacrev(hartree_fock, argnums=(2,))
#hessfunc = jax.jacfwd(gradfunc, argnums=(2,))
#cubefunc = jax.jacfwd(hessfunc, argnums=(2,))
#quarfunc = jax.jacfwd(cubefunc, argnums=(2,))

