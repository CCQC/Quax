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
    cond1 = indices[:,0] >= indices[:,1]
    cond2 = indices[:,2] >= indices[:,3]
    # is this right?
    #TMP TODO
    #old_cond3 = indices[:,0] * (indices[:,0] + 1)/2 + indices[:,1] >= indices[:,2]*(indices[:,2]+1)/2 + indices[:,3]
    cond3 = indices[:,0] * (indices[:,0] + 1)/2 + indices[:,1] >= indices[:,2]*(indices[:,2]+1)/2 + indices[:,3]

#    condij = indices[:,0] < indices[:,1]
#    condkl = indices[:,2] < indices[:,3]
#    # all indices where i<j, compute j(j+1) / 2 + i. all indices where i>=j, compute i(i+1) / 2 + j
#    tmpIJ = onp.empty(indices.shape[0])
#    tmpIJ[condij] = indices[condij, 1] * (indices[condij,1] + 1) / 2 + indices[condij,0]
#    tmpIJ[~condij] = indices[~condij, 0] * (indices[~condij,0] + 1) / 2 + indices[~condij,1]
#    # all indices where k<l, compute l(l+1) / 2 + k. all indices where k>=l, compute k(k+1) / 2 + l
#    tmpKL = onp.empty(indices.shape[0])
#    tmpKL[condkl] = indices[condkl, 3] * (indices[condkl,3] + 1) / 2 + indices[condkl,2]
#    tmpKL[~condkl] = indices[~condkl, 2] * (indices[~condkl,2] + 1) / 2 + indices[~condkl,3]
#    # Bool for IJ>=KL
#    cond3 = tmpIJ >= tmpKL
    #print(cond3)
    #print(np.allclose(old_cond3, cond3))
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
atom1_basis = np.array([0.5, 0.4])
atom2_basis = np.array([0.5, 0.4])
#atom1_basis = np.array([0.5])
#atom2_basis = np.array([0.4])
basis = np.concatenate((atom1_basis, atom2_basis))
print(basis.shape)
nbf_per_atom = np.array([atom1_basis.shape[0],atom2_basis.shape[0]])
charge_per_atom = np.array([1.0,1.0])

# Psi4's INDEX2(i,j) function
def index2(i,j):
    if (i < j):
        return int(j * (j + 1) / 2 + i)
    else:
        return int(i * (i + 1) / 2 + j)

def index4(i,j,k,l):
    return index2(index2(i,j),index2(k,l))

def build_pj_pk(g, indices, nbf):
    '''g: unique teis as a vector 
       indices: indices of each unique tei  (i<=j k<=l, IJ<=KL)
    '''
    pj = onp.zeros((g.shape[0]))
    pk = onp.zeros((g.shape[0]))

    for idx,val in enumerate(g):
        i,j,k,l = indices[idx][0],indices[idx][1],indices[idx][2],indices[idx][3]
        bra = index2(i,j)
        ket = index2(k,l)
        braket = index2(bra,ket)
        pj[braket] += val

        if i != j and k != l:
            bra = index2(i,l)
            ket = index2(j,k)
            braket = index2(bra,ket)
            if i == l or j == k:
                pk[braket] += val
            else:
                pk[braket] += 0.5 * val

        bra = index2(i,k)
        ket = index2(j,l)
        braket = index2(bra,ket)
        if i == k or j == l:
            pk[braket] += val
        else:
            pk[braket] += 0.5 * val

    for ij in range(int(nbf * (nbf + 1) / 2)):
        r = index2(ij,ij)
        pj[r] *= 0.5
        pk[r] *= 0.5
    return pj, pk

def build_fock(in_D, H, p_j, p_k):
    nbf = in_D.shape[0]
    tmpD = in_D.copy()
    # Multiply off diagonal of density by 2
    b = onp.eye(nbf, dtype=bool)
    tmpD[~b] *= 2
    tmpD = tmpD[onp.tril_indices(nbf)]
    J = onp.zeros_like(tmpD)
    K = onp.zeros_like(tmpD)

    pJKi = 0
    for pq in range(int(nbf * (nbf + 1) / 2)):
        # D_pq: Density matrix value 
        D_pq = tmpD[pq]
        # D_rs: Position in Density vector
        D_rs = 0
        # J_rs: Position in Coulomb vector
        J_rs = 0
        # K_rs: Position in Exchnge vector
        K_rs = 0

        # J_pq is a value
        J_pq = 0.0
        K_pq = 0.0
        for rs in range(0, pq+1):
            J_pq += p_j[pJKi] * tmpD[D_rs] # might need * 2
            J[J_rs] += p_j[pJKi] * D_pq # might need * 2

            K_pq += p_k[pJKi] * tmpD[D_rs] 
            K[K_rs] += p_k[pJKi] * D_pq

            D_rs += 1
            J_rs += 1
            K_rs += 1
            pJKi += 1
        J[pq] += J_pq
        K[pq] += K_pq

    print("K matrix eleemnts")
    print(K)
    F_tmp = J * 2 - K
    F_noH = onp.zeros((nbf,nbf))
    #xs, ys = onp.triu_indices(nbf,0) 
    xs, ys = onp.tril_indices(nbf,0) 
    F_noH[xs,ys] = F_tmp
    F_noH[ys,xs] = F_tmp

    F = F_noH + H
    return F 

def hartree_fock(x1,y1,z1,x2,y2,z2):
    geom = onp.hstack((x1,y1,z1,x2,y2,z2)).reshape(-1,3)
    nbf = basis.shape[0]  
    S,T,V = oei(geom,basis,nbf_per_atom,charge_per_atom)
    S = onp.asarray(S)
    T = onp.asarray(T)
    V = onp.asarray(V)

    g, indices = tei(geom,basis) 
    pj,pk = build_pj_pk(g, indices, nbf)

    H = T + V
    A = orthogonalizer(S)
    Enuc = nuclear_repulsion(geom[0],geom[1])
    D = onp.zeros((nbf,nbf))
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



