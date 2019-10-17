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
    #v = onp.arange(nbf, dtype=onp.int16)
    v = onp.arange(nbf)
    indices = cartesian_product(v,v,v,v)
    # 'Canonical' order, i>=j, k>=l, IJ>=KL
    #cond1 = indices[:,0] >= indices[:,1]
    #cond2 = indices[:,2] >= indices[:,3]
    #cond3 = indices[:,0] * (indices[:,0] + 1)/2 + indices[:,1] >= indices[:,2]*(indices[:,2]+1)/2 + indices[:,3]
    #mask = cond1 & cond2 & cond3
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

geom = np.array([0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]).reshape(-1,3)

#atom1_basis = np.repeat(np.array([0.5, 0.4, 0.3, 0.2]),8)
#atom2_basis = np.repeat(np.array([0.5, 0.4, 0.3, 0.2]),8)
#atom1_basis = np.array([0.5, 0.4])
#atom2_basis = np.array([0.5, 0.4])
atom1_basis = np.array([0.5])
atom2_basis = np.array([0.4])
basis = np.concatenate((atom1_basis, atom2_basis))
print(basis.shape)
nbf_per_atom = np.array([atom1_basis.shape[0],atom2_basis.shape[0]])
charge_per_atom = np.array([1.0,1.0])

def index2(i,j):
    if (i < j):
        return int(j * (j + 1) / 2 + i)
    else:
        return int(i * (i + 1) / 2 + j)

def index4(i,j,k,l):
    return index2(index2(i,j),index2(k,l))


#def vectorized_index2(indices, I=0, J=1, K=2, L=3):
#    '''Give positions of i,j,k,l '''
#    indices = onp.asarray(indices)
#    i,j,k,l = indices[:,I], indices[:,J], indices[:,K], indices[:,L]
#    print(i)
#    print(j)
#
#    cond1 = i < j
#    print(indices[cond1,1])
#    print(indices[cond1,0])
#    compound = onp.zeros((indices.shape[0]))
#
#    print(cond1)
#    if np.any(cond1): 
#        compound[cond1] = indices[cond1, J]  * (indices[cond1, J] + 1) / 2 + indices[cond1, I]
#    if np.any(~cond1): 
#        compound[~cond1] = indices[~cond1, I]  * (indices[~cond1, I] + 1) / 2 + indices[~cond1, J]
#
#    #compound[cond1] = indices[cond1, j]  * (indices[cond1, j] + 1) / 2 + indices[cond1, i]
#    print(compound)

def build_c_ikjl(indices):
    indices = onp.asarray(indices)
    i,j,k,l = indices[:,0], indices[:,1], indices[:,2], indices[:,3]
    cond = (i == k) | (j == l)
    c_ikjl = onp.where(cond, -0.5, -0.25)
    return c_ikjl
    
def build_c_iljk(indices):
    indices = onp.asarray(indices)
    i,j,k,l = indices[:,0], indices[:,1], indices[:,2], indices[:,3]
    cond1 = (i == j) | (k == l) # yield 0.0
    cond2 = (i == l) | (j == k) # yield 0.5 else 0.25
    c_iljk = onp.where(cond1, 0.0, onp.where(cond2, -0.5, -0.25))
    return c_iljk



def vectorized_index2(indicesI, indicesJ):
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
    compound_IJKL = vectorized_index2(vectorized_index2(indicesI, indicesJ),vectorized_index2(indicesK, indicesL))
    return compound_IJKL





def build_pk(g, indices, nbf):
    '''g: unique teis as a vector 
       indices: indices of each unique tei  (i<=j k<=l, IJ<=KL)
    '''
    pk = onp.zeros((g.shape[0]))
#    print("IJKL IKJL ILJK")
    for idx,val in enumerate(g):
        i,j,k,l = indices[idx][0],indices[idx][1],indices[idx][2],indices[idx][3]
        IJKL = index4(i,j,k,l)
        # J part
        pk[IJKL] += val

#        print(idx, '-->',IJKL, end=' ')

        IKJL = index4(i,k,j,l)
        if i == k or j == l:
            pk[IKJL] -= 0.5 * val
            print("c_ikjl", -0.5)
        else:
            pk[IKJL] -= 0.25 * val
            print("c_ikjl", -0.25)
#        print('  ', idx, '-->', IKJL, end=' ')

        # K/2 parts
        if i != j and k != l:
            ILJK = index4(i,l,j,k)
            if i == l or j == k:
                pk[ILJK] -= 0.5 * val
                print("c_iljk", -0.5)
            else:
                pk[ILJK] -= 0.25 * val
                print("c_iljk", -0.25)
#            print('   ',idx, '-->', ILJK, end='\n')
        else:
            print("c_iljk", 0.0)
#            print('  ',idx, '-->',-1, end='\n')


    # Set diagonal to one half value
    for ij in range(int(nbf * (nbf + 1) / 2)):
        r = index2(ij,ij)
        pk[r] *= 0.5
    return pk


def PK(g, indices, nbf):


    C_IKJL = build_c_ikjl(indices)
    C_ILJK = build_c_iljk(indices)










def indices_only(indices):


    for indice in indices:
        i,j,k,l = indice
        IJKL = index4(i,j,k,l)
        IKJL = index4(i,k,j,l)
        ILJK = index4(i,l,j,k)
    #    if i == k or j == l:
    #        c1 = 0.5
    #    else:
    #        c2 = 0.25

    #    if i != j and k != l:
    #        ILJK = index4(i,l,j,k)
    #        if i == l or j == k:
    #            c1 = 0.5
    #        else:
    #            c1 = 0.25


        

def test(geom):
    nbf = basis.shape[0]  
    g, indices = tei(geom,basis) 
    pk = build_pk(g, indices, nbf)

    indices = find_indices(nbf)
    C_IKJL = build_c_ikjl(indices)
    C_ILJK = build_c_iljk(indices)
    print(C_IKJL)
    print(C_ILJK)

    #indices_only(indices)

    #IJKL = vectorized_index4(indices[:,0], indices[:,1], indices[:,2], indices[:,3])
    #IKJL = vectorized_index4(indices[:,0], indices[:,2], indices[:,1], indices[:,3])
    #ILJK = vectorized_index4(indices[:,0], indices[:,3], indices[:,1], indices[:,2])



test(geom)





