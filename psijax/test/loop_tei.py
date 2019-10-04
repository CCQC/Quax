import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=200)

#@jax.jit
@jax.jarrett
def normalize(aa):
    '''Normalization constant for s primitive basis functions. Argument is orbital exponent coefficient'''
    aa = ((2*aa)/np.pi)**(3/4)
    return aa

@jax.jarrett
def boys(arg):
    return jax.scipy.special.erf(np.sqrt(arg + 1e-9)) * np.sqrt(np.pi) / (2 * np.sqrt(arg + 1e-9))

#@jax.jarrett
@jax.jit
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
    F = boys(arg)
    G = F * Na * Nb * Nc * Nd * c1 * c2 * 2 * np.pi**2 / (g1 * g2) * np.sqrt(np.pi / (g1 + g2))
    return G

#def build_tei(basis, centers):
#@jax.jit
def update_old(I,basis,centers, i,j,k,l):
    return jax.ops.index_update(I, jax.ops.index[i,j,k,l], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))

#@jax.jit
def update(I,basis,centers, i,j,k,l):
    # thanks Sherill
    I  = jax.ops.index_update(I, jax.ops.index[i,j,k,l], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
    I  = jax.ops.index_update(I, jax.ops.index[k,l,i,j], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
    I  = jax.ops.index_update(I, jax.ops.index[j,i,l,k], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
    I  = jax.ops.index_update(I, jax.ops.index[l,k,j,i], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
    I  = jax.ops.index_update(I, jax.ops.index[j,i,k,l], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
    I  = jax.ops.index_update(I, jax.ops.index[l,k,i,j], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
    I  = jax.ops.index_update(I, jax.ops.index[i,j,l,k], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
    return jax.ops.index_update(I, jax.ops.index[k,l,j,i], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))

#@jax.jit
#TODO vectorize with outer products?
def find_indices_old(nbf):
    indices = []
    for i in range(nbf):
        for j in range(nbf):
            #if i >= j:
                for k in range(nbf):
                    for l in range(nbf):
                        if i>=j and k>=l and (i*(i+1)/2 + j >= k*(k+1)/2 + l): # thanks Crawford
                            indices.append([i,j,k,l])
    return np.asarray(indices)

def find_indices(nbf):
    v = onp.arange(nbf)
    indices = cartesian_product(v,v,v,v)
    cond1 = indices[:,0] >= indices[:,1]
    cond2 = indices[:,2] >= indices[:,3]
    cond3 = indices[:,0] * (indices[:,0] + 1) / 2 + indices[:,1] >= indices[:,2]*(indices[:,2]+1)/2 + indices[:,3]
    mask = cond1 & cond2 & cond3
    #return indices[mask,:]
    return np.asarray(indices[mask,:])


# Find all indices of ERI tensor given 4 arrays (np.arange(nbf), np.arange(nbf), np.arange(nbf), np.arange(nbf)s)
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = onp.result_type(*arrays)
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    #arr = onp.empty([len(a) for a in arrays] + [la])
    for i, a in enumerate(onp.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)



#@jax.jit
# Still too slow, try making array of unique indices, and unrolling the loop with lax.scan
def build_tei(basis, centers):
    nbf = basis.size
    I = np.zeros((nbf,nbf,nbf,nbf))

    for i in range(nbf):
        for j in range(nbf):
            for k in range(nbf):
                for l in range(nbf):
                    if i>=j and k>=l and (i*(i+1)/2 + j >= k*(k+1)/2 + l): # thanks Crawford
                        I = update(I,basis,centers,i,j,k,l)
    return I


#@jax.jit
def fast_tei(basis, centers, nbf):
    I = np.zeros((nbf,nbf,nbf,nbf))
    indices = find_indices(nbf)
    def body_func(I, indices):
        i,j,k,l = indices
        I  = jax.ops.index_update(I, jax.ops.index[i,j,k,l], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
        I  = jax.ops.index_update(I, jax.ops.index[k,l,i,j], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
        I  = jax.ops.index_update(I, jax.ops.index[j,i,l,k], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
        I  = jax.ops.index_update(I, jax.ops.index[l,k,j,i], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
        I  = jax.ops.index_update(I, jax.ops.index[j,i,k,l], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
        I  = jax.ops.index_update(I, jax.ops.index[l,k,i,j], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
        I  = jax.ops.index_update(I, jax.ops.index[i,j,l,k], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
        return jax.ops.index_update(I, jax.ops.index[k,l,j,i], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
    #I_final = jax.lax.scan(body_func, I, indices)
    test = jax.pmap(body_func, (None,0)) #lax.scan(body_func, I, indices)
    #test = jax.vmap(body_func, (None,0)) #lax.scan(body_func, I, indices)
    I_final = test(I, indices) 
    return I_final


#@jax.jit
def fast_tei2(basis,centers,nbf):
    indices = find_indices(nbf)
    def body_func(idx):
        i,j,k,l = idx
        tei = eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l])
        return tei
    test = jax.vmap(body_func, (0,)) 
    unique_teis = test(indices)
    #print(unique_teis)
    #I  = jax.ops.index_update(I, jax.ops.index[i,j,k,l], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
    #I  = jax.ops.index_update(I, jax.ops.index[k,l,i,j], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
    #I  = jax.ops.index_update(I, jax.ops.index[j,i,l,k], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
    #I  = jax.ops.index_update(I, jax.ops.index[l,k,j,i], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
    #I  = jax.ops.index_update(I, jax.ops.index[j,i,k,l], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
    #I  = jax.ops.index_update(I, jax.ops.index[l,k,i,j], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))
    #I  = jax.ops.index_update(I, jax.ops.index[i,j,l,k], eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l]))

    # Compute all unique indices for a given basis set size
    # for each unique indices, 
        # jax.ops.index_update(I), but up to 8 times, depending on the permutational species type
        # Type 1: all same (00|00), (11|11), etc, none
        # Type 2: 3a,1b (00|01)  (4 non redundant permutations)
        # Type 3: 2a,1b,1c (00|12)
        # Type 4: 2a,1b,1c (00|01)
    # unroll it with lax.scan, or vectorized with vmap, or parallelize with pmap

geom = np.array([0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]).reshape(-1,3)

def build_basis(geom):
    atom1_basis = np.repeat(np.array([0.5, 0.4, 0.3, 0.2]),4)
    atom2_basis = np.repeat(np.array([0.5, 0.4, 0.3, 0.2]),4)
    #atom1_basis = np.array([0.5, 0.4, 0.3, 0.2])
    #atom2_basis = np.array([0.5, 0.4, 0.3, 0.2])
    #atom1_basis = np.array([0.5, 0.4])
    #atom2_basis = np.array([0.5, 0.4])
    #atom1_basis = np.array([0.5])
    #atom2_basis = np.array([0.4])
    basis = np.concatenate((atom1_basis, atom2_basis))
    centers = np.concatenate((np.tile(geom[0],atom1_basis.size).reshape(-1,3), np.tile(geom[1],atom2_basis.size).reshape(-1,3)))
    return basis, centers


def benchmark(geom):
    basis, centers = build_basis(geom)
    #print(basis.size)
    I = build_tei(basis, centers)
    fake = np.sum(I)
    return fake

#val = benchmark(geom)
#basis, centers = build_basis(geom)
#I = build_tei(basis, centers)

basis, centers = build_basis(geom)
#print(basis.size)
#I = fast_tei(basis, centers, 24)
#print(I)
I = fast_tei2(basis, centers, basis.shape[0])
print(I)

#basis, centers = build_basis(geom)
#print(basis.shape)
#I = build_tei(basis, centers)
#print(I)
