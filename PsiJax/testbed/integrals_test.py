import jax
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=200)


#@jax.jarrett
@jax.jit
def gp(aa,bb,A,B):
    '''Gaussian product theorem. Returns center and coefficient of product'''
    R = (aa * A + bb * B) / (aa + bb)
    c = np.exp(np.dot(A-B,A-B) * (-aa * bb / (aa + bb)))
    return R,c

@jax.jarrett
@jax.jit
def normalize(aa):
    '''Normalization constant for s primitive basis functions. Argument is orbital exponent coefficient'''
    #N = ((2*aa)/np.pi)**(3/4)
    #return N
    aa = ((2*aa)/np.pi)**(3/4)
    return aa

#@jax.jarrett
@jax.jit
def overlap(aa, bb, A, B):
    '''Computes a single overlap integral over two primitive s-orbital basis functions'''
    Na = normalize(aa)
    Nb = normalize(bb)
    R,c = gp(aa,bb,A,B)
    S = Na * Nb * c * (np.pi / (aa + bb)) ** (3/2)
    return S

@jax.jit
def build_oei(basisA, basisB, A, B):
    '''Builds overlap/kinetic/potential one-electron integral matrix of diatomic molecule with s-orbital basis functions, based on 'mode' argument'''
    nbfA = basisA.size
    nbfB = basisB.size 
    nbf = nbfA + nbfB
    I = np.zeros((nbf,nbf))
    basis = np.concatenate((basisA,basisB), axis=0).reshape(-1,1)

    An = np.tile(A, nbfA).reshape(nbfA,3)
    Bn = np.tile(B, nbfB).reshape(nbfB,3)
    centers = np.concatenate((An,Bn),axis=0)

    for i,b1 in enumerate(basis):
        for j in range(i+1):
            s = overlap(b1, basis[j], centers[i], centers[j])
            I = jax.ops.index_update(I, jax.ops.index[i,j], s[0])
            I = jax.ops.index_update(I, jax.ops.index[j,i], s[0])
    return I


geom = np.array([0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]).reshape(-1,3)

#basis = np.array([0.5, 0.4, 0.3, 0.2])
#basis = np.repeat(np.array([0.5, 0.4, 0.3, 0.2]), 4)
#print('basis', basis)
#print(overlap(basis[0], basis[1], geom[0], geom[1]))

#overlap(basis[0], basis[1], geom[0], geom[1])
S = build_oei(basis, basis, geom[0],geom[1])
print(S)



