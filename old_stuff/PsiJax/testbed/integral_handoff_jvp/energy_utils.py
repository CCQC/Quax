import jax
from jax.config import config; config.update("jax_enable_x64", True)
config.enable_omnistaging()
import jax.numpy as np
import numpy as onp
from functools import partial

def nuclear_repulsion(geom, nuclear_charges):
    """
    Compute the nuclear repulsion energy in a.u.
    """
    natom = nuclear_charges.shape[0]
    nuc = 0
    for i in range(natom):
        for j in range(i):
            nuc += nuclear_charges[i] * nuclear_charges[j] / np.linalg.norm(geom[i] - geom[j])
    return nuc

def symmetric_orthogonalization(S):
    """
    Compute the symmetric orthogonalization transform U = S^(-1/2)
    where S is the overlap matrix
    """
    # Warning: Higher order derivatives for some larger basis sets (TZ on) give NaNs for this algo 
    eigval, eigvec = np.linalg.eigh(S)
    cutoff = 1.0e-12
    above_cutoff = (abs(eigval) > cutoff * np.max(abs(eigval)))
    val = 1 / np.sqrt(eigval[above_cutoff])
    vec = eigvec[:, above_cutoff]
    A = vec.dot(np.diag(val)).dot(vec.T)
    return A

def cholesky_orthogonalization(S):
    """
    Compute the canonical orthogonalization transform U = VL^(-1/2) 
    where V is the eigenvectors and L diagonal inverse sqrt eigenvalues of the overlap matrix
    by way of cholesky decomposition
    Scharfenberg, Peter; A New Algorithm for the Symmetric (Lowdin) Orthonormalization; Int J. Quant. Chem. 1977
    """
    return np.linalg.inv(np.linalg.cholesky(S)).T

@jax.jit
def tei_transformation(G, C):
    G = np.einsum('pqrs, sS, rR, qQ, pP -> PQRS', G, C, C, C, C, optimize='optimal')
    return G

@jax.jit
def partial_tei_transformation(G, Ci, Cj, Ck, Cl):
    G = np.einsum('pqrs, sS, rR, qQ, pP -> PQRS', G, Ci, Cj, Ck, Cl, optimize='optimal')
    return G
    
