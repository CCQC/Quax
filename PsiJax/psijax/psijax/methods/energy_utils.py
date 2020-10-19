import jax
from jax.config import config; config.update("jax_enable_x64", True)
config.enable_omnistaging()
from jax.experimental import loops
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

def old_tei_transformation(G, C):
    """
    Transform TEI's to MO basis.
    This algorithm is worse than below, since it creates intermediate arrays in memory.
    """
    G = np.einsum('pqrs, pP, qQ, rR, sS -> PQRS', G, C, C, C, C, optimize='optimal')
    return G

@jax.jit
def tmp_transform(G, C):
    return np.tensordot(C, G, axes=(0,0))

def tei_transformation(G, C):
    """
    New algo for TEI transform
    It's faster than psi4.MintsHelper.mo_transform()
    """
    # New algo: call same jitted einsum routine, but transpose array
    G = tmp_transform(G, C)           # (A,b,c,d)
    G = np.transpose(G, (1,0,2,3))    # (b,A,c,d)
    G = tmp_transform(G, C)           # (B,A,c,d)
    G = np.transpose(G, (2,0,1,3))    # (c,B,A,d)
    G = tmp_transform(G, C)           # (C,B,A,d)
    G = np.transpose(G, (3,0,1,2))    # (d,C,B,A)
    G = tmp_transform(G, C)           # (D,C,B,A)
    return G

def partial_tei_transformation(G, Ci, Cj, Ck, Cl):
    G = np.einsum('pqrs, pP, qQ, rR, sS -> PQRS', G, Ci, Cj, Ck, Cl, optimize='optimal')
    return G
    
