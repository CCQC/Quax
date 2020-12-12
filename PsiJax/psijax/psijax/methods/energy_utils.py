import jax
from jax.config import config; config.update("jax_enable_x64", True)
from jax.experimental import loops
import jax.numpy as jnp
from functools import partial

def nuclear_repulsion(geom, nuclear_charges):
    """
    Compute the nuclear repulsion energy in a.u.
    """
    natom = nuclear_charges.shape[0]
    nuc = 0
    for i in range(natom):
        for j in range(i):
            nuc += nuclear_charges[i] * nuclear_charges[j] / jnp.linalg.norm(geom[i] - geom[j])
    return nuc

def symmetric_orthogonalization(S):
    """
    Compute the symmetric orthogonalization transform U = S^(-1/2)
    where S is the overlap matrix
    """
    # Warning: Higher order derivatives for some larger basis sets (TZ on) give NaNs for this algo 
    eigval, eigvec = jnp.linalg.eigh(S)
    cutoff = 1.0e-12
    above_cutoff = (abs(eigval) > cutoff * jnp.max(abs(eigval)))
    val = 1 / jnp.sqrt(eigval[above_cutoff])
    vec = eigvec[:, above_cutoff]
    A = vec.dot(jnp.diag(val)).dot(vec.T)
    return A

def cholesky_orthogonalization(S):
    """
    Compute the canonical orthogonalization transform U = VL^(-1/2) 
    where V is the eigenvectors and L diagonal inverse sqrt eigenvalues of the overlap matrix
    by way of cholesky decomposition
    Scharfenberg, Peter; A New Algorithm for the Symmetric (Lowdin) Orthonormalization; Int J. Quant. Chem. 1977
    """
    return jnp.linalg.inv(jnp.linalg.cholesky(S)).T

def old_tei_transformation(G, C):
    """
    Transform TEI's to MO basis.
    This algorithm is worse than below, since it creates intermediate arrays in memory.
    """
    G = jnp.einsum('pqrs, pP, qQ, rR, sS -> PQRS', G, C, C, C, C, optimize='optimal')
    return G

@jax.jit
def transform(C, G):
    return jnp.tensordot(C, G, axes=[(0,),(3,)])

def tei_transformation(G, C):
    """
    New algo for TEI transform
    It's faster than psi4.MintsHelper.mo_transform() for basis sets <~120.
    """
    G = transform(C,G)
    G = transform(C,G)
    G = transform(C,G)
    G = transform(C,G)
    return G

def partial_tei_transformation(G, Ci, Cj, Ck, Cl):
    G = jnp.einsum('pqrs, pP, qQ, rR, sS -> PQRS', G, Ci, Cj, Ck, Cl, optimize='optimal')
    return G
    
def cartesian_product(*arrays):
    '''
    JAX-friendly version of cartesian product. 
    '''
    tmp = jnp.asarray(jnp.meshgrid(*arrays, indexing='ij')).reshape(len(arrays),-1).T
    return tmp
