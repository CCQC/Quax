import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=500)

@jax.jit
def cartesian_product(*arrays):
    '''JAX-friendly version of cartesian product. Same order as other function, more memory requirements though.'''
    #tmp = np.asarray(np.meshgrid(*arrays, indexing='ij')).reshape(len(arrays),-1).T
    tmp = np.meshgrid(*arrays, indexing='ij')
    return tmp

def old_cartesian_product(*arrays):
    '''Generalized cartesian product of any number of arrays'''
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

@jax.jit
def boys(arg):
    """
    JAX-compatible function for evaluating F0(x) (boys function for 0 angular momentum):
    F0(x) = sqrt(pi/(4x)) * erf(sqrt(x)). 
    For small x, denominator blows up, so a taylor expansion is typically used:   
    F0(x) = sum over k to infinity:  (-x)^k / (k!(2k+1))
    """
    # First option: just force it to be 1 when 'arg' is small (
    # This is relatively fast, but inaccurate for small arg, i.e. gives 1.0 instead of 0.999999999333367) or w/e
    #TODO what cutoff can you get away with???
    sqrtarg = np.sqrt(arg)
    result = np.where(arg < 1e-10, 1.0, jax.scipy.special.erf(sqrtarg) * np.sqrt(np.pi) / (2 * sqrtarg))
    # alternative, use expansion, however the expansion must go to same order as angular momentum, otherwise
    #potential/eri integrals are wrong. 
    # It is unknown whether nuclear derivatives on top of integral derivatives will cause numerical issues
    # This currently just supports up to g functions. (arg**4 term)
    # Could just add a fudge factor arg**10 / 1e10 to cover higher order derivatives dimensional cases or some other function which is better? 
    #result = np.where(arg < 1e-8, 1 - (arg / 3) + (arg**2 / 10) - (arg**3 / 42) + (arg**4 / 216), jax.scipy.special.erf(np.sqrt(arg)) * np.sqrt(np.pi / (4 * arg)))
    return result

def gaussian_product(alpha_bra,alpha_ket,A,C):
    '''Gaussian product theorem. Returns center and coefficient of product'''
    R = (alpha_bra * A + alpha_ket * C) / (alpha_bra + alpha_ket)
    c = np.exp(np.dot(A-C,A-C) * (-alpha_bra * alpha_ket / (alpha_bra + alpha_ket)))
    return R,c

def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns a (s|s) overlap integral
    """
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    ss = ((np.pi / (alpha_bra + alpha_ket))**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(A-C, A-C)) / (alpha_bra + alpha_ket)))
    return ss * c1 * c2

def kinetic_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns a (s|s) kinetic integral
    """
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    P = (alpha_bra * alpha_ket) / (alpha_bra + alpha_ket)
    ab = -1.0 * np.dot(A-C, A-C)
    K = overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2) * (3 * P + 2 * P * P * ab)
    return K

def potential_ss(Ax, Ay, Az, Cx, Cy, Cz, geom, charge, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns a (s|s) potential integral
    """
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    g = alpha_bra + alpha_ket
    P, c = gaussian_product(alpha_bra,alpha_ket,A,C)
    V = 0
    # For every atom
    for i in range(geom.shape[0]):
        arg = g * np.dot(P - geom[i], P - geom[i])
        F = boys(arg)
        V += -charge[i] * F * c * 2 * np.pi / g
    return V * c1 * c2

geom = np.array([0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]).reshape(-1,3)
charge = np.array([1.0,1.0])
Ax = 0.0
Ay = 0.0
Az = -0.849220457955
Cx = 0.0
Cy = 0.0
Cz = 0.849220457955
alpha_bra = 0.5 
alpha_ket = 0.4
c1 = 0.4237772081237576
c2 = 0.35847187357690596

print(overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
print(kinetic_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2))
print(potential_ss(Ax, Ay, Az, Cx, Cy, Cz, geom, charge, alpha_bra, alpha_ket, c1, c2))





