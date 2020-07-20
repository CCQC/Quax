import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)

def double_factorial(n):
    '''The double factorial function for small Python integer `n`.'''
    return np.prod(np.arange(n, 1, -2))

@jax.jit
def odd_double_factorial(x): # this ones jittable, roughly equal speed
    n = (x + 1)/2
    return 2**n * np.exp(jax.scipy.special.gammaln(n + 0.5)) / (np.pi**(0.5))

def cartesian_product(*arrays):
    '''Generalized cartesian product of any number of arrays'''
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

@jax.jit
def normalize(aa,ax,ay,az):
    '''
    Normalization constant for gaussian basis function. 
    aa : orbital exponent
    ax : angular momentum component x
    ay : angular momentum component y
    az : angular momentum component z
    '''
    f = np.sqrt(odd_double_factorial(2*ax-1) * odd_double_factorial(2*ay-1) * odd_double_factorial(2*az-1))
    N = (2*aa/np.pi)**(3/4) * (4 * aa)**((ax+ay+az)/2) / f
    return N
# Vectorized version of normalize
vectorized_normalize = jax.vmap(normalize)

def contracted_normalize(exponents,coeff,ax,ay,az):
    '''Normalization constant for a single contracted gaussian basis function'''
    K = exponents.shape[0]  # Degree of contraction K
    L = ax + ay + az        # Total angular momentum L
    c_times_c = np.outer(coeff,coeff)
    a_plus_a = np.broadcast_to(exponents, (K,K)) + np.transpose(np.broadcast_to(exponents, (K,K)), (1,0))
    prefactor = (np.pi**(1.5) * double_factorial(2*ax-1) * double_factorial(2*ay-1) * double_factorial(2*az-1)) / 2**L
    sum_term = np.sum(c_times_c / (a_plus_a**(L + 1.5)))
    return (prefactor * sum_term) ** -0.5

def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * np.dot(A-C, A-C)) / (aa + bb)))
    return ss

# Vectorized version of overlap_ss
vectorized_overlap_ss = jax.jit(jax.vmap(overlap_ss, (None,None,None,None,None,None,0,0)))

def prep_coefficients(exp,coeffs,ax,ay,az):
    '''
    Builds-in primitive and contracted normalization constants into contraction coefficients

    Parameters:
    -----------
    exp : ndarray
        1-D numpy array of exponents for this contracted basis function
    coeffs : ndarray
        1-D numpy array of coefficients for this contracted basis function
    ax: int 
        Angular momentum x component
    ay: int 
        Angular momentum y component
    az: int 
        Angular momentum z component

    Returns:
    --------
    coeffs : 1-D numpy array of normalized contraction coefficients 
    '''
    size = exp.shape[0]
    # vectorize inputs and obtain all primitive normalization constants
    ax_v, ay_v, az_v = np.repeat(ax,size), np.repeat(ay,size), np.repeat(az,size),
    primitive_norms = vectorized_normalize(exp, ax_v, ay_v, az_v)
    coeffs = coeffs * primitive_norms
    #print(coeffs)
    # obtain contraction normalization constants
    contracted_norm = contracted_normalize(exp, coeffs, ax, ay, az)
    coeffs = coeffs * contracted_norm
    return coeffs




geom = np.array([[0.0,0.0,-0.849220457955],
                 [0.0,0.0, 0.849220457955]])

#contracted_overlap = jax.jit(jax.vmap(overlap_ss, (None,None,None,None,None,None,0,0)))

# This is a basis function
#exps =   np.array([0.5,0.4])
#coeffs = np.array([0.75,0.25])
#exps =   np.array([0.5,0.4,0.3,0.2])
#coeffs = np.array([0.1,0.2,0.3,0.4])
#
#bra_size = exps.shape[0]
#ket_size = exps.shape[0]
#
## Generate coefficients with correct normalization
#ang_mom_x, ang_mom_y, ang_mom_z = np.zeros_like(exps), np.zeros_like(exps), np.zeros_like(exps)
#primitive_norms = jax.vmap(normalize)(exps, ang_mom_x, ang_mom_y, ang_mom_z)
#coeffs = coeffs * primitive_norms
#contracted_norm = contracted_normalize(exps, coeffs, 0, 0, 0)
#coeffs = coeffs * contracted_norm
#
#Ax, Ay, Az = geom[0]
#Bx, By, Bz = geom[1]
#
#
#bra_exp = np.tile(exps, bra_size)
#ket_exp = np.repeat(exps, ket_size)
#
#result = vectorized_overlap_ss(Ax, Ay, Az, Bx, By, Bz, bra_exp, ket_exp)
#print(result)
#
#c1_full = np.tile(coeffs, bra_size)
#c2_full = np.repeat(coeffs, ket_size)
#print('coefficients')
#print(c1_full * c2_full)
#print(np.outer(coeffs,coeffs))
#
#print(c1_full * c2_full * result)
#final = np.sum(c1_full * c2_full * result)
#print(final)


mydict = {
          0: {'am':'s',
              'atom' : 0,
              'coeff': np.array([0.1,0.2,0.3,0.4]),
              'exp':   np.array([0.1,0.2,0.3,0.4])},
          1: {'am':'s',
              'atom' : 0,
              'coeff': np.array([0.5,0.6,0.7,0.8]),
              'exp':   np.array([0.5,0.6,0.7,0.8])},
          2: {'am':'s',
              'atom' : 1,
              'coeff': np.array([0.1,0.2,0.3,0.4]),
              'exp':   np.array([0.1,0.2,0.3,0.4])},
          3: {'am':'s',
              'atom' : 1,
              'coeff': np.array([0.5,0.6,0.7,0.8]),
              'exp':   np.array([0.5,0.6,0.7,0.8])},
         }

nshells = len(mydict)
for i in range(nshells):
    for j in range(nshells):
        # Load data for this contracted integral
        c1 = mydict[i]['coeff']
        c2 = mydict[j]['coeff']
        exp1 = mydict[i]['exp']
        exp2 = mydict[j]['exp']
        atom1 = mydict[i]['atom']
        atom2 = mydict[j]['atom']
        Ax,Ay,Az = geom[atom1]
        Bx,By,Bz = geom[atom2]

        #TODO hard-coded. How to deal with px, py, pz? Th
        ax,ay,az = 0,0,0
        # Psi4 normalized coefficients from basis.shell.coef are these coefficients
        #c1_norm = prep_coefficients(exp1, c1, ax, ay, az)
        #c2_norm = prep_coefficients(exp2, c2, ax, ay, az)
        print(c1_norm)
        print(c2_norm)

        exp_combos = cartesian_product(exp1,exp2)
        primitives = vectorized_overlap_ss(Ax,Ay,Az,Bx,By,Bz,exp_combos[:,0],exp_combos[:,1])

        # this is a different method, is it right?
        coefficients = np.einsum('i,j->ij', c1_norm, c2_norm).flatten()
        #print(coefficients)
        result = np.sum(primitives * coefficients)


