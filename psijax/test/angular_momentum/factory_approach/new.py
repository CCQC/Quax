import jax
import jax.numpy as np
import numpy as onp
onp.set_printoptions(linewidth=500)
from jax.config import config; config.update("jax_enable_x64", True)

def double_factorial(n):
    '''The double factorial function for small Python integer `n`.'''
    return np.prod(np.arange(n, 1, -2))

def normalize(aa,ax,ay,az):
    '''
    Normalization constant for gaussian basis function. 
    aa : orbital exponent
    ax : angular momentum component x
    ay : angular momentum component y
    az : angular momentum component z
    '''
    f = np.sqrt(double_factorial(2*ax-1) * double_factorial(2*ay-1) * double_factorial(2*az-1))
    N = (2*aa/np.pi)**(3/4) * (4 * aa)**((ax+ay+az)/2) / f
    return N

def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * np.dot(A-C, A-C)) / (aa + bb)))
    return ss

def overlap_factory(args, start_am, target_am, current, old=None, dim=6):
    '''
    An arbitrary angular momentum one electron integral function factory.
    Takes in some base s-orbital integral function, as well as target angular momentum vector
    Uses the fact that (a+1i|b) = 1/2 alpha * [ d/dAi (a|b) + ai (a-1i|b)]
    where alpha is the exponent on a, ai is the current angular momentum at index i in (a|b)

    Parameters
    ----------
    args: tuple
        Arguments of the function `current`, for overlap is (Ax, Ay, Az, Cx, Cy, Cz, aa, bb)
    start_am : onp.array
        Starting angular momentum vector (bra|ket):= [ax,ay,az,bx,by,bz] for integral (a|b). 
        Always will start with (s|s) := onp.array([0,0,0,0,0,0]), and this function will recursively construct
        function which computes integral with `target_am`
    target_am: onp.array
        Target angular momentum vector. Recursion will halt when we construct a function which computes the integral
        (a|b) with this desired angular momentum on the basis functions.
        A (px|dyz) integral target am would be, for example, [1,0,0,0,1,1].
    current: function 
        Current function for computing a one electron integral with arguments (Ax, Ay, Az, Cx, Cy, Cz, aa, bb)
        Starts with (s|s) function, gets promoted through the recursion.
    old: function
        Function for computing previous (a-1|b) or (a|b-1) function.
    Returns
    -------
    A function which computes the one electron integral with proper angular momentum on the basis functions
    NOTE: could just optionally return the evaluated integral!!!
    '''
    Ax, Ay, Az, Bx, By, Bz, aa, bb = args
    for idx in range(dim): 
        if start_am[idx] != target_am[idx]:
            # Define coefficients of terms
            ai = start_am[idx]
            # Build one-increment higher angular momentum function
            if ai == 0:
                if idx<=2:
                    def new(Ax, Ay, Az, Bx, By, Bz, aa, bb): 
                        return (1/(2*aa)) * (jax.grad(current,idx)(Ax, Ay, Az, Bx, By, Bz, aa, bb))
                else:
                    def new(Ax, Ay, Az, Bx, By, Bz, aa, bb): 
                        return (1/(2*bb)) * (jax.grad(current,idx)(Ax, Ay, Az, Bx, By, Bz, aa, bb))
            else:
                if idx<=2:
                    def new(Ax, Ay, Az, Bx, By, Bz, aa, bb): 
                        return (1/(2*aa)) * (jax.grad(current,idx)(Ax, Ay, Az, Bx, By, Bz, aa, bb) + ai * old(Ax, Ay, Az, Bx, By, Bz, aa, bb))
                else:
                    def new(Ax, Ay, Az, Bx, By, Bz, aa, bb): 
                        return (1/(2*bb)) * (jax.grad(current,idx)(Ax, Ay, Az, Bx, By, Bz, aa, bb) + ai * old(Ax, Ay, Az, Bx, By, Bz, aa, bb))
            promotion = onp.zeros(dim)
            promotion[idx] += 1
            return overlap_factory(args, start_am + promotion, target_am, new, current, dim=dim)
        else:
            continue
    return current

geom = np.array([[0.0,0.0,-0.849220457955],
                 [0.0,0.0, 0.849220457955]])
charge = np.array([1.0,1.0])

## S P test
exponents = np.repeat(0.5, 4)
nbf_per_atom = np.array([1,3])
centers = np.repeat(geom, nbf_per_atom, axis=0)
angular_momentum = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
nbf = exponents.shape[0]

Ax, Ay, Az = centers[0]
Bx, By, Bz = centers[1]
start_am = onp.array([0,0,0,0,0,0])
target_am = onp.array([0,0,0,1,0,0])
alpha_bra = exponents[0]
alpha_ket = exponents[1]

args = (Ax,Ay,Az,Bx,By,Bz,alpha_bra,alpha_ket)

overlap_s_px = jax.jit(overlap_factory(args, start_am, onp.array([0,0,0,0,1,0]), overlap_ss, old=None, dim=6))
overlap_s_py = jax.jit(overlap_factory(args, start_am, onp.array([0,0,0,0,1,0]), overlap_ss, old=None, dim=6))
overlap_s_pz = jax.jit(overlap_factory(args, start_am, onp.array([0,0,0,0,0,1]), overlap_ss, old=None, dim=6))

for i in range(nbf):
    for j in range(nbf):
        aa = exponents[i]
        bb = exponents[j]
        Ax, Ay, Az = centers[i]
        Bx, By, Bz = centers[j]

        Na = normalize(aa, 0, 0, 0)
        Nb = normalize(bb, 0, 0, 1)
        print(Na * Nb * overlap_s_pz(Ax,Ay,Az,Bx,By,Bz,aa,bb))





















