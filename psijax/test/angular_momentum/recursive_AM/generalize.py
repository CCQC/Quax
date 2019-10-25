import jax
import jax.numpy as np
import numpy as onp
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

def recursively_promote_oei(args, start_am, target_am, current, old=None):
    '''
    An arbitrary angular momentum one electron integral function factory.
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
    for idx in range(6): 
        if start_am[idx] != target_am[idx]:
            # Define coefficients of terms
            ai = start_am[idx]
            if idx <= 2:
                alpha = args[-2] 
            else:
                alpha = args[-1] 
            # Build one-increment higher angular momentum function
            if ai == 0:
                def new(*args): 
                    return (1/(2*alpha)) * (jax.grad(current,idx)(*args))
            else:
                def new(*args): 
                    # Uh oh... `old` is incorrect when going from [1,0,0,0,0,0] to [1,0,0,1,0,0], right?
                    return (1/(2*alpha)) * (jax.grad(current,idx)(*args) + ai * old(*args))
              
            # Increment angular momentum vector by one in the proper place.
            promotion = onp.zeros(6)
            promotion[idx] += 1
            return recursively_promote_oei(args, start_am + promotion, target_am, new, current)
        else:
            continue
    return current


# H2 with 0.5 exponents on basis function    
args = (0.0,0.0, 0.849220457955,0.0,0.0, 0.849220457955, 0.5, 0.5)
bra_am = onp.array([0,2,0])
ket_am = onp.array([0,0,2])

start_am = onp.array([0,0,0,0,0,0])
target_am = onp.hstack((bra_am, ket_am))

newfunc = recursively_promote_oei(args, start_am, target_am, overlap_ss, old=None)
Na = normalize(args[-2],bra_am[0],bra_am[1],bra_am[2])
Nb = normalize(args[-1],ket_am[0],ket_am[1],ket_am[2])

val = newfunc(*args) 
print(Na * Nb * val)


# Create exponents vector for H: S 0.5 and H: D 0.5
exponents = np.repeat(0.5, 7)
nbf_per_atom = np.array([1,6])
geom = np.array([[0.0,0.0,-0.849220457955],
                 [0.0,0.0, 0.849220457955]])
centers = np.repeat(geom, nbf_per_atom, axis=0)
# dxx dxy dxz dyy dyz dzz
angular_momentum = np.array([[0,0,0], [2,0,0], [1,1,0], [1,0,1], [0,2,0], [0,1,1], [0,0,2]])
print(exponents)
print(centers)
print(angular_momentum)

for i, aa in enumerate(exponents):
    for j, bb in enumerate(exponents):
        Ax, Ay, Az = centers[i]
        Bx, By, Bz = centers[j]
        pi, pj, pk = angular_momentum[i]
        qi, qj, qk = angular_momentum[j]

        start_am = onp.array([0,0,0,0,0,0])
        target_am = onp.array([pi,pj,pk,qi,qj,qk])
        args = (Ax,Ay,Az,Bx,By,Bz,aa,bb)
        integral_func = recursively_promote_oei(args, start_am, target_am, overlap_ss, old=None)

        Na = normalize(args[-2],pi, pj, pk)
        Nb = normalize(args[-1],qi, qj, qk)
        val = integral_func(*args) 
        print(Na * Nb * val)








