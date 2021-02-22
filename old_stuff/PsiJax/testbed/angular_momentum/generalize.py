import jax
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
# basis functions exponents:  [3,4,5,43,4,5,6,34,34]
# basis function centers [ [0,3,4], [34,34,545], [34,34,34],...]


# S =   
#        s, px, py, pz, s, px, py, pz
#     s
#    px
#    py
#    pz
#     s
#    px
#    py
#    pz

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

val = overlap_ss(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955, 0.5, 0.5)
Na = normalize(0.5,0,0,0)
Nb = normalize(0.5,0,0,0)
print(Na * Nb * val)


def overlap_general(Ax, Ay, Az, Cx, Cy, Cz, aa, bb, amA, amB):

    ss = overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, aa, bb)
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * np.dot(A-C, A-C)) / (aa + bb)))
    return ss

Ax = 0.0
Ay = 0.0
Az =-0.849220457955
Bx = 0.0
By = 0.0
Bz = 0.849220457955
aa = 0.5
bb = 0.5

amA = [1,0,0]
amB = [1,0,0]


# Returns function that computes (px|s)

# Idea: make overlap function which takes a factor to multiply the result by
# (px|s) (s|px)
def overlap_ps(ss_func, i, args):
    '''Retruns function which computes integrals if i = 0, (px|s), 1 (py|s), 2 (pz|s), 3 (s|px), 4 (s|py), 5 (s|pz) '''
    if i <= 2: 
        return jax.grad(ss_func, i)(*args) / (2 * aa)
    else:
        return jax.grad(ss_func, i)(*args) / (2 * bb)


def make_overlap_ps(ss_func, i):
    '''Returns function which evaluates (p|s) or (s|p) and the factor which must be multiplied to get original integral.'''
    if i <= 2: 
        return jax.grad(ss_func, i), (1 / (2 * aa))
    else:
        return jax.grad(ss_func, i), (1 / (2 * bb))

def eval_overlap_ps(ss_func, i, args):
    func, factor = make_overlap_ps(ss_func, i)
    return func(*args) * factor

def make_overlap_ds(ss_func, i, j):
    ps_func, ps_factor = make_overlap_ps(ss_func,i)
    if i <= 2: 
        return jax.grad(ps_func, i), ps_factor, (1 / (2*aa))
    else:
        return jax.grad(ss_func, i), ps_factor, (1 / (2 * bb))

def eval_overlap_ds(ss_func, i, j, args):
    func, ps_factor, ds_factor = make_overlap_ds(ss_func, i, j)
    if i <= 2: 
        func(*args) * ps_factor * ds_factor + amA[i] * ds_factor * ss_func(*args)
    else:
        func(*args) * ps_factor * ds_factor + amB[i] * ds_factor * ss_func(*args)
    
    



#px_s = jax.grad(overlap_ss, 0)
#val = px_s(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955, 0.5, 0.5)
#print(val)

px_s = overlap_ps(overlap_ss, 2, (Ax, Ay, Az, Bx, By, Bz, aa, bb))
#val = get_px_s(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955, 0.5, 0.5)
print(px_s)


