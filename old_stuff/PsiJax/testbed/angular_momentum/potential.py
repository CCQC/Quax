import jax
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)

def gp(aa,bb,A,B):
    '''Gaussian product theorem. Returns center and coefficient of product'''
    R = (aa * A + bb * B) / (aa + bb)
    c = np.exp(np.dot(A-B,A-B) * (-aa * bb / (aa + bb)))
    return R,c

def boys(arg):
    '''F0(x) boys function'''
    return jax.scipy.special.erf(np.sqrt(arg + 1e-9)) * np.sqrt(np.pi) / (2 * np.sqrt(arg + 1e-9))

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

def potential(Ax, Ay, Az, Cx, Cy, Cz, aa, bb, geom, charge):
    '''
    Computes a single unnormalized electron-nuclear 
    potential energy integral over two primitive 
    s-orbital basis functions
    Ax,Ay,Az,Cx,Cy,Cz: cartesian coordinates of centers
    aa,bb: gaussian primitive exponents
    geom: Nx3 array of cartesian geometry 
    charge: N array of charges
    '''
    A = np.array([Ax, Ay, Az])
    B = np.array([Cx, Cy, Cz])
    g = aa + bb
    eps = 1 / (4 * g)
    P, c = gp(aa,bb,A,B)
    V = 0
    # For every atom
    for i in range(geom.shape[0]):
        arg = g * np.dot(P - geom[i], P - geom[i])
        F = boys(arg)
        V += -charge[i] * F * c * 2 * np.pi / g
    return V

geom = np.array([[0.0,0.0,-0.849220457955],
                 [0.0,0.0, 0.849220457955]])
charge = np.array([1.0,1.0])

#val = potential(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955, 0.5, 0.5,geom,charge)
## normalize
#Na = normalize(0.5,0,0,0)
#Nb = normalize(0.5,0,0,0)
#print(Na * Nb * val)

# Try (px|px)
px_px = jax.grad(jax.grad(potential, 2), 5)
val = px_px(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955, 0.5, 0.5,geom,charge) / (2 * 0.5 * 2 * 0.5)
Na = normalize(0.5,1,0,0)
Nb = normalize(0.5,1,0,0)
print(Na)
print(Na * Nb * val)


