import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=500)

def boys(arg):
    '''
    F0(x) boys function. When x near 0, use taylor expansion, 
       F0(x) = sum over k to infinity:  (-x)^k / (k!(2k+1))
    Otherwise,
       F0(x) = sqrt(pi/(4x)) * erf(sqrt(x))
    '''
    if arg < 1e-8:
        boys = 1 - (arg / 3) + (arg**2 / 10) - (arg**3 / 42) + (arg**4 / 216)
    else:
        boys = jax.scipy.special.erf(np.sqrt(arg)) * np.sqrt(np.pi / (4 * arg))
    return boys

# jittable boys function

@jax.jit
#@jax.jarrett
def boys2(arg):
    #NOTE This expansion must go to same order as angular momentum, otherwise potential/eri integrals are wrong. 
    # This currently just supports up to g functions. (arg**4 term)
    #WAIT what about nuclear derivatives on top of this? Need longer expansion probably?
    # Could just add a fudge factor arg**10 / 1e10 to cover higher order derivatives dimensional cases
    # or some other function which is better?  + (x)
    result = np.where(arg < 1e-8, 1 - (arg / 3) + (arg**2 / 10) - (arg**3 / 42) + (arg**4 / 216), jax.scipy.special.erf(np.sqrt(arg)) * np.sqrt(np.pi / (4 * arg)))
    #result = np.where(arg < 1e-8, 1 - (arg / 3) + (arg**2 / 10) - (arg**3 / 42) , jax.scipy.special.erf(np.sqrt(arg)) * np.sqrt(np.pi / (4 * arg)))
    return result


print(boys(0.5))
print(boys2(0.5))

print(boys(1e-9))
print(boys2(1e-9))

print(jax.jacfwd(boys2)(1e-9))
print(jax.jit(jax.jacfwd(boys2))(1e-9))

print(jax.jacfwd(boys2)(1e-9))
print(jax.jit(jax.jacfwd(boys2))(1e-9))

print(jax.jit(jax.jacfwd(jax.jit(jax.jacfwd(jax.jit(jax.jacfwd(boys2))))))(1e-9))
print(jax.jacfwd(jax.jacfwd(jax.jacfwd(boys2)))(1e-9))

print(jax.jit(jax.jacfwd(jax.jit(jax.jacfwd(jax.jit(jax.jacfwd(boys2))))))(1e-9))
print(jax.jacfwd(jax.jacfwd(jax.jacfwd(boys2)))(1e-9))


#print(jax.jit(jax.jacfwd(jax.jit(jax.jacfwd(boys2))))(1e-9))
#print(jax.jit(jax.jacfwd(jax.jit(jax.jacfwd(jax.jit(jax.jacfwd(boys2))))))(1e-9))
#print(jax.jit(jax.jacfwd(jax.jit(jax.jacfwd(jax.jit(jax.jacfwd(jax.jit(jax.jacfwd(boys2))))))))(1e-9))
#print(jax.jit(jax.jacfwd(jax.jit(jax.jacfwd(jax.jit(jax.jacfwd(jax.jit(jax.jacfwd(jax.jit(jax.jacfwd(jax.jit(jax.jacfwd(boys2))))))))))))(1e-9))

#print(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(boys2)))))(1e-9))
#print(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(boys2)))))(1e-9))
#print(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jit(jax.jacfwd(boys2))))))(1e-9))
#print(jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(boys))))))(1e-9))
#print(jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(boys2))))))(1e-9))
#print(jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(jax.jacrev(boys))))))(1e-9))

