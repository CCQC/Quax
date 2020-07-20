import jax
import jax.numpy as np
import numpy as onp
from scipy import special
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)


def find_boys_arg(A,B,C,D,aa,bb,cc,dd):
    zeta = aa + bb 
    eta = cc + dd
    boys_arg = (zeta * eta / (zeta + eta)) * \
               np.dot((aa * A + bb * B) / zeta - (cc * C + dd * D) / eta, (aa * A + bb * B) / zeta - (cc * C + dd * D) / eta)
    print(boys_arg)

find_boys_arg(np.array([0.0,0.0,0.0]), np.array([0.0,0.0,10.0]), np.array([0.0,0.0,20.0]), np.array([0.0,0.0,30.0]), 30000, 2.53, 0.111, 30000)

"""
In this scheme, we leverage 'static_argnums' to obtain boys function values and treat them as constants.
This way no grid needs to be stored!
"""

@partial(jax.jit, static_argnums=(0,1))
def boys_analytic(n,x):
    return special.hyp1f1(n+0.5,n+1.5,-x) / (2 * n + 1)
    
#def boys_general(n_dum, x_dum, x, n):
@partial(jax.jit, static_argnums=(2,3))
def boys_general(n,x, n2, x2):
    '''
    Eqns 21, 24, 24, 25, 26 from Weiss, Ochsenfeld, J. Comp. Chem. 2015 36 1390
    Note eqn 23 typo: First term F0 should be Fn+0
    '''
    with jax.disable_jit():
        f0 = boys_analytic(n2,x2) 
        f1 = boys_analytic(n2+1,x2) 
        f2 = boys_analytic(n2+2,x2) 
        f3 = boys_analytic(n2+3,x2) 

    a0 = f0 + x * f1 + 0.5 * x**2 * f2 + (1/6) * x**3 * f3
    a1 = f1 + x * f2 + 0.5 * x**2 * f3 
    a2 = 0.5 * f2 + 0.5 * x * f3
    a3 = (1/6) * f3
    F = a0 - x * (a1 - x * (a2 - x * a3))
    return F

@partial(jax.jit, static_argnums=(1))
def boys0(x, x2):
    '''
    Eqns 21, 24, 24, 25, 26 from Weiss, Ochsenfeld, J. Comp. Chem. 2015 36 1390
    Note eqn 23 typo: First term F0 should be Fn+0
    '''
    with jax.disable_jit():
        f0 = boys_analytic(0,x2) 
        f1 = boys_analytic(1,x2) 
        f2 = boys_analytic(2,x2) 
        f3 = boys_analytic(3,x2) 

    a0 = f0 + x * f1 + 0.5 * x**2 * f2 + (1/6) * x**3 * f3
    a1 = f1 + x * f2 + 0.5 * x**2 * f3 
    a2 = 0.5 * f2 + 0.5 * x * f3
    a3 = (1/6) * f3
    F = a0 - x * (a1 - x * (a2 - x * a3))
    return F

print(boys0(0.5,0.5))

# Cant jit this, otherwise its mad. If anything thats a tracer gets pass 
#@jax.jit
#def test(a,b,c,d):
#    return boys_general(a,b,c,d)

#test(1,0.5,1,0.5)

#print(boys_general(0, 0.5, 0, 0.5), boys_analytic(0,0.5))
#print(boys_general(1, 0.6, 1, 0.6), boys_analytic(1,0.6))
#print(boys_general(2, 0.7, 2, 0.7), boys_analytic(2,0.7))
#print(boys_general(1, 0.5, 1, 0.5), boys_analytic(1,0.5))
#print(boys_general(2, 0.5, 2, 0.5), boys_analytic(2,0.5))
#print(boys_general(3, 0.5, 3, 0.5), boys_analytic(3,0.5))





#@jax.jit
#def boys_general(n, x):
#    denom = 2 * n + 1
#    with jax.disable_jit():
#        num = special.hyp1f1(n+0.5,n+1.5,-x)
#    return num / denom
#
#print(boys_general(0, 30),np.sqrt(np.pi) / (2 * np.sqrt(30)))
