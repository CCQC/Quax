import jax
import jax.numpy as np
import numpy as onp

def f(arg):

    def method1(arg):
        return 2*arg

    def method2(arg):
        return arg

    result = jax.lax.cond(arg < 1, arg, method1, arg, method2)
    return result

#gradient = jax.grad(f)
#gradient = jax.jacfwd(f)
gradient = jax.jacrev(f)

gradient(0.5)


