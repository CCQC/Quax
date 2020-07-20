import jax
import jax.numpy as np

def func(x):
    return np.sum(x**4)

def contract(f, x, vs):
    if vs is not None:
        v, vs = vs[0], vs[1:]
        #return jax.jvp(lambda z: contract(f,z,vs), x)(v)
        #return jax.jvp(lambda z: contract(f,z,vs), x, v)
        return jax.jvp(contract(f,x,vs), x, v)
    else:
        return jax.grad(f)(x)


x = np.array([1.0,2.0,3.0,4.0])

vs = np.array([[1.0,0.0,0.0,0.0],
               [0.0,1.0,0.0,0.0],
               [0.0,0.0,1.0,0.0],
               [0.0,0.0,0.0,1.0]])

y = func(x)
gradfunc = jax.grad(func)
hessfunc = jax.jacfwd(gradfunc)
g = gradfunc(x)
h = hessfunc(x)

print(y)
print(g)
print(h)

what = contract(func, x, vs)
print(what)
