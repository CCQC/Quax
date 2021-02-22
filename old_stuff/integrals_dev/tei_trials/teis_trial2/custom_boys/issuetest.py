import numpy as onp
from jax.core import Primitive
from jax.interpreters.ad import defvjp
from jax import grad
import jax

# Define function to be differentiate
def foo(x):
    return foo_p.bind(x)
foo_p = Primitive('foo')

def f(x):
    return onp.sin(x)
foo_p.def_impl(f)

def dfoo(g, x):
    return g*bar(x)

defvjp(foo_p, dfoo)

def bar(x):
    return bar_p.bind(x)
bar_p = Primitive('bar')

def g(x):
    return onp.cos(x)
bar_p.def_impl(g)

def dbar(g, x):
    return -g*foo(x)

defvjp(bar_p, dbar)

jax.interpreters.ad.defjvp(bar_p, dbar)



print(jax.jacrev(foo)(0.5))
print(jax.jacrev(bar)(0.5))

print(jax.jacfwd(bar)(0.5))
print(jax.jacfwd(foo)(0.5))
