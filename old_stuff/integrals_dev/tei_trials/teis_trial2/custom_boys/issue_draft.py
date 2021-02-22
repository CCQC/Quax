import jax
from jax.interpreters import ad

def foo(x):
    return foo_p.bind(x)
foo_p = jax.core.Primitive('foo')

def foo_eval_rule(x):
    return jax.lax.rsqrt(x) * jax.lax.erf(jax.lax.sqrt(x))

def foo_jvp_rule(g, x):
    return g*(-(foo(x) - jax.lax.exp(-x)) / (2 * (x)))

foo_p.def_impl(foo_eval_rule)
ad.defjvp(foo_p, foo_jvp_rule)
print(foo(0.5))
print(jax.jacfwd(foo)(0.5))

print(jax.jacrev(foo)(0.5))
print(jax.grad(foo)(0.5))
print(jax.value_and_grad(foo)(0.5))








