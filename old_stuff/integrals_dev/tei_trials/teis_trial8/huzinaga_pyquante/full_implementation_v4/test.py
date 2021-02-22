import jax


@jax.jit
def f(x):
    return 2 * x

f(2)
