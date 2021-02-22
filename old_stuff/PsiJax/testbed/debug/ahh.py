
import jax
import jax.numpy as np

# Create a random symmetric matrix
seed = jax.random.PRNGKey(0)
tmp = jax.random.uniform(seed, (2,2))
a = np.dot(tmp, tmp.T)

def test(inp):
    val, vec = np.linalg.eigh(inp)
    return np.dot(np.dot(vec, inp), vec.T)

def test_deriv(func, inp):
    grad_func = jax.jacfwd(func)
    hess_func = jax.jacfwd(grad_func)
    cube_func = jax.jacfwd(hess_func) # This derivative returns NaN, but jax.jacrev works!
    print(grad_func(inp))
    print(hess_func(inp))
    print(cube_func(inp))

test_deriv(test, a)
