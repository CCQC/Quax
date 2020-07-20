import jax
import jax.numpy as np

# Create random vector of size N 
N = 2000
seed = jax.random.PRNGKey(0)
vec = jax.random.uniform(seed, (N,))

def idx_update_func(vec):
    y1 = np.sin(vec)
    y2 = y1**2
    y3 = y2 * 3
    M = np.zeros((N,N))
    M = jax.ops.index_update(M, np.diag_indices(N), y3)
    return np.sum(M)

def func(vec):
    y1 = np.sin(vec)
    y2 = y1**2
    y3 = y2 * 3
    M = np.zeros((N,N))
    M = M + np.diag(y3)
    return np.sum(M)

def test(function, arg):
    val = function(arg)
    #grad = jax.jacrev(function)(arg)
    #hess = jax.jacfwd(jax.jacrev(function))(arg)

# Peak memory usage: ~3.1 GB 
#test(idx_update_func, vec) 

# Peak memory usage: ~0.64 GB 
test(func, vec)           
