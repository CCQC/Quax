import jax
import jax.numpy as np

seed = jax.random.PRNGKey(0)
A = jax.random.uniform(seed, (5,5))
B = jax.random.uniform(seed, (5,5,5,5))

def test(A,B):
    C = np.einsum('pqrs,rs->pq', B, A)
    D = np.einsum('prqs,rs->pq', B, A)
    out = np.einsum('pq,pq->', C, D)
    return out


grad_func = jax.jacfwd(test)
hess_func = jax.jacfwd(grad_func)
cube_func = jax.jacfwd(hess_func)
print(grad_func(A,B))
print(hess_func(A,B))

print(cube_func(A,B))
# ^^^ This fails with RuntimeError: Invalid argument: Input dimension should be either 1 
# or equal to the output dimension it is broadcasting into; 
# the 3th operand dimension is 5, the 3th output dimension is 25.: 
# This is a bug in JAX's shape-checking rules; please report it!

# Using reverse-mode instead for the third derivatives works fine:
#cube_func = jax.jacrev(hess_func)
#print(cube_func(A,B))

# Also, making the third derivatives non-zero causes no issue! 
def test2(A,B):
    C = np.einsum('pqrs,rs->pq', B, A)
    D = np.einsum('prqs,rs->pq', B, A)
    out = np.einsum('pq,pq->', C**2, D**2)
    return out

grad_func = jax.jacfwd(test2)
hess_func = jax.jacfwd(grad_func)
cube_func = jax.jacfwd(hess_func)
print(grad_func(A,B))
print(hess_func(A,B))
print(cube_func(A,B))

