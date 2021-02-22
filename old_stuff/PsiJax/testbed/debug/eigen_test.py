import jax
import jax.numpy as np

# Create a random symmetric matrix
seed = jax.random.PRNGKey(0)
tmp = jax.random.uniform(seed, (2,2))
a = np.dot(tmp, tmp.T) 

# Test derivatives of expressions involving eigenvectors
def test(inp):
    val, vec = np.linalg.eigh(inp)
    #return vec.T @ val @ vec
    #return np.einsum('ij,ji->', vec, vec) 
    #return np.tensordot('ij,ji->', vec, vec) 
    return np.tensordot(vec,vec, axes=([1,0],[0,1]))
    #return np.dot(vec,inp)

def test_deriv(func, inp):
    grad_func = jax.jit(jax.jacfwd(func))
    hess_func = jax.jit(jax.jacfwd(grad_func))
    cube_func = jax.jit(jax.jacfwd(hess_func)) # This derivative returns NaN, but jax.jacrev works here!
    print(grad_func(inp))
    print(hess_func(inp))
    print(cube_func(inp))

print(a)
print(test(a))
test_deriv(test, a)

## Test derivatives of expressions involving eigenvalues 
#def test2(a):
#    val, vec = np.linalg.eigh(a)
#    return np.dot(np.dot(val, a), val.T)
#    
#test_fwd_deriv(test1)
#test_fwd_deriv(test2)
#test_rev_deriv(test1)
#test_rev_deriv(test2)

#grad_func = jax.jacrev(test)
#hess_func = jax.jacrev(grad_func)
#cube_func = jax.jacrev(hess_func)
#print(grad_func(a))
#print(hess_func(a))
#print(cube_func(a))

