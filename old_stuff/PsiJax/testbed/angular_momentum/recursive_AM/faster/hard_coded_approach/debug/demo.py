import jax
import jax.numpy as np
import numpy as onp

def f(x,y):
    return 2 * x + 3 * y 

# Passing integers or tuple of integers is fine
g = jax.grad(f, argnums=0)(2.0,2.0)
g = jax.grad(f, argnums=(0,1))(2.0,2.0)

# Passing integer from a Python list works
diff_indices = [0,1]
g = jax.grad(f, argnums=diff_indices[0])(2.0,2.0)

# Passing integer from an array does not work
diff_indices = np.array([0,1])
g = jax.grad(f, argnums=diff_indices[0])(2.0,2.0)





seed = jax.random.PRNGKey(0)
inp = jax.random.uniform(seed, (5,))
grad_indices = np.array([0,0,1,0,1])

def partialf(i):
    idx = grad_indices[i]
    grad_result = jax.grad(f, argnums=idx)(inp[i],inp[i]) 
    return grad_result

#results = jax.lax.map(partialf, np.arange(5))

grad_indices = [0,0,1,0,1]
for i in range(5):
    idx = grad_indices[i]
    grad_result = jax.grad(f, grad_indices[i])(inp[i],inp[i]) 


    
#for i in range(5):
#    result = f(inp1[i], inp2[i])  
#    grad_result = jax.grad(f, grad_indices[i])(inp1[i],inp2[i]) 

#def f_and_partialf(index):
#    result = f(x,y,z)
#    
#    #grad_result = jax.grad(f,grad_idx)
#    grad_result = grad_idx
#    return result, grad_result
#
#vectorized = jax.vmap(f_and_partialf, in_axes=(0,0,0,0), out_axes=(0,0))
##vectorized = jax.lax.map(f_and_partialf, I#)
#vectorized(np.arange(5),np.arange(5),np.arange(5),np.array([0,2,1,2,0]))
#
#
