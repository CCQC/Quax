import jax
import jax.numpy as np
import numpy as onp

def f(x,y):
    return 2 * x + 3 * y 

inp = np.array([1.0,2.0,3.0,4.0,5.0])
grad_indices = np.array([0,0,1,0,1]) # this now has to be an array

def partialf(i):
    idx = grad_indices[i]
    grad_result = jax.grad(f, argnums=idx)(inp[i],inp[i]) 
    return grad_result

grad_result = jax.lax.map(partialf, np.arange(5))


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
