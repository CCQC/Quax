import jax
import jax.numpy as np
import numpy as onp


@jax.jit
def cartesian_product(*arrays):
    '''JAX-friendly version of cartesian product. Same order as other function, more memory requirements though.'''
    tmp = np.asarray(np.meshgrid(*arrays, indexing='ij')).reshape(len(arrays),-1).T
    #tmp = np.meshgrid(*arrays, indexing='ij')
    return np.array(tmp)

def old_cartesian_product(*arrays):
    '''Generalized cartesian product of any number of arrays'''
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)



a = np.arange(2)
res1 = cartesian_product(a,a,a,a)
b = onp.arange(2)
res2 = old_cartesian_product(b,b,b,b)
print(res1.shape)
print(res2.shape)

print(onp.allclose(a,b))

