import jax
import jax.numpy as np
import numpy as onp

def cartesian_product(*arrays):
    '''Generalized cartesian product
       Used to find all *indices* of values in an ERI tensor of size (nbf,nbf,nbf,nbf) 
       given 4 arrays:
       (np.arange(nbf), np.arange(nbf), np.arange(nbf), np.arange(nbf))'''
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


#angular_momentum = np.array([0,0,0]) 
angular_momentum = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]) 
print(angular_momentum)
print((angular_momentum.T))


c = cartesian_product(angular_momentum,angular_momentum)
print(c)

