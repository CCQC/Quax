import numpy as np

# THIS IS THE BEST.
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

def dstack_product(*arrays):
    return np.dstack(
        np.meshgrid(*arrays, indexing='ij')
        ).reshape(-1, len(arrays))

def cartesian_product2(*arrays):
    ndim = len(arrays)
    return np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim)


def find_indices(indices):
    '''Find a set of indices of ERI tensor corresponding to unique two-electron integrals'''
    cond1 = (indices[:,0] >= indices[:,1]) & (indices[:,2] >= indices[:,3])
    cond2 = indices[:,0] * (indices[:,0] + 1)/2 + indices[:,1] >= indices[:,2]*(indices[:,2]+1)/2 + indices[:,3]
    mask = cond1 & cond2
    return indices[mask,:]


a = np.arange(140, dtype=np.int8)
#indices1 = cartesian_product(a,a,a,a)
indices2 = cartesian_product2(a,a,a,a)


#unique1 = find_indices(indices1)
#unique2 = find_indices(indices2)


#indices2 = np.dstack(np.meshgrid(a,a,a,a)).reshape(-1,4)
#
#m = np.meshgrid(a,a,a,a)
#indices3 = np.vstack((m[0].flatten(), m[1].flatten(), m[2].flatten(), m[3].flatten())).reshape(-1,4)
#
##print(indices1)
##print(indices2)
#
#print(np.allclose(indices1,indices2))
#print(np.allclose(indices1,indices3))
#print(np.allclose(indices2,indices3))
#
#def find_indices(indices):
#    '''Find a set of indices of ERI tensor corresponding to unique two-electron integrals'''
#    cond1 = indices[:,0] >= indices[:,1]
#    cond2 = indices[:,2] >= indices[:,3]
#    cond3 = indices[:,0] * (indices[:,0] + 1)/2 + indices[:,1] >= indices[:,2]*(indices[:,2]+1)/2 + indices[:,3]
#    mask = cond1 & cond2 & cond3
#    unique = np.asarray(indices[mask,:])
#    return unique
#
#print(find_indices(indices1))
#print(find_indices(indices2))
#print(find_indices(indices3))
#

