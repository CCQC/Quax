import itertools
import numpy as np
# convert a deriv_vec to a flattened cartesian nuclear derivative index
# [0,0,2,0,0,0] ---> 0 1  2  3  4  5  ----> 11
#                      6  7  8  9 10
#                        11 12 13 14
#                        ^^ 15 16 17


#deriv_vec = np.array([0,0,2,0,0,0])
#dim = deriv_vec.shape[0]
#tup = np.repeat(np.arange(dim), deriv_vec)
# This is not right, does not locate upper triangle.
#flt_idx = np.ravel_multi_index(tup, (dim,dim))

#natoms = 2
#deriv_order = 2
#vals = np.arange(natoms * 3)
#combos = list(itertools.combinations(vals, deriv_order))

def get_deriv_vec_idx(deriv_vec):
    """
    Given a derivative vector of shape NCART, 
    find the flattened generalized upper triangle index of 
    the cartesian derivative tensor it corresponds to.
    """
    dim = deriv_vec.shape[0]
    vals = np.arange(dim, dtype=int)
    deriv_order = np.sum(deriv_vec)

    deriv_vecs = []
    for c in itertools.combinations_with_replacement(vals, deriv_order):
        tmp_deriv_vec = np.zeros_like(deriv_vec, dtype=int)
        for i in c:
            tmp_deriv_vec[i] += 1
        deriv_vecs.append(tmp_deriv_vec)

    deriv_vecs = np.asarray(deriv_vecs)
    idx = np.argwhere(np.all(deriv_vecs==deriv_vec,axis=1)).reshape(-1)[0]
    return idx

deriv_vec = np.array([0,0,2,0,0,0])
idx = get_deriv_vec_idx(deriv_vec)
print(idx)

deriv_vec = np.array([0,1,0,1,0,0])
idx = get_deriv_vec_idx(deriv_vec)
print(idx)


