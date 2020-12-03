import numpy as onp
import itertools

def get_deriv_vec_idx(deriv_vec):
    """
    Used to lookup appropriate slice of disk-saved integral derivative tensor 
    which corresponds to a particular derivative vector.
    Given a derivative vector of shape NCART, 
    find the flattened generalized upper triangle index of 
    the cartesian derivative tensor. 
    """
    dim = deriv_vec.shape[0]
    vals = onp.arange(dim, dtype=int)
    deriv_order = onp.sum(deriv_vec)

    deriv_vecs = []
    for c in itertools.combinations_with_replacement(vals, deriv_order):
        tmp_deriv_vec = onp.zeros_like(deriv_vec, dtype=int)
        for i in c:
            tmp_deriv_vec[i] += 1
        deriv_vecs.append(tmp_deriv_vec)

    deriv_vecs = onp.asarray(deriv_vecs)
    idx = onp.argwhere(onp.all(deriv_vecs==deriv_vec,axis=1)).reshape(-1)[0]
    return idx

# Sum over all partitions of the set range(deriv_order)
def partition(collection):
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [[first]] + smaller

def get_required_deriv_vecs(natoms, deriv_order, address):
    """
    Simulates the Faa Di Bruno formula, giving a set of partial derivative operators which are required
    to find a particular higher order partial derivative operator, as defined by `deriv_order` and `address`.

    The returned partial derivative operators are each represented by vectors of length NCART where NCART is 3 * natom.
    The value of each index in these vectors describes how many times to differentiate wrt that particular cartesian coordinate.
    For example, a 2 atom system has atoms A,B and all derivative vectors have indices which correspond to the coordinates: [Ax, Ay, Az, Bx, By, Bz].
    A derivative vector [1,0,0,2,0,0] therefore represents the partial derivative (d^3)/(dAx dBx dBx).

    Parameters
    ----------
    natoms : int
        The number of atoms in the system. The cartesian nuclear derivative tensor for this `natom` system
        has a dimension size 3 * natom
    deriv_order : int
        The order of differentiation.  The cartesian nuclear derivative tensor for this `natom` system
        has rank `deriv_order` and dimension size 3 * natoms
    address : tuple of int
        A tuple of integers which describe which cartesian partial derivative
        we wish to compute. Each integer in the tuple is in the range [0, NCART-1]
    Returns
    -------
    partial_derivatives : arr
        An array of partial derivatives of dimensions (npartials, NCART)
    """
    address = list(address)
    deriv_vecs = []
    nparams = natoms * 3
    for p in partition(address):
        for sub in p:
            # List of zeros
            deriv_vec = [0] * nparams 
            for i in sub:
                deriv_vec[i] += 1
            deriv_vecs.append(deriv_vec)
    partial_derivatives = onp.unique(onp.asarray(deriv_vecs), axis=0)
    return partial_derivatives 


