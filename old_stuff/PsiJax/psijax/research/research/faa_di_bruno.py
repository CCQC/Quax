# The goal here is given a partial derivative order and address (or really just the address)
# find all required deriv_vecs according to Fa Di Bruno
# can verify by printing the deriv_vec in the JVP rules.
import numpy as np

# 3rd derivative, once wrt x0, y0, z0 each
#address = [0,1,2]
address = [2,3,4,5]
deriv_order = len(address)
natoms = 2
nparams = natoms * 3

# What we want: a set of unique derivative vectors 

# In this case:
# [1,1,1,0,0,0]   d3y/dx1dx2dx3
# [1,0,0,0,0,0]   dy/dx1
# [0,1,1,0,0,0]  ...
# [0,1,0,0,0,0]
# [1,0,1,0,0,0]
# [0,0,1,0,0,0]
# [1,1,0,0,0,0]
# [1,0,0,0,0,0]
# [0,1,0,0,0,0]
# [0,0,1,0,0,0]

# However, there are duplicates!
# These should be the only ones JAX ever requests
#[1, 1, 1, 0, 0, 0]
#[0, 1, 1, 0, 0, 0]
#[1, 1, 0, 0, 0, 0]
#[1, 0, 1, 0, 0, 0]
#[1, 0, 0, 0, 0, 0]
#[0, 1, 0, 0, 0, 0]
#[0, 0, 1, 0, 0, 0]


# Sum over all partitions of the set arange(deriv_order)
    # find cardinality of the partition
    # 

def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return
    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller


#for i in partition(address):
#    print(i)

# Don't you jsut want to add one to the index provided in partition(address)?
deriv_vecs = []

for p in partition(address):
    for sub in p:
        # List of zeros
        deriv_vec = [0] * nparams 
        for i in sub:
            deriv_vec[i] += 1
            #deriv_vecs.append(deriv_vec)
        deriv_vecs.append(deriv_vec)

# Above are not unique, so use np.unique
#final = np.unique(np.asarray(deriv_vecs), axis=0)
#print(final)

# Given a derivative order, a derivative tensor address (which derivative operator inthe NCART x NCART ... x NCART array?
def get_required_deriv_vecs(natoms, deriv_order, address):
    """
    Simulates the Faa Di Bruno formula, giving a set of partial derivative operators which are required
    to find a particular higher order partial derivative operator, as defined by `deriv_order` and `address`.

    The returned partial derivative operators are each represented by vectors of length NCART where NCART is 3 * natom.
    The value of each index in these vectors describes how many times to differentiate wrt that particular cartesian coordinate.
    For example, 2 atom system atoms A,B have a derivative vector where the indices represent [Ax, Ay, Az, Bx, By, Bz].
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
    if len(address) != deriv_order:
        raise Exception("Deriv order and address do not match")
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
    print(len(deriv_vecs))
    partial_derivatives = np.unique(np.asarray(deriv_vecs), axis=0)
    return partial_derivatives 

#res = get_required_deriv_vecs(3, [3,4,5], 2)
#print(res)
#res = get_required_deriv_vecs(2, 3, [3,4,5])
#print(res)

res = get_required_deriv_vecs(4, 1, (0,       ))
print(res.shape)
print(res)
res = get_required_deriv_vecs(4, 2, (0,1      ))
print(res.shape)
print(res)
res = get_required_deriv_vecs(4, 3, (0,1,2    ))
print(res.shape)
print(res)
res = get_required_deriv_vecs(4, 4, (0,1,2,3  ))
print(res.shape)
print(res)
res = get_required_deriv_vecs(4, 5, (0,1,2,3,4))
print(res.shape)
print(res)
res = get_required_deriv_vecs(4, 6, (0,1,2,3,4,5))
#print(res)
print(res.shape)
print(res)

#import itertools
#print(list(itertools.combinations([0,1,2,3,4,5], 2)))

