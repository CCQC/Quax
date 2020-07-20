import jax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=500)

def cartesian_product(*arrays):
    '''Generalized cartesian product of any number of arrays'''
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


# Function definitions. We always return a vector of all primitive values.
# NOTES:
# The base function just computes a single primitive. 
# The vectorized versions can compute many primitives with the same centers at the same time.
# All functions return a vector of primitives, the number of which is dependent on the angular momentum
# (s|p) creates 3 primitives. (p|p) creates 9 (redundant for now)

# investigate shapes of each function output
A = np.array([0.0, 0.0, -0.849220457955])
B = np.array([0.0, 0.0,  0.849220457955])
alpha_bra = 0.5 
alpha_ket = 0.4 
c1 = 1
c2 = 1

@jax.jit
def overlap_ss(A, B, aa, bb, c1=1, c2=1):
    ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * np.dot(A-B, A-B)) / (aa + bb)))
    return np.array([ss * c1 * c2])

@jax.jit
def overlap_ps(A, B, alpha_bra, alpha_ket,c1,c2):
    oot_alpha_bra = 1 / (2 * alpha_bra)
    first = jax.jacrev(overlap_ss,0)(A,B,alpha_bra,alpha_ket,c1,c2).reshape(3,1)
    print("ps shape")
    print(first.shape)
    return (oot_alpha_bra * first).reshape(-1)

def overlap_ds(A, B, alpha_bra, alpha_ket,c1,c2):
    '''
    Returns (dxx,s) (dxy,s) (dxz,s) (dyy,s) (dyz,s) (dzz,s) 
    '''
    oot_alpha_bra = 1 / (2 * alpha_bra)
    first_term = jax.jacfwd(overlap_ps, 0)(A,B,alpha_bra,alpha_ket,c1,c2).reshape(9,1)
    #print(jax.jacfwd(overlap_ps, 0)(A,B,alpha_bra,alpha_ket,c1,c2).shape)
    print("ds shape")
    print(first_term.shape)
    unique_mask = np.arange(9).reshape(3,3)[np.triu_indices(3)]
    first_term = first_term[unique_mask].reshape(-1)
    ai = np.array([[1,0,0,1,0,1]])
    second_term = (ai  * overlap_ss(A,B,alpha_bra,alpha_ket,c1,c2)).reshape(-1)
    print('SECOND',second_term)
    return oot_alpha_bra * (first_term + second_term)


def overlap_fs(A, B, alpha_bra, alpha_ket,c1,c2):
    '''
    Differentiation spawns
    1    2    3            4    5                 6            7    8                9                10
    fxxx fxxy fxxz    fxxy fxyy fxyz    fxxz fxyz fxzz    fxyy fyyy fyyz   fxyz fyyz fyzz   fxzz fyzz fzzz

    0    1    2            4    5                 8            10   11               14               17

    --------------------------------------------------    -------------------------------          -----
    3x3 upper triangle                                     lower 3 elements of 3x3 upper tri       last element of 3x3 upper tri


    a = np.arange(9).reshape(3,3)[np.triu_indices(3)]
    0  1  2
       4  5
          8

    b = a[3:] + 6
    10 11
       14

    c = b[2:] + 3
    17

*** 0   1   2  
       +3  +3  
 **     4   5  
       +3  +3  
  *         8  
       +3  +3
 **    10  11
           +3
  *        14
           +3
  *        17

    Returns (fxx,s) (dxy,s) (dxz,s) (dyy,s) (dyz,s) (dzz,s) 

    From a 3x3x3 array, we want the generalized lower triangle:
    [0,0,0]
    [0,0,1]
    [0,0,2]
    [0,1,1]
    [0,1,2]
    [0,2,2]
    [1,1,1]
    [1,1,2]
    [1,2,2]
    [2,2,2]

    eff, nevermind its these indices, what are these even??:
    [0,0,0]
    [0,0,1]
    [0,0,2]
    [0,1,1]
    [0,1,2]
    [0,2,2]
    [1,0,1]
    [1,0,2]
    [1,1,2]
    [1,2,2]


    np.arange(27).reshape(3,3,3)[generalized lower triangle indices]
    will tell you which (f|s)'s to pull out of the vector. 


    '''
    oot_alpha_bra = 1 / (2 * alpha_bra)
    first_term = jax.jacfwd(overlap_ds, 0)(A,B,alpha_bra,alpha_ket,c1,c2).reshape(18,1)
    print("fs shape")
    print(first_term.shape)
    #unique_mask = np.arange(9).reshape(3,3)[np.triu_indices(3)]
    #first_term = first_term[unique_mask].reshape(-1)
    #ai = np.array([[1,0,0,1,0,1]])
    #second_term = (ai  * overlap_ss(A,B,alpha_bra,alpha_ket,c1,c2)).reshape(-1)
    #return oot_alpha_bra * (first_term + second_term)



print(overlap_ss(A,B,alpha_bra,alpha_ket,c1,c2))
print(overlap_ps(A,B,alpha_bra,alpha_ket,c1,c2))
print(overlap_ds(A,B,alpha_bra,alpha_ket,c1,c2))
#overlap_ds(A,B,alpha_bra,alpha_ket,c1,c2)
#overlap_fs(A,B,alpha_bra,alpha_ket,c1,c2)


#print(overlap_ps(A,B,alpha_bra,alpha_ket,c1,c2))
#print(overlap_ds(A,B,alpha_bra,alpha_ket,c1,c2))
#print(overlap_fs(A,B,alpha_bra,alpha_ket,c1,c2))

#@jax.jit
#def overlap_ps(A, B, alpha_bra, alpha_ket,c1,c2):
#    oot_alpha_bra = 1 / (2 * alpha_bra)
#    return (oot_alpha_bra * jax.jacrev(overlap_ss,0)(A,B,alpha_bra,alpha_ket,c1,c2)).reshape(-1)

def overlap_sp(A, B, alpha_bra, alpha_ket,c1,c2):
    return overlap_ps(B, A, alpha_ket, alpha_bra,c2,c1)

@jax.jit
def overlap_pp(A, B, alpha_bra, alpha_ket,c1,c2):
    '''Returns 1d vector (px|px) (px|py) (px|pz) (py|py) (py|pz) (pz|pz)'''
    oot_alpha_ket = 1 / (2 * alpha_ket)
    result = oot_alpha_ket * (jax.jacfwd(overlap_ps, 1)(A,B,alpha_bra,alpha_ket,c1,c2))
    return result[np.triu_indices(3)]

@jax.jit
def overlap_ds(A,B,alpha_bra,alpha_ket,c1,c2):
    '''
    Returns (dxx,s) (dxy,s)  (dxz,s) (dyy,s) (dyz,s) (dzz,s) 
    '''
    oot_alpha_bra = 1 / (2 * alpha_bra)
    result = oot_alpha_bra * (jax.jacfwd(overlap_ps, 0)(A,B,alpha_bra,alpha_ket,c1,c2) + np.eye(3) * overlap_ss(A,B,alpha_bra,alpha_ket,c1,c2))  
    iu = np.triu_indices(3)
    return result[iu]

def new_overlap_ds(A,B,alpha_bra,alpha_ket,c1,c2):
    '''
    Returns (dxx,s) (dxy,s)  (dxz,s) (dyy,s) (dyz,s) (dzz,s) 
    '''
    oot_alpha_bra = 1 / (2 * alpha_bra)
    first_term = jax.jacfwd(overlap_ps, 0)(A,B,alpha_bra,alpha_ket,c1,c2).reshape(9,1)
    print(first_term)
    unique_mask = np.arange(9).reshape(3,3)[np.triu_indices(3)]
    first_term = first_term[unique_mask].reshape(-1)
    ai = np.array([[1,0,0,1,0,1]])
    second_term = (ai  * overlap_ss(A,B,alpha_bra,alpha_ket,c1,c2)).reshape(-1)
    return oot_alpha_bra * (first_term + second_term)


@jax.jit
def overlap_sd(A,B,alpha_bra,alpha_ket,c1,c2):
    return overlap_ds(B,A,alpha_ket,alpha_bra,c2,c1)

@jax.jit
def overlap_dp(A,B,alpha_bra,alpha_ket,c1,c2): 
    '''
    Returns a 1x18 array:
    (dxx,px) (dxx,py) (dxx,pz) (dxy,px) (dxy,py) (dxy,pz) (dxz,px) (dxz,py) (dxz,pz) (dyy,px) (dyy,py) (dyy,pz) (dyz,px) (dyz,py) (dyz,pz) (dzz,px) (dzz,py) (dzz,pz)
    '''
    oot_alpha_ket = 1 / (2 * alpha_ket) # use ket, since we are promoting ket from s-->p
    # This is a 18x1 array of d by p functions. Could also use overlap_pp_block instead, i think? 
    return (oot_alpha_ket * jax.jacfwd(overlap_ds, 1)(A,B,alpha_bra,alpha_ket,c1,c2)).reshape(-1)

@jax.jit
def overlap_pd(A,B,alpha_bra,alpha_ket,c1,c2):
    return overlap_dp(B,A,alpha_ket,alpha_bra,c2,c1)

def overlap_dd(A,B,alpha_bra,alpha_ket,c1,c2): 
    '''
    Returns flattened 6x6 array:
    (dxx,dxx) (dxx,dxy) (dxx,dxz) (dxx,dyy) (dxx,dyz) (dxx,dzz)
    (dxy,dxx) (dxy,dxy) (dxy,dxz) (dxy,dyy) (dxy,dyz) (dxy,dzz)
    (dxz,dxx) (dxz,dxy) (dxz,dxz) (dxz,dyy) (dxz,dyz) (dxz,dzz)
    (dyy,dxx) (dyy,dxy) (dyy,dxz) (dyy,dyy) (dyy,dyz) (dyy,dzz)
    (dyz,dxx) (dyz,dxy) (dyz,dxz) (dyz,dyy) (dyz,dyz) (dyz,dzz)
    (dzz,dxx) (dzz,dxy) (dzz,dxz) (dzz,dyy) (dzz,dyz) (dzz,dzz)



    This has 21 unique elements along an upper triangle:
    np.arange(36).reshape(6,6)[np.triu_indices(6,0,6)].shape

    '''
    oot_alpha_ket = 1 / (2 * alpha_ket) # use ket, since we are promoting ket from p-->d

    # Create array structure (dxx, d*), 
    #                        (dxy, d*),
    first_term = jax.jacfwd(overlap_dp, 1)(A,B,alpha_bra,alpha_ket,c1,c2).reshape(6,9)
    # Create vector which pulls out unique (d*|d*) from a flattened 3x3  (1x9 block)
    unique_mask = np.arange(9).reshape(3,3)[np.triu_indices(3)]
    first_term = first_term[:,unique_mask]
    # only want to add second term when bra and ket have dupicate subindices (dii|djj) 
    ai = np.outer(np.array([1,0,0,1,0,1]),np.array([1,0,0,1,0,1]))
    #ai = np.array([[1,0,0,1,0,1],
    #               [0,0,0,0,0,0],
    #               [0,0,0,0,0,0],
    #               [1,0,0,1,0,1],
    #               [0,0,0,0,0,0],
    #               [1,0,0,1,0,1]])
    #second_term = ai * overlap_ds(A,B,alpha_bra,alpha_ket,c1,c2)
    second_term = (ai * overlap_ds(A,B,alpha_bra,alpha_ket,c1,c2)).T
    result = oot_alpha_ket * (first_term + second_term)
    # Take upper triangle only
    return result[np.triu_indices(6)].reshape(-1)
    #return result


def overlap_fd(A,B,alpha_bra,alpha_ket,c1,c2): 
    ''' 
    Should simplify to a 10x6 array
    (fxxx|dxx) (fxxx|dxy) ... (fxxx|dzz)
    (fxxy|dxx) (fxxy|dxy) ... (fxxy|dzz)
    ...        ...        ... ...
    (fzzz|dxx) (fzzz|dxy) ... (fzzz|dzz)

    This has 21 unique elements along an upper triangle:
    np.arange(60).reshape(10,6)[np.triu_indices(10,0,6)]
    # np.arange(orderbra * orderket).reshape(orderbra,orderket)[np.triu_indices(orderbra,0,orderket)]


    # Issue: unique values are haphazardly placed in jacfwd if overlap_dd returns flattened values

                  1    
    (dxx,dxx) --> (fxxx,dxx) 
    (dxx,dxy) --> (fxxx,dxy) 
    (dxx,dxz) --> (fxxx,dxz)
    (dxx,dyy) --> (fxxx,dyy)
    (dxx,dyz) --> (fxxx,dyz)
    (dxx,dzz) --> (fxxx,dzz)

                  2
    (dxy,dxx) --> (fxxy,dxx) 
    (dxy,dxy) --> (fxxy,dxx)
    (dxy,dxz) --> (fxxy,dxz)
    (dxy,dyy) --> (fxxy,dyy)
    (dxy,dyz) --> (fxxy,dyz)
    (dxy,dzz) --> (fxxy,dzz)

                  3
    (dxz,dxx) --> (fxxz,dxx) 
    (dxz,dxy) --> (fxxz,dxx)
    (dxz,dxz) --> (fxxz,dxz)
    (dxz,dyy) --> (fxxz,dyy)
    (dxz,dyz) --> (fxxz,dyz)
    (dxz,dzz) --> (fxxz,dzz)

                  4          7          8
    (dyy,dxx) --> (fxyy,dxx) (fyyy,dxx) (fyyz,dxx)
    (dyy,dxy) --> (fxyy,dxx) (fyyy,dxx) (fyyz,dxx)
    (dyy,dxz) --> (fxyy,dxz) (fyyy,dxz) (fyyz,dxz)
    (dyy,dyy) --> (fxyy,dyy) (fyyy,dyy) (fyyz,dyy)
    (dyy,dyz) --> (fxyy,dyz) (fyyy,dyz) (fyyz,dyz)
    (dyy,dzz) --> (fxyy,dzz) (fyyy,dzz) (fyyz,dzz)

                  5          9
    (dyz,dxx) --> (fxyz,dxx) (fyzz,dxx) 
    (dyz,dxy) --> (fxyz,dxx) (fyzz,dxx)  
    (dyz,dxz) --> (fxyz,dxz) (fyzz,dxz)  
    (dyz,dyy) --> (fxyz,dyy) (fyzz,dyy)  
    (dyz,dyz) --> (fxyz,dyz) (fyzz,dyz)  
    (dyz,dzz) --> (fxyz,dzz) (fyzz,dzz)

                  6                    10
    (dzz,dxx) --> (fxzz,dxx)           (fzzz,dxx)
    (dzz,dxy) --> (fxzz,dxx)           (fzzz,dxx)
    (dzz,dxz) --> (fxzz,dxz)           (fzzz,dxz)
    (dzz,dyy) --> (fxzz,dyy)           (fzzz,dyy)
    (dzz,dyz) --> (fxzz,dyz)           (fzzz,dyz)
    (dzz,dzz) --> (fxzz,dzz)           (fzzz,dzz)


    # if overlap_dd returns block values:
    list of 3 2d arrays, each being a differentiation w.r.t x:
1   (fxxx,dxx) (fxxx,dxy) (fxxx,dxz) (fxxx,dyy) (fxxx,dyz) (fxxx,dzz)
2   (fxxy,dxx) (fxxy,dxy) (fxxy,dxz) (fxxy,dyy) (fxxy,dyz) (fxxy,dzz)
3   (fxxz,dxx) (fxxz,dxy) (fxxz,dxz) (fxxz,dyy) (fxxz,dyz) (fxxz,dzz)
4   (fxyy,dxx) (fxyy,dxy) (fxyy,dxz) (fxyy,dyy) (fxyy,dyz) (fxyy,dzz)
5   (fxyz,dxx) (fxyz,dxy) (fxyz,dxz) (fxyz,dyy) (fxyz,dyz) (fxyz,dzz)
6   (fxzz,dxx) (fxzz,dxy) (fxzz,dxz) (fxzz,dyy) (fxzz,dyz) (fxzz,dzz)

    (fxxy,dxx) (fxxy,dxy) (fxxy,dxz) (fxxy,dyy) (fxxy,dyz) (fxxy,dzz)
    (fxyy,dxx) (fxyy,dxy) (fxyy,dxz) (fxyy,dyy) (fxyy,dyz) (fxyy,dzz)
    (fxyz,dxx) (fxyz,dxy) (fxyz,dxz) (fxyz,dyy) (fxyz,dyz) (fxyz,dzz)
7   (fyyy,dxx) (fyyy,dxy) (fyyy,dxz) (fyyy,dyy) (fyyy,dyz) (fyyy,dzz)
8   (fyyz,dxx) (fyyz,dxy) (fyyz,dxz) (fyyz,dyy) (fyyz,dyz) (fyyz,dzz)
9   (fyzz,dxx) (fyzz,dxy) (fyzz,dxz) (fyzz,dyy) (fyzz,dyz) (fyzz,dzz)

    (fxxz,dxx) (fxxz,dxy) (fxxz,dxz) (fxxz,dyy) (fxxz,dyz) (fxxz,dzz)
    (fxyz,dxx) (fxyz,dxy) (fxyz,dxz) (fxyz,dyy) (fxyz,dyz) (fxyz,dzz)
    (fxzz,dxx) (fxzz,dxy) (fxzz,dxz) (fxzz,dyy) (fxzz,dyz) (fxzz,dzz)
    (fyyz,dxx) (fyyz,dxy) (fyyz,dxz) (fyyz,dyy) (fyyz,dyz) (fyyz,dzz)
    (fyzz,dxx) (fyzz,dxy) (fyzz,dxz) (fyzz,dyy) (fyzz,dyz) (fyzz,dzz)
10  (fzzz,dxx) (fzzz,dxy) (fzzz,dxz) (fzzz,dyy) (fzzz,dyz) (fzzz,dzz)


If given vector of 21 unique (d|d)'s
    (dxx,dxx)->   (fxxx,dxx)    (fxxy,dxx)    (fxxz,dxx)
    (dxx,dxy)->   (fxxx,dxy)   *(fxxy,dxy)    (fxxz,dxy)
    (dxx,dxz)->   (fxxx,dxz)   *(fxxy,dxz)   -(fxxz,dxz)
    (dxx,dyy)->   (fxxx,dyy)   *(fxxy,dyy)   -(fxxz,dyy)
    (dxx,dyz)->   (fxxx,dyz)   *(fxxy,dyz)   -(fxxz,dyz)
    (dxx,dzz)->   (fxxx,dzz)   *(fxxy,dzz)   -(fxxz,dzz)

    (dxy,dxy)->  *(fxxy,dxy)    (fxyy,dxy)    (fxyz,dxy)
    (dxy,dxz)->  *(fxxy,dxz)    (fxyy,dxz)    (fxyz,dxz)
    (dxy,dyy)->  *(fxxy,dyy)    (fxyy,dyy)    (fxyz,dyy)
    (dxy,dyz)->  *(fxxy,dyz)    (fxyy,dyz)    (fxyz,dyz)
    (dxy,dzz)->  *(fxxy,dzz)    (fxyy,dzz)    (fxyz,dzz)

    (dxz,dxz)->  -(fxxz,dxz)    (fxyz,dxz)    (fxzz,dxz)
    (dxz,dyy)->  -(fxxz,dyy)    (fxyz,dyy)    (fxzz,dyy)
    (dxz,dyz)->  -(fxxz,dyz)    (fxyz,dyz)    (fxzz,dyz)
    (dxz,dzz)->  -(fxxz,dzz)    (fxyz,dzz)    (fxzz,dzz)

    (dyy,dyy)->  $(fxyy,dyy)    (fyyy,dyy)    (fyyz,dyy)
    (dyy,dyz)->  $(fxyy,dyz)    (fyyy,dyz)    (fyyz,dyz)
    (dyy,dzz)->  $(fxyy,dzz)    (fyyy,dzz)    (fyyz,dzz)

    (dyz,dyz)->   (fxyz,dyz)    (fyyz,dyz)    (fyzz,dyz)
    (dyz,dzz)->   (fxyz,dzz)    (fyyz,dzz)    (fyzz,dzz)

    (dzz,dzz)->   (fxzz,dzz)    (fyzz,dzz)    (fzzz,dzz)

    Alternative look:

    # uniques:
    (dxx, dxx) (dxx,dxy)  (dxx,dxz)  ... (dxy,dxy)   ... (dyz,dyz)   (dyz,dzz)   (dzz,dzz)
        |          |          |      ...     |               |           |           |
        v          v          v      ...     v               v           v           v
    (fxxx,dxx) (fxxx,dxy) (fxxx,dxz) ... (fxxy,dxy)  ... (fxyz,dyz)  (fxyz,dyz)  (fxzz,dyz)
3   (fxxy,dxx) (fxxy,dxy) (fxxy,dxz) ... (fxxy,dxy)  ... (fyyz,dyz)  (fyyz,dyz)  (fyzz,dyz)
    (fxxz,dxx) (fxxz,dxy) (fxxz,dxz) ... (fxxz,dxy)  ... (fyzz,dyz)  (fyzz,dyz)  (fzzz,dyz)

                                21
    '''
    oot_alpha_bra = 1 / (2 * alpha_bra)
    first_term = jax.jacfwd(overlap_dd, 0)(A,B,alpha_bra,alpha_ket,c1,c2)
    print(first_term.shape)
    print(first_term)
    print(first_term)
    first_term = jax.jacfwd(overlap_dd, 0)(A,B,alpha_bra,alpha_ket,c1,c2).T.reshape(-1)
    mask = np.arange(60).reshape(10,6)[np.triu_indices(10,0,6)]
    print(first_term[mask])
 


# investigate shapes of each function output
A = np.array([0.0, 0.0, -0.849220457955])
B = np.array([0.0, 0.0,  0.849220457955])
alpha_bra = 0.5 
alpha_ket = 0.4 
c1 = 1
c2 = 1

#print(overlap_dd(A,B,alpha_bra,alpha_ket,c1,c2).shape)
#overlap_fd(A,B,alpha_bra,alpha_ket,c1,c2)

#print(new_overlap_ds(A,B,alpha_bra,alpha_ket,c1,c2))

#print(overlap_dd(A,B,alpha_bra,alpha_ket,c1,c2).shape)


# Vectorized versions of overlap functions, packed into a dictionary 
# Can be passed vectors of alpha_bra, alpha_ket, c1, c2
#overlap_funcs = {}
#overlap_funcs['ss'] = jax.jit(jax.vmap(overlap_ss, (None,None,0,0,0,0)))
#overlap_funcs['ps'] = jax.jit(jax.vmap(overlap_ps, (None,None,0,0,0,0)))
#overlap_funcs['sp'] = jax.jit(jax.vmap(overlap_sp, (None,None,0,0,0,0)))
#overlap_funcs['pp'] = jax.jit(jax.vmap(overlap_pp, (None,None,0,0,0,0)))
#overlap_funcs['ds'] = jax.jit(jax.vmap(overlap_ds, (None,None,0,0,0,0)))
#overlap_funcs['sd'] = jax.jit(jax.vmap(overlap_sd, (None,None,0,0,0,0)))
#overlap_funcs['dp'] = jax.jit(jax.vmap(overlap_dp, (None,None,0,0,0,0)))
#overlap_funcs['pd'] = jax.jit(jax.vmap(overlap_pd, (None,None,0,0,0,0)))
#overlap_funcs['dd'] = jax.jit(jax.vmap(overlap_dd, (None,None,0,0,0,0)))

#from basis import basis_dict,geom,basis_set
#nbf = basis_set.nbf()
#nshells = len(basis_dict)


def compute_overlap(geom,basis):
    nshells = len(basis_dict)
    S = np.zeros((nbf,nbf))

    for i in range(nshells):
        for j in range(nshells):
            # Load data for this contracted integral
            c1, c2 = basis_dict[i]['coef'], basis_dict[j]['coef']
            exp1, exp2 =  basis_dict[i]['exp'], basis_dict[j]['exp']
            atom1, atom2 = basis_dict[i]['atom'], basis_dict[j]['atom']
            row_idx, col_idx = basis_dict[i]['idx'], basis_dict[j]['idx']
            row_idx_stride, col_idx_stride = basis_dict[i]['idx_stride'], basis_dict[j]['idx_stride']
            A, B = geom[atom1], geom[atom2]
        
            # Function identifier
            lookup = basis_dict[i]['am'] +  basis_dict[j]['am']
    
            # Expand exponent and coefficient data to compute all primitive combinations with vectorized functions
            exp_combos = cartesian_product(exp1,exp2)
            coeff_combos = cartesian_product(c1,c2)
            primitives = overlap_funcs[lookup](A,B,exp_combos[:,0],exp_combos[:,1],coeff_combos[:,0],coeff_combos[:,1])
            result = np.sum(primitives, axis=0)
            print(lookup)
            print(result)
    
            row_indices = np.repeat(row_idx, row_idx_stride)+ np.arange(row_idx_stride)
            col_indices = np.repeat(col_idx, col_idx_stride)+ np.arange(col_idx_stride)
            indices = cartesian_product(row_indices,col_indices)
            S = jax.ops.index_update(S, (indices[:,0],indices[:,1]), result)
    return S

#S = compute_overlap(geom, basis_dict)
#print(S)
