import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)

def double_factorial(n):
    '''The double factorial function for small Python integer `n`.'''
    return np.prod(np.arange(n, 1, -2))

def normalize(aa,ax,ay,az):
    '''
    Normalization constant for gaussian basis function. 
    aa : orbital exponent
    ax : angular momentum component x
    ay : angular momentum component y
    az : angular momentum component z
    '''
    f = np.sqrt(double_factorial(2*ax-1) * double_factorial(2*ay-1) * double_factorial(2*az-1))
    N = (2*aa/np.pi)**(3/4) * (4 * aa)**((ax+ay+az)/2) / f
    return N

@jax.jit
def overlap_ss(A, B, alpha_bra, alpha_ket):
    ss = ((np.pi / (alpha_bra + alpha_ket))**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(A-B,A-B)) / (alpha_bra + alpha_ket)))
    return ss


def new_normalize(exps, coeffs, ax,ay,az):
    '''Normalizes a contracted or uncontracted basis function'''
    L = ax + ay + az
    # Normalization constants of primitives, in a vector
    primitive_Ns = np.sqrt(np.power(2,2*(ax+ay+az)+1.5)*
                        np.power(exps,ax+ay+az+1.5)/
                        double_factorial(2*ax-1)/double_factorial(2*ay-1)/
                        double_factorial(2*az-1)/np.power(np.pi,1.5))
    #f = np.sqrt(double_factorial(2*ax-1) * double_factorial(2*ay-1) * double_factorial(2*az-1))
    #primitive_Ns = (2*exps/np.pi)**(3/4) * (4 * exps)**((ax+ay+az)/2) / f
    
    #TODO remove redundant double factorial calls
    # Compute full basis function normalization constant 
    prefactor = (np.pi**(1.5) * double_factorial(2*ax-1) * double_factorial(2*ay-1) * double_factorial(2*az-1)) / 2**L
    K = exps.shape[0]  # Degree of contraction K
    N = 0 
    for i in range(K):
        for j in range(K):
            N += primitive_Ns[i] * primitive_Ns[j] * coeffs[i] * coeffs[j] / (exps[i] + exps[j])**(L+1.5)
    
    N *= prefactor
    N = N**-0.5
    return N

def cnorm(exponents,coeffs,ax,ay,az):
    K = exponents.shape[0]  # Degree of contraction K
    L = ax + ay + az
    prefactor = (np.pi**(1.5) * double_factorial(2*ax-1) * double_factorial(2*ay-1) * double_factorial(2*az-1)) / 2**L
    c_times_c = np.outer(coeffs,coeffs) 
    a_plus_a = np.broadcast_to(exponents, (K,K)) + np.transpose(np.broadcast_to(exponents, (K,K)), (1,0))
    N = 1 / np.sqrt(np.sum(c_times_c / (a_plus_a**(L + 1.5))))
    return N
    
def contracted_normalize(exponents,coeff,ax,ay,az):
    '''Normalization constant for a single contracted gaussian basis function'''
    K = exponents.shape[0]  # Degree of contraction K
    L = ax + ay + az        # Total angular momentum L
    # all possible combinations of ci * cj
    c_times_c = np.outer(coeff,coeff) 
    # all possible combinations of alphai + alphaj
    a_plus_a = np.broadcast_to(exponents, (K,K)) + np.transpose(np.broadcast_to(exponents, (K,K)), (1,0))
    prefactor = (np.pi**(1.5) * double_factorial(2*ax-1) * double_factorial(2*ay-1) * double_factorial(2*az-1)) / 2**L
    sum_term = np.sum(c_times_c / (a_plus_a**(L + 1.5)))
    return (prefactor * sum_term) ** -0.5


#@jax.jit
# have to do every combo, you were just doing diagonal
#def contracted_overlap_ss(A, B, alpha_bra, alpha_ket, c_bra, c_ket):
#    ss = 0
#    for i in range(c_bra.shape[0]):
#        ss += ((np.pi / (alpha_bra[i] + alpha_ket[i]))**(3/2) * np.exp((-alpha_bra[i] * alpha_ket[i] * np.dot(A-B,A-B)) / (alpha_bra[i] + alpha_ket[i]))) * c_bra[i] * c_ket[i]
#    #ss = np.sum((np.pi / (alpha_bra + alpha_ket))**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(A-B,A-B)) / (alpha_bra + alpha_ket)) * c_bra * c_ket )
#    return ss

#def contracted_overlap_ss(A, B, alpha_bra, alpha_ket, c_bra, c_ket):
#    size = alpha_bra.shape[0]
#    ss = 0
#    AB = np.dot(A-B,A-B)
#    for i in range(size):
#        for j in range(size):
#            ss += c_bra[i] * c_ket[j] * np.exp((-alpha_bra[i] * alpha_ket[j] * AB) / (alpha_bra[i] + alpha_ket[j])) * (np.pi / (alpha_bra[i] + alpha_ket[j]))**(1.5)
#            #ss += normalize(alpha_bra[i],0,0,0) * normalize(alpha_ket[i],0,0,0) * c_bra[i] * c_ket[j] * np.exp((-alpha_bra[i] * alpha_ket[j] * AB) / (alpha_bra[i] + alpha_ket[j])) * (np.pi / (alpha_bra[i] + alpha_ket[j]))**(1.5)
#            #ss += normalize(alpha_bra[i],0,0,0) * normalize(alpha_ket[i],0,0,0) * c_bra[i] * c_ket[j] * np.exp((-alpha_bra[i] * alpha_ket[j] * AB) / (alpha_bra[i] + alpha_ket[j])) * (np.pi / (alpha_bra[i] + alpha_ket[j]))**(1.5)
#    return ss

def contracted_overlap_ss(A, B, alpha_bra, alpha_ket, c_bra, c_ket):
    size = alpha_bra.shape[0]
    AB = np.dot(A-B,A-B)
    # all possible combinations of c_bra * c_ket
    c_times_c = np.outer(c_bra,c_ket) 
    # all possible combinations of alpha_bra * alpha_ket
    a_times_a = np.outer(alpha_bra,alpha_ket)
    # all possible combinations of alpha_bra + alpha_ket
    a_plus_a = np.outer(alpha_bra, np.ones_like(alpha_ket)) + np.transpose(np.outer(alpha_ket, np.ones_like(alpha_bra)))
    ss = (np.pi / a_plus_a)**(1.5) * np.exp(-a_times_a * AB / a_plus_a) * c_times_c
    return np.sum(ss)


@jax.jit
def overlap_ps_block(A, B, alpha_bra, alpha_ket):
    oot_alpha_bra = 1 / (2 * alpha_bra)
    return oot_alpha_bra * jax.jacrev(overlap_ss,0)(A,B,alpha_bra,alpha_ket)

@jax.jit
def overlap_sp_block(A, B, alpha_bra, alpha_ket): # not really needed is it?
    oot_alpha_bra = 1 / (2 * alpha_bra)
    return oot_alpha_bra * jax.jacrev(overlap_ss,1)(A,B,alpha_bra,alpha_ket)

@jax.jit
def overlap_pp_block(A, B, alpha_bra, alpha_ket):
    # We are promoting the ket, so the factor is the ket exponent
    oot_alpha_ket = 1 / (2 * alpha_ket)
    # No second term, ai is 0 since we are promoting the ket and theres no AM in the ket.
    return oot_alpha_ket * (jax.jacfwd(overlap_ps_block, 1)(A,B,alpha_bra,alpha_ket))

#@jax.jit
#def overlap_ds_block(A,B,alpha_bra,alpha_ket):
#    # We are promoting the bra a second time, factor is bra exponent
#    oot_alpha_bra = 1 / (2 * alpha_bra)
#    #                      # This is of shape (3,3) all dij combos symmetric matrix    # Thus a_i factor has to be 3x3 identity, so that only 
#    return oot_alpha_bra * (jax.jacfwd(overlap_ps_block, 0)(A,B,alpha_bra,alpha_ket) + np.eye(3) * overlap_ss(A,B,alpha_bra,alpha_ket))

@jax.jit
def overlap_ds_block(A,B,alpha_bra,alpha_ket):
    '''
    Returns a 1x6 array:
    (dxx,s) (dxy,s)  (dxz,s) (dyy,s) (dyz,s) (dzz,s) 
    '''
    # We are promoting the bra a second time, factor is bra exponent
    oot_alpha_bra = 1 / (2 * alpha_bra)
    #                      # This is of shape (3,3) all dij combos symmetric matrix    # Thus a_i factor has to be 3x3 identity, so that only 
    result = oot_alpha_bra * (jax.jacfwd(overlap_ps_block, 0)(A,B,alpha_bra,alpha_ket) + np.eye(3) * overlap_ss(A,B,alpha_bra,alpha_ket))  
    # This result is a 3x3 array containing all (dxx,s) (dxy,s) (dyx,s), only need upper or lower triangle
    # Return upper triangle ((dxx, dxy, dxz, dyy, dyz, dzz) | s) as a vector
    iu = np.triu_indices(3)
    return result[iu]

@jax.jit
def overlap_dp_block(A,B,alpha_bra,alpha_ket): 
    '''
    Returns a 1x18 array:
    (dxx,px) (dxx,py) (dxx,pz) (dxy,px) (dxy,py) (dxy,pz) (dxz,px) (dxz,py) (dxz,pz) (dyy,px) (dyy,py) (dyy,pz) (dyz,px) (dyz,py) (dyz,pz) (dzz,px) (dzz,py) (dzz,pz)
    If called directly, should reshape into a 6x3 block!
    (dxx,px) (dxx,py) (dxx,pz) 
    (dxy,px) (dxy,py) (dxy,pz) 
    (dxz,px) (dxz,py) (dxz,pz) 
    (dyy,px) (dyy,py) (dyy,pz) 
    (dyz,px) (dyz,py) (dyz,pz) 
    (dzz,px) (dzz,py) (dzz,pz)
    '''
    oot_alpha_ket = 1 / (2 * alpha_ket) # use ket, since we are promoting ket from s-->p
    # This is a 18x1 array of d by p functions. Could also use overlap_pp_block instead, i think? 
    return np.ravel(oot_alpha_ket * jax.jacfwd(overlap_ds_block, 1)(A,B,alpha_bra,alpha_ket))

@jax.jit
def overlap_dd_block(A,B,alpha_bra,alpha_ket): 
    '''
    Returns a 6x6 array:
    (dxx,dxx) (dxx,dxy) (dxx,dxz) (dxx,dyy) (dxx,dyz) (dxx,dzz)
    (dxy,dxx) (dxy,dxy) (dxy,dxz) (dxy,dyy) (dxy,dyz) (dxy,dzz)
    (dxz,dxx) (dxz,dxy) (dxz,dxz) (dxz,dyy) (dxz,dyz) (dxz,dzz)
    (dyy,dxx) (dyy,dxy) (dyy,dxz) (dyy,dyy) (dyy,dyz) (dyy,dzz)
    (dyz,dxx) (dyz,dxy) (dyz,dxz) (dyz,dyy) (dyz,dyz) (dyz,dzz)
    (dzz,dxx) (dzz,dxy) (dzz,dxz) (dzz,dyy) (dzz,dyz) (dzz,dzz)
    '''
    oot_alpha_ket = 1 / (2 * alpha_ket) # use ket, since we are promoting ket from p-->d
    # The jacfwd (first) term is an 18x3 array           # ai coeffs are   # the second term is
    # (dxx,px) --> (dxx,dxx) (dxx, dxy), (dxx, dxz)      1, 0, 0           (dxx|s) (dxx|s) (dxx|s)
    # (dxx,py) --> (dxx,dyx) (dxx, dyy), (dxx, dyz)      0, 1, 0           (dxx|s) (dxx|s) (dxx|s)
    # (dxx,pz) --> (dxx,dzx) (dxx, dzy), (dxx, dzz)      0, 0, 1           (dxx|s) (dxx|s) (dxx|s)
    # (dxy,px) --> (dxy,dxx) (dxy, dxy), (dxy, dxz)      1, 0, 0           (dxy|s) (dxy|s) (dxy|s)
    # (dxy,py) --> (dxy,dyx) (dxy, dyy), (dxy, dyz)      0, 1, 0           (dxy|s) (dxy|s) (dxy|s)
    # (dxy,pz) --> (dxy,dzx) (dxy, dzy), (dxy, dzz)      0, 0, 1           (dxy|s) (dxy|s) (dxy|s)
    # ....                                               ...              
    # (dzz,px) --> (dzz,dxx) (dzz, dxy), (dzz, dxz)      1, 0, 0           (dzz|s) (dzz|s) (dzz|s)
    # (dzz,py) --> (dzz,dyx) (dzz, dyy), (dzz, dyz)      0, 1, 0           (dzz|s) (dzz|s) (dzz|s)
    # (dzz,pz) --> (dzz,dzx) (dzz, dzy), (dzz, dzz)      0, 0, 1           (dzz|s) (dzz|s) (dzz|s)
    first_term = jax.jacfwd(overlap_dp_block, 1)(A,B,alpha_bra,alpha_ket)
    factor = np.tile(np.eye(3),(6,1))
    tmp_second_term = overlap_ds_block(A,B,alpha_bra,alpha_ket)
    second_term = factor * np.repeat(tmp_second_term, 9).reshape(18,3)

    result = oot_alpha_ket * (first_term + second_term)
    # result is of same signature as jacfwd (first) term above
    # It contains duplicates in each 3x3 sub-array (upper and lower triangle are equal)
    # reshape and grab out just upper triangle as a vector, reshape into matrix
    iu1,iu2 = np.triu_indices(3)
    result = result.reshape(6,3,3)[:,iu1,iu2].reshape(6,6)
    return result
    

#d_dB_overlap_ps


geom = np.array([[0.0,0.0,-0.849220457955],
                 [0.0,0.0, 0.849220457955]])
charge = np.array([1.0,1.0])
A = np.array([0.0,0.0,-0.849220457955])
#B = np.array([0.0,0.0,-0.849220457955])
B = np.array([0.0,0.0, 0.849220457955])
alpha_bra = 0.5
alpha_ket = 0.5

# This is a basis function
exps =   np.array([0.5,
                   0.4])
coeffs = np.array([1.0,
                   1.0])

#tmp_N = new_normalize(exps, coeffs, 0,0,0)
#print(tmp_N)
#print(tmp_N * coeffs)


#tmp_N = cnorm(exps,coeffs,0,0,0)
#print(tmp_N)
#print(tmp_N * coeffs)

tmp_N = contracted_normalize(exps, coeffs, 0, 0, 0)
#print("Normalization constant", tmp_N)
c_o = contracted_overlap_ss(A, A, exps, exps, coeffs, coeffs)
print(tmp_N * tmp_N * c_o)
c_o = contracted_overlap_ss(A, B, exps, exps, coeffs, coeffs)
print(tmp_N * tmp_N * c_o)
c_o = contracted_overlap_ss(B, A, exps, exps, coeffs, coeffs)
print(tmp_N * tmp_N * c_o)
c_o = contracted_overlap_ss(B, B, exps, exps, coeffs, coeffs)
print(tmp_N * tmp_N * c_o)


#0.21238156276178832 *0.17965292907913089



#print("Raw normalization constant")
#print(tmp_N)
#print("normalization constant times coefficients")
#print(tmp_N * coeffs)

#print("Raw overlap")
#print(c_o)
#print("normalized overlap")
#print(tmp_N * tmp_N * c_o)
#print(tmp_N * c_o)

s_N = 0.4237772081237576
p_N = 0.5993114751532237
d_N = 0.489335770373359


## (s|s)
#print(s_N * s_N * overlap_ss(A,B,alpha_bra,alpha_ket))       # YUP
## (p|s)
#print(p_N * s_N * overlap_ps_block(A,B,alpha_bra,alpha_ket)) # YUP
## (p|p)
#print(p_N * p_N * overlap_pp_block(A,B,alpha_bra,alpha_ket)) # YUP
## (d|s)
#print(d_N * s_N * overlap_ds_block(A,B,alpha_bra,alpha_ket)) # YUP
## (d|p)
#print(d_N * p_N * overlap_dp_block(A,B,alpha_bra,alpha_ket).reshape(6,3))  # YUP
## (d|d)
#print(d_N * d_N * overlap_dd_block(A,B,alpha_bra,alpha_ket))


#print('hard coded')
#print(overlap_ps_block(A,B,alpha_bra,alpha_ket))

#print('hard coded')
#print(overlap_pp_block(A,B,alpha_bra,alpha_ket))

#print('hard coded')
#print(overlap_ds_block(A,B,alpha_bra,alpha_ket))


#overlap_dp_block(A,B,alpha_bra,alpha_ket)

#dd_block = overlap_dd_block(A,B,alpha_bra,alpha_ket)
#print(dd_block * 0.489335770373359)
#for i in range(1000):
#    overlap_pp_block(A,B,alpha_bra,alpha_ket)
