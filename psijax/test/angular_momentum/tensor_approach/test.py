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

@jax.jit
def overlap_ps_block(A, B, alpha_bra, alpha_ket):
    oot_alpha = 1 / (2 * alpha_bra)
    return oot_alpha * jax.jacrev(overlap_ss,0)(A,B,alpha_bra,alpha_ket)

@jax.jit
def overlap_sp_block(A, B, alpha_bra, alpha_ket): # not really needed
    oot_alpha = 1 / (2 * alpha_bra)
    return oot_alpha * jax.jacrev(overlap_ss,1)(A,B,alpha_bra,alpha_ket)

@jax.jit
def overlap_pp_block(A, B, alpha_bra, alpha_ket):
    oot_alpha_bra = 1 / (2 * alpha_bra)
    oot_alpha_ket = 1 / (2 * alpha_ket)
    # No second term, ai is 0 since we are promoting the ket and theres no AM in the ket.
    return oot_alpha_ket * (jax.jacfwd(overlap_ps_block, 1)(A,B,alpha_bra,alpha_ket))

def overlap_ds_block(A,B,alpha_bra,alpha_ket):
    oot_alpha = 1 / (2 * alpha_bra)
    
    overlap_ps_block, 0)
     



geom = np.array([[0.0,0.0,-0.849220457955],
                 [0.0,0.0, 0.849220457955]])
charge = np.array([1.0,1.0])
A = np.array([0.0,0.0,-0.849220457955])
B = np.array([0.0,0.0, 0.849220457955])
alpha_bra = 0.5
alpha_ket = 0.5

print('hard coded')
print(overlap_ps_block(A,B,alpha_bra,alpha_ket))

print('hard coded')
print(overlap_pp_block(A,B,alpha_bra,alpha_ket))

for i in range(1000):
    overlap_pp_block(A,B,alpha_bra,alpha_ket)

#print(overlap_pp_block(A,B,alpha_bra,alpha_ket))
#print(overlap_pp_block(A,B,alpha_bra,alpha_ket))
#print(overlap_pp_block(A,B,alpha_bra,alpha_ket))
#print(overlap_pp_block(A,B,alpha_bra,alpha_ket))
#print(overlap_pp_block(A,B,alpha_bra,alpha_ket))
#print(overlap_pp_block(A,B,alpha_bra,alpha_ket))
#print(overlap_pp_block(A,B,alpha_bra,alpha_ket))
#print(overlap_pp_block(A,B,alpha_bra,alpha_ket))
#print(overlap_pp_block(A,B,alpha_bra,alpha_ket))
#

## S S test
#exponents = np.repeat(0.5, 2)
#nbf_per_atom = np.array([1,1])
#angular_momentum = np.array([[0,0,0], [0,0,0]])

## S P test
#exponents = np.repeat(0.5, 4)
#nbf_per_atom = np.array([1,3])
#angular_momentum = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])

## P P test 
#exponents = np.repeat(0.5, 6)
#nbf_per_atom = np.array([3,3])
#angular_momentum = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,0],[0,1,0],[0,0,1]])

## S P D test
#exponents = np.repeat(0.5, 10)
#nbf_per_atom = np.array([4,6])
#angular_momentum =  np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [2,0,0], [1,1,0], [1,0,1], [0,2,0], [0,1,1], [0,0,2]])

#centers = np.repeat(geom, nbf_per_atom, axis=0)
#nbf = exponents.shape[0]
#
## Preprocess. Get unique oei indices, their 6d angular momentum vectors, and the indices which need differentiating
## Separate integrals into classes
## For every one electron integral, determine its integral class (s|s) (p|s) (p|p) etc
#unique_oei_indices = []
#oei_angular_momentum = []
#integral_class = []
#grad_indices = []
#for i in range(nbf):
#    for j in range(i+1):
#        unique_oei_indices.append([i,j])
#        pi, pj, pk = angular_momentum[i]
#        qi, qj, qk = angular_momentum[j]
#        target_am = onp.array([pi,pj,pk,qi,qj,qk])
#        oei_angular_momentum.append(target_am)
#        #integral_class.append([onp.sum(angular_momentum[i]), onp.sum(angular_momentum[j])])
#        bra = onp.sum(angular_momentum[i])
#        ket = onp.sum(angular_momentum[i])

#        if bra == ket == 0:
#
#        if (bra == 1 and ket == 0) or (bra == 0 and ket == 1):
#
#        if (bra == 1 and ket == 1): 
