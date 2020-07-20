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

def overlap_ss(centers, aa, bb):
    Ax, Ay, Az = centers[0]
    Cx, Cy, Cz = centers[1]
    ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * ((Ax - Cx)**2 + (Ay - Cy)**2 + (Az - Cz)**2))) / (aa + bb))
    return ss

print(overlap_ss(np.array([[0.0,0.0,-0.849220457955], [0.0,0.0, 0.849220457955]]), 0.5, 0.5))
print(jax.jacfwd(overlap_ss)(np.array([[0.0,0.0,-0.849220457955], [0.0,0.0, 0.849220457955]]), 0.5, 0.5))
print(jax.jacfwd(jax.jacfwd(overlap_ss))(np.array([[0.0,0.0,-0.849220457955], [0.0,0.0, 0.849220457955]]), 0.5, 0.5))




#def overlap_p_s(args, exponent_vector, idx1):
#    alpha = exponent_vector[idx1]
#
#    return (1 / (2 * alpha)) * jax.jacfwd(overlap_ss) # Full gradient 



def overlap_s_s(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
    ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * ((Ax - Cx)**2 + (Ay - Cy)**2 + (Az - Cz)**2))) / (aa + bb))
    return ss, (aa,bb)

    

def overlap_pz_s(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
    deriv_ss = jax.grad(overlap_s_s, argnums=2, has_aux=True)
    tmp, exps = deriv_ss(Ax, Ay, Az, Cx, Cy, Cz, aa, bb)
    aa,bb = exps
    return (1 / (2 * aa)) * tmp

def overlap_dzz_s(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
    deriv_ss = jax.grad(overlap_s_s, argnums=2, has_aux=True) # function which is derivative of ss w.r.t Az
    term2, junk = deriv_ss(Ax, Ay, Az, Cx, Cy, Cz, aa, bb)
    term1 = jax.grad(overlap_pz_s, 2)(Ax, Ay, Az, Cx, Cy, Cz, aa, bb)
    return (1 / (2 * aa)) * (term1 + term2)

args = (0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955,0.5, 0.5)
#tmp1 = overlap_s_s(*args)
#$tmp1 = overlap_pz_s(*args) * normalize(args[-2], 0,0,1) * normalize(args[-1], 0,0,0)
#$tmp2 = overlap_dzz_s(*args) * normalize(args[-2], 0,0,2) * normalize(args[-1], 0,0,0)
#$tmp2 = overlap_dzz_s(*args) * 0.4237772081237576

#print(tmp1)
#print(tmp2)
#overlap_px_s = 


#grad_overlap = jax.grad(overlap_ss)
#
#def overlap_ps(args, exponent_vector, idx1):
#    alpha = exponent_vector[idx1]
#    # Gradient w.r.t. ALL cartesians. this gives 
#    return (1 / (2 * alpha)) * jax.grad(overlap_ss)
#
#def overlap_pp(args, exponent_vector, idx1, idx2):
#    alpha1 = exponent_vector[idx1]
#    alpha2 = exponent_vector[idx2]
#    #return (1 / (2 * alpha2)) * jax.grad((1 / (2 * alpha1)) * jax.grad(overlap_ss)(*args)[idx1])(*args)[idx2]
#    return (1 / (2 * alpha2)) * jax.grad(overlap_ps(args, exponent_vector, idx1))(*args)[idx2]
#
#geom = np.array([[0.0,0.0,-0.849220457955],
#                 [0.0,0.0, 0.849220457955]])
#charge = np.array([1.0,1.0])
#
#
#args = (np.array([0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955]),0.5, 0.5)
#print(overlap_ss(*args))
#print(overlap_ps(args, np.array([0.5,0.5,0.5,0.5,0.5,0.5]), 2))
#print(overlap_pp(args, np.array([0.5,0.5,0.5,0.5,0.5,0.5]), 2, 2))
#
#np.array([0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955])
### S S test
##exponents = np.repeat(0.5, 2)
##nbf_per_atom = np.array([1,1])
##angular_momentum = np.array([[0,0,0], [0,0,0]])
#
## S P test
#exponents = np.repeat(0.5, 4)
#nbf_per_atom = np.array([1,3])
#angular_momentum = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
#
### P P test 
##exponents = np.repeat(0.5, 6)
##nbf_per_atom = np.array([3,3])
##angular_momentum = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,0],[0,1,0],[0,0,1]])
#
## S P D test
#exponents = np.repeat(0.5, 10)
#nbf_per_atom = np.array([4,6])
#angular_momentum =  np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [2,0,0], [1,1,0], [1,0,1], [0,2,0], [0,1,1], [0,0,2]])
#
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


        #integral_class.append([onp.sum(angular_momentum[i]), onp.sum(angular_momentum[j])])

        #mask = np.where(target_am != 0, True, False)
        #if np.any(mask):
        #    cart_indices = np.arange(6)[mask]
        #    tmp_grad_idx = np.repeat(cart_indices, target_am[mask])
        #    grad_indices.append(tmp_grad_idx)
        #else:
        #    grad_indices.append([])
             
#print(onp.asarray(oei_angular_momentum))
#
#print(onp.asarray(integral_class))
#test, blah = onp.unique(integral_class, return_inverse=True)
#print(blah)

#print(np.sort(np.asarray(integral_class), axis=0))
#print(uniq)
#print(counts)
#print(oei_angular_momentum)
#unique_oei_indices = np.asarray(unique_oei_indices)
#oei_angular_momentum = np.asarray(oei_angular_momentum)
#grad_indices = np.asarray(grad_indices)

#print(oei_angular_momentum)


    

