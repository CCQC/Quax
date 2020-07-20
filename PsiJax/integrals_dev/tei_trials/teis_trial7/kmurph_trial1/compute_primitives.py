import psi4
import jax 
from jax import lax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from jax.experimental import loops
from integrals_utils import primitive_eri, np_cartesian_product
from pprint import pprint
np.set_printoptions(linewidth=300)

# Define molecule
molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         units bohr
                         """)

# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))

basis_name = 'cc-pvdz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)

max_prim = basis_set.max_nprimitive()
max_am = basis_set.max_am()
biggest_K = max_prim**4
nbf = basis_set.nbf()
nshells = len(basis_dict)
max_size = (max_am + 1) * (max_am + 2) // 2
shell_quartets = np_cartesian_product(np.arange(nshells), np.arange(nshells), np.arange(nshells), np.arange(nshells))
print("Number of basis functions: ",nbf)
print("Number of shells: ", nshells)
print("Number of shell quartets (redundant): ", shell_quartets.shape[0])
print("Max angular momentum: ", max_am)
print("Largest number of primitives: ", max_prim)
print("Largest contraction: ", biggest_K)
#pprint(basis_dict)

def am_vectors(am, length=3):
    '''
    Builds up all possible angular momentum component vectors of with total angular momentum 'am'
    am = 2 ---> [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    Returns a generator which must be converted to an iterable,
    for example, call the following: [list(i) for i in am_vectors(2)]

    Works by building up each possibility :
    For a given value in reversed(range(am+1)), find all other possible values for other entries in length 3 vector
     value     am_vectors(am-value,length-1)    (value,) + permutation
       2 --->         [0,0]                 ---> [2,0,0] ---> dxx
       1 --->         [1,0]                 ---> [1,1,0] ---> dxy
         --->         [0,1]                 ---> [1,0,1] ---> dxz
       0 --->         [2,0]                 ---> [0,2,0] ---> dyy
         --->         [1,1]                 ---> [0,1,1] ---> dyz
         --->         [0,2]                 ---> [0,0,2] ---> dzz
    '''
    if length == 1:
        yield (am,)
    else:
        # reverse so angular momentum order is canonical, e.g., dxx dxy dxz dyy dyz dzz
        for value in reversed(range(am + 1)): 
            for permutation in am_vectors(am - value,length - 1):
                yield (value,) + permutation

def preprocess(shell_quartets, basis):
    exps = []                
    coeffs = []
    centers = []
    repeats = []
    total_am = []                 
    for quartet in shell_quartets:
        # Construct data rows all primitives in this shell quartet
        i,j,k,l = quartet
        c1, aa, atom1, am1, size1 = onp.asarray(basis[i]['coef']), onp.asarray(basis[i]['exp']), basis[i]['atom'], basis[i]['am'], basis[i]['idx_stride']
        c2, bb, atom2, am2, size2 = onp.asarray(basis[j]['coef']), onp.asarray(basis[j]['exp']), basis[j]['atom'], basis[j]['am'], basis[j]['idx_stride']
        c3, cc, atom3, am3, size3 = onp.asarray(basis[k]['coef']), onp.asarray(basis[k]['exp']), basis[k]['atom'], basis[k]['am'], basis[k]['idx_stride']
        c4, dd, atom4, am4, size4 = onp.asarray(basis[l]['coef']), onp.asarray(basis[l]['exp']), basis[l]['atom'], basis[l]['am'], basis[l]['idx_stride']

        am_vec1 = onp.array([list(i) for i in am_vectors(am1)])
        am_vec2 = onp.array([list(i) for i in am_vectors(am2)])
        am_vec3 = onp.array([list(i) for i in am_vectors(am3)])
        am_vec4 = onp.array([list(i) for i in am_vectors(am4)])

        tmp_indices = np_cartesian_product(np.arange(size1),np.arange(size2),np.arange(size3),np.arange(size4))
        all_am = np.hstack((am_vec1[tmp_indices[:,0]],am_vec2[tmp_indices[:,1]],am_vec3[tmp_indices[:,2]],am_vec4[tmp_indices[:,3]]))
        
        # Create exp, coeff arrays and pad to same size (largest contraction) so they can be put into an array
        # THIS REALLY ISN'T NECESSARY, JUST SIMPLER. YOU COULD IN THEORY JUST PASS THROUGH EACH CONTRACTION SIZE IN BATCHES WHEN COMPUTING (multiple vmap function evaluations)
        exp_combos = np_cartesian_product(aa,bb,cc,dd)  # of size (K, 4)
        coeff_combos = onp.prod(np_cartesian_product(c1,c2,c3,c4), axis=1)  # of size K
        K = exp_combos.shape[0]
        exps_padded = onp.pad(exp_combos, ((0, biggest_K - K), (0,0)), constant_values=1.0)
        coeffs_padded = onp.pad(coeff_combos, (0, biggest_K - K), constant_values=0.0)
        #ALTERNATIVE: just pad the basis dict to even contraction, then push the cartesian product into the function which evaluastes the integrals
        # Need to collect the 'repeats' arg for np.repeat after the loop  such that they are (nbf**4, K, 4) and (nbf**4, K, 1)
        exps.append(exps_padded)
        coeffs.append(coeffs_padded)
        repeats.append(all_am.shape[0])
        centers.append([atom1,atom2,atom3,atom4])                                          
        total_am.append(all_am)

    #print(tmp_indices.shape)
    #print(tmp_indices)

    exps = onp.asarray(exps)
    coeffs = onp.asarray(coeffs)
    centers = onp.asarray(centers)
    repeats = onp.asarray(repeats)
    exps_final = onp.repeat(exps, repeats, axis=0)
    coeffs_final = onp.repeat(coeffs, repeats, axis=0)
    centers_final = onp.repeat(centers, repeats, axis=0)
    am_final = onp.vstack(total_am)

    print('exps shape',exps_final.shape)
    print('coeffs shape',coeffs_final.shape)
    print('am shape',am_final.shape)
    print('centers shape', centers_final.shape)

    return np.asarray(am_final), np.asarray(exps_final), np.asarray(centers_final), np.asarray(coeffs_final)


am, exps, centers, coeffs = preprocess(shell_quartets,basis_dict)


#def compute(


# test one contraction
#a = exps[0,:,0]
#b = exps[0,:,1]
#c = exps[0,:,2]
#d = exps[0,:,3]
#RA = geom[centers[0,0]]
#RB = geom[centers[0,1]]
#RC = geom[centers[0,2]]
#RD = geom[centers[0,3]]
#coeff = coeffs[0,:]

# Make function which does not evaluate dummy padded primitives in contractions
mapped_primitive_eri = jax.vmap(lambda L,a,b,c,d,RA,RB,RC,RD,contraction :
                                np.where(contraction > 0.0, primitive_eri(L,a,b,c,d,RA,RB,RC,RD,contraction), 0.0), 
                                (None,0,0,0,0,None,None,None,None,0))

center_vectors = geom[centers] # Shape (nbf**4, 4,3) 
RA = center_vectors[:,0,:] # Shape (nbf**4, 3)
RB = center_vectors[:,1,:] # Shape (nbf**4, 3)
RC = center_vectors[:,2,:] # Shape (nbf**4, 3)
RD = center_vectors[:,3,:] # Shape (nbf**4, 3)

print(am.shape, exps.shape, RA.shape, RB.shape, RC.shape, RD.shape, coeffs.shape)
#(10000, 12) (10000, 81, 4) (10000, 3) (10000, 3) (10000, 3) (10000, 3) (10000, 81)

# Expand dimensionality and compute all primitives at once
am = np.repeat(am, 81, axis=0)
exps = exps.reshape(-1,4)
RA = np.repeat(RA, 81, axis=0)
RB = np.repeat(RB, 81, axis=0)
RC = np.repeat(RC, 81, axis=0)
RD = np.repeat(RD, 81, axis=0)
coeffs = coeffs.reshape(-1)

print(am.shape, exps.shape, RA.shape, RB.shape, RC.shape, RD.shape, coeffs.shape) 
#(810000, 12) (810000, 4) (810000, 3) (810000, 3) (810000, 3) (810000, 3) (810000,)

#print(coeffs)
#for i in coeffs:
#    print(i)

#all_eri = jax.jit(jax.vmap(lambda L,a,b,c,d,RA,RB,RC,RD,contraction :
#                                np.where(contraction == 0.0, 0.0, primitive_eri(L,a,b,c,d,RA,RB,RC,RD,contraction)), 
#                                (0,0,0,0,0,0,0,0,0,0)))


#result = all_eri(am, exps[:,0],exps[:,1],exps[:,2],exps[:,3], RA, RB, RC, RD, coeffs)
#result = all_eri(am[:50000], exps[:50000,0],exps[:50000,1],exps[:50000,2],exps[:50000,3], RA[:50000], RB[:50000], RC[:50000], RD[:50000], coeffs[:50000])
#print(result.shape)

def tmp(superarg):
    L,a,b,c,d,RA,RB,RC,RD,contraction = superarg
    return primitive_eri(L,a,b,c,d,RA,RB,RC,RD,contraction)

jax.lax.map(tmp, (am[:50000], exps[:50000,0],exps[:50000,1],exps[:50000,2],exps[:50000,3], RA[:50000], RB[:50000], RC[:50000], RD[:50000], coeffs[:50000]))

#jax.lax.map(lambda 
#primitive_eri(L,a,b,c,d,RA,RB,RC,RD,contraction))

# Now flip back to contracted leading axis and sum
# reshape(-1,biggest_K)






# jit contracted eri takes about 15 sec to compile
#print(contracted_eri(am[0], exps[0,:,0], exps[0,:,1], exps[0,:,2], exps[0,:,3], RA[0], RB[0], RC[0], RD[0], coeffs[0]))

#for i in range(10000):
#    print(i)
#    fast_contracted_eri(am[i], exps[i,:,0], exps[i,:,1], exps[i,:,2], exps[i,:,3], RA[i], RB[i], RC[i], RD[i], coeffs[i])

#for i in range(10000):
#    #contracted_eri(am[0],a,b,c,d,RA,RB,RC,RD,coeff)
#    contracted_eri(am[0],a,b,c,d,RA,RB,RC,RD,coeff)
#    print(i)


## Super slow for unkown reasons
##result = all_eri(am, exps[:,:,0],exps[:,:,1],exps[:,:,2],exps[:,:,3], RA, RB, RC, RD, coeffs)
#result = all_eri(am[:100], exps[:100,:,0],exps[:100,:,1],exps[:100,:,2],exps[:100,:,3], RA[:100], RB[:100], RC[:100], RD[:100], coeffs[:100])
#print(result.shape)
#result = all_eri(am[100:200], exps[100:200,:,0],exps[100:200,:,1],exps[100:200,:,2],exps[100:200,:,3], RA[100:200], RB[100:200], RC[100:200], RD[100:200], coeffs[100:200])
#print(result.shape)


# Try nested double vmap DOESNT WORK. looks like the innervmap specializes based on argument shapes so different sizes of outer vmaps
#rule_them_all = jax.vmap(jax.vmap(lambda L,a,b,c,d,RA,RB,RC,RD,contraction :
#                                np.where(coeff > 0.0, primitive_eri(L,a,b,c,d,RA,RB,RC,RD,contraction), 0.0), 
#                                (None,0,0,0,0,None,None,None,None,0)), (0,0,0,0,0,0,0,0,0,0))

# does removing where help?
#rule_them_all = jax.vmap(jax.vmap(primitive_eri,(None,0,0,0,0,None,None,None,None,0)), (0,0,0,0,0,0,0,0,0,0))

#result = rule_them_all(am[:100], exps[:100,:,0],exps[:100,:,1],exps[:100,:,2],exps[:100,:,3], RA[:100], RB[:100], RC[:100], RD[:100], coeffs[:100])
#print(result.shape)
#result = rule_them_all(am[:200], exps[:200,:,0],exps[:200,:,1],exps[:200,:,2],exps[:200,:,3], RA[:200], RB[:200], RC[:200], RD[:200], coeffs[:200])
#print(result.shape)

#for i in range(100):
#    l = i * 100
#    u = (i+1) * 100
#    all_eri(am[l:u], exps[l:u,:,0],exps[l:u,:,1],exps[l:u,:,2],exps[l:u,:,3], RA[l:u], RB[l:u], RC[l:u], RD[l:u], coeffs[l:u])
#    print(l,u)
#result = rule_them_all(am[:200], exps[:200,:,0],exps[:200,:,1],exps[:200,:,2],exps[:200,:,3], RA[:200], RB[:200], RC[:200], RD[:200], coeffs[:200])
#result = rule_them_all(am, exps[:,:,0],exps[:,:,1],exps[:,:,2],exps[:,:,3], RA, RB, RC, RD, coeffs)

#result = all_eri(am[100:200], exps[100:200,:,0],exps[100:200,:,1],exps[100:200,:,2],exps[100:200,:,3], RA[100:200], RB[100:200], RC[100:200], RD[100:200], coeffs[100:200])


#result = all_eri(am[:200], exps[:200,:,0],exps[:200,:,1],exps[:200,:,2],exps[:200,:,3], RA[:200], RB[:200], RC[:200], RD[:200], coeffs[:200])
#print(result.shape)
#print(result)


##@jax.jit
#def contracted_eri(superarg):
#    '''Must pass single 12 component angular momentum vector, 
#    a vector of each orbital exponent of size equal to contraction size, 
#    single geometry vectors, and a coefficient vector equal to contraction size
#    '''
#    am, a,b,c,d,RA,RB,RC,RD,coeff = superarg
#    result = mapped_primitive_eri(am, a,b,c,d,RA,RB,RC,RD,coeff)     
#    return np.sum(result)


#result = jax.lax.map(contracted_eri, (am, exps[:,:,0],exps[:,:,1],exps[:,:,2],exps[:,:,3], RA, RB, RC, RD, coeffs))


#print(am.shape)
#print(exps[:,:,0].shape)
#print(exps[:,:,1].shape)
#print(exps[:,:,2].shape)
#print(exps[:,:,3].shape)
#print(RA.shape) 
#print(RB.shape)
#print(RC.shape)
#print(RD.shape)
#print(coeffs.shape)
#
##print(geom[centers].shape)
# 
##print(centers)
#
#
##all_eri(am,exps,centers,coeffs)
#
#
