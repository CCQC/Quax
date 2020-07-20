import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product, find_unique_shells
from functools import partial
from jax.experimental import loops
from pprint import pprint
from eri import *

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


@partial(jax.jit, static_argnums=(9,))
def general(A,B,C,D,aa,bb,cc,dd,coeff,am):
    if am == 'ssss':
        # lax.map may be more efficient! vmap is more convenient since we dont have to map every argument 
        # although partial could probably be used with lax.map to mimic None behavior in vmap
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_ssss(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(A,B,C,D,aa,bb,cc,dd,coeff)
    if am == 'psss':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_psss(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(A,B,C,D,aa,bb,cc,dd,coeff)
    if am == 'pppp':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_pppp(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(A,B,C,D,aa,bb,cc,dd,coeff)
    return np.sum(primitives, axis=0)


A = np.array([-0.4939594255,-0.2251760374, 0.3240754142])
B = np.array([ 0.4211401526, 1.8106751596,-0.1734137286])
C = np.array([-0.5304044183, 1.5987236612, 2.0935583523])
D = np.array([ 1.9190079941, 0.0838367286, 1.4064021040])

alpha = np.array([0.2,0.2]) 
beta  = np.array([0.3,0.3]) 
gamma = np.array([0.4,0.4])
delta = np.array([0.5,0.5])
coeff = np.array([1.0,1.0])

res = general(A,B,C,D,alpha,beta,gamma,delta,coeff,'ssss')
#:print(res)
#:res = general(A,B,C,D,alpha,beta,gamma,delta,coeff,'psss')
#:print(res)
#:res = general(A,B,C,D,alpha,beta,gamma,delta,coeff,'pppp')
#:print(res)


def mapper(args,am):
    basis_data, geom_data = args
    A = geom_data[0]
    B = geom_data[1]
    C = geom_data[2]
    D = geom_data[3]
    aa = basis_data[:,0] 
    bb = basis_data[:,1] 
    cc = basis_data[:,2] 
    dd = basis_data[:,3] 
    coeff = basis_data[:,4] 

    #A,B,C,D,aa,bb,cc,dd,coeff = args
    #A,B,C,D,aa,bb,cc,dd,coeff = args

    if am == 'ssss':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_ssss(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(A,B,C,D,aa,bb,cc,dd,coeff)
    if am == 'psss':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_psss(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(A,B,C,D,aa,bb,cc,dd,coeff)
    if am == 'pppp':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_pppp(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(A,B,C,D,aa,bb,cc,dd,coeff)
    return np.sum(primitives, axis=0)

# Evaluate 5 (ss|ss) integrals with map
fourcenter = np.vstack((A,B,C,D))
#print(geom)
geom_data = np.asarray([fourcenter, fourcenter, fourcenter, fourcenter, fourcenter])
A = np.tile(A, (5,1))
B = np.tile(B, (5,1))
C = np.tile(C, (5,1))
D = np.tile(D, (5,1))

tmp = np.vstack((alpha,beta,gamma,delta,coeff))
print(tmp)
basis_data = np.asarray([tmp, tmp, tmp, tmp, tmp]) 
basis_data = np.transpose(basis_data, (0,2,1))

#result = jax.lax.map(partial(mapper, am='ssss'), (basis_data, geom_data))
#print(result)

#result = jax.lax.map(partial(mapper, am='psss'), (basis_data, geom_data))
#print(result)

#result = jax.lax.map(partial(mapper, am='pppp'), (basis_data, geom_data))
#print(result)


# is it possible to do one vmap call? can you map over
# This is the winner (FOR NOW), way faster than lax.map.
def vmapper(A,B,C,D,aa,bb,cc,dd,coeff,am):
    if am == 'ssss':
        # lax.map may be more efficient! vmap is more convenient since we dont have to map every argument 
        # although partial could probably be used with lax.map to mimic None behavior in vmap
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_ssss(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(A,B,C,D,aa,bb,cc,dd,coeff)
    if am == 'psss':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_psss(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(A,B,C,D,aa,bb,cc,dd,coeff)
    if am == 'pppp':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_pppp(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(A,B,C,D,aa,bb,cc,dd,coeff)
    return np.sum(primitives, axis=0)

# this works
test = jax.vmap(vmapper, (0,0,0,0,0,0,0,0,0,None))

#result = test(A, B, C, D, np.tile(alpha,(5,1)), np.tile(beta,(5,1)), np.tile(gamma,(5,1)), np.tile(delta,(5,1)), np.tile(coeff,(5,1)), 'ssss')
#print(result)
#result = test(A, B, C, D, np.tile(alpha,(5,1)), np.tile(beta,(5,1)), np.tile(gamma,(5,1)), np.tile(delta,(5,1)), np.tile(coeff,(5,1)), 'psss')
#print(result)

result = test(A, B, C, D, np.tile(alpha,(5,1)), np.tile(beta,(5,1)), np.tile(gamma,(5,1)), np.tile(delta,(5,1)), np.tile(coeff,(5,1)), 'pppp')
print(result)


# all contractions must be padded to same size
# will likely need padded indices as well? but you may not have to USE the pads. You can slice them each time, since you know how much it will take.


#sketch
#def ssss(carry, i)
#    # basis data: atomidx1, atomidx2, atomidx3, atomidx4, aa, bb, cc, dd, coeff
#    G, basis_data = carry
#    A, B, C, D = geom[basis_data[0]], geom[basis_data[1]], geom[basis_data[2]], geom[basis_data[3]]
#    aa, bb, cc, dd, coeff = basis_data[4],basis_data[5],basis_data[6],basis_data[7],basis_data[8]
#    val = general(A,B,C,D,aa,bb,cc,dd,coeff,'ssss')
#    # indices could be sliced so they are not huge. (you know the slice size, it is dependent on which am case)
#    G = jax.ops.index_update(G, TODO, val)
#
#        
#    return new_carry, 0






#for i in range(50000):
#    res = general(A,B,C,D,alpha,beta,gamma,delta,coeff,'ssss')
#
#for i in range(50000):
#    res = general(A,B,C,D,alpha,beta,gamma,delta,coeff,'psss')


#for i in range(100000):
#    res = general(A,B,C,D,alpha,beta,gamma,delta,coeff,'ssss')
#    res = general(A,B,C,D,alpha,beta,gamma,delta,coeff,'psss')
