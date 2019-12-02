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

max_prim = basis_set.max_nprimitive()
biggest_K = max_prim**4
nbf = basis_set.nbf()
nshells = len(basis_dict)

shell_quartets = old_cartesian_product(np.arange(nshells), np.arange(nshells), np.arange(nshells), np.arange(nshells))

def preprocess(shell_quartets, basis):
    """Args: shell quartet indices, basis dictionary"""
    exps = []
    coeffs = []
    centers = []
    ams = []
    for quartet in shell_quartets:
        i,j,k,l = quartet
        #basis_dict[i]['am'], basis_dict[i]['idx'], basis_dict[i]['idx_stride']
        c1, aa, atom1, am1 = onp.asarray(basis[i]['coef']), onp.asarray(basis[i]['exp']), basis[i]['atom'], basis[i]['am']
        c2, bb, atom2, am2 = onp.asarray(basis[j]['coef']), onp.asarray(basis[j]['exp']), basis[j]['atom'], basis[j]['am']
        c3, cc, atom3, am3 = onp.asarray(basis[k]['coef']), onp.asarray(basis[k]['exp']), basis[k]['atom'], basis[k]['am']
        c4, dd, atom4, am4 = onp.asarray(basis[l]['coef']), onp.asarray(basis[l]['exp']), basis[l]['atom'], basis[l]['am']


        exp_combos = old_cartesian_product(aa,bb,cc,dd)
        coeff_combos = np.prod(cartesian_product(c1,c2,c3,c4), axis=1)
        am_vec = np.array([am1, am2, am3, am4])
        # Pad all arrays to same size (largest contraction)
        K = exp_combos.shape[0]
        exps.append(onp.pad(exp_combos, ((0, biggest_K - K), (0,0))))
            
        coeffs.append(onp.pad(coeff_combos, (0, biggest_K - K)))
        
        centers.append([atom1,atom2,atom3,atom4])
        ams.append(am_vec)

    #TODO take care of different cases of psss, ppss, psps, ppps within preprocess function
    # Sort all data by angular momentum case. 
    # NOTE can get places where start/stop using onp.unique
    am = onp.asarray(ams)
    sort_indx = onp.lexsort((am[:,3],am[:,2],am[:,1],am[:,0])) 
    exps = onp.asarray(exps)[sort_indx]
    coeffs = onp.asarray(coeffs)[sort_indx]
    centers = onp.asarray(centers)[sort_indx].tolist()
    am = am[sort_indx]
    return np.asarray(exps), np.asarray(coeffs), centers, np.asarray(am)

exps, coeffs, centers, am = preprocess(shell_quartets, basis_dict)
#print(exps.shape)
print(coeffs.shape)
#print(len(centers))
#print(am.shape)

def general(A,B,C,D,aa,bb,cc,dd,coeff,am):
    # NOTE lax.map may be more efficient! vmap is more convenient since we dont have to map every argument 
    # although partial could probably be used with lax.map to mimic None behavior in vmap
    if am == 'ssss':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_ssss(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(A,B,C,D,aa,bb,cc,dd,coeff)
    if am == 'psss':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_psss(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(A,B,C,D,aa,bb,cc,dd,coeff)
    if am == 'spss':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_psss(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(B,A,C,D,bb,aa,cc,dd,coeff)
    if am == 'ssps':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_psss(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(C,D,A,B,cc,dd,aa,bb,coeff)
    if am == 'sssp':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_psss(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(D,C,B,A,dd,cc,bb,aa,coeff)

    if am == 'ppss':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_ppss(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(A,B,C,D,aa,bb,cc,dd,coeff)
    if am == 'sspp':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_ppss(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(C,D,A,B,cc,dd,aa,bb,coeff)
    if am == 'psps':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_psps(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(A,B,C,D,aa,bb,cc,dd,coeff)
    if am == 'pppp':
        primitives = jax.vmap(
                     lambda A,B,C,D,aa,bb,cc,dd,coeff : np.where(coeff == 0, 0, eri_pppp(A,B,C,D,aa,bb,cc,dd,coeff)),
                     (None,None,None,None,0,0,0,0,0))(A,B,C,D,aa,bb,cc,dd,coeff)
    return np.sum(primitives, axis=0)

# Map the general two electron integral function over a leading axis of shell quartets
V_general = jax.vmap(general, (0,0,0,0,0,0,0,0,0,None))

def compute(geom, exps, coeffs, centers, am):
    centers = np.take(geom, centers, axis=0)

    A = centers[:,0,:]
    B = centers[:,1,:]
    C = centers[:,2,:]
    D = centers[:,3,:]
    aa = exps[:,:,0]
    bb = exps[:,:,1]
    cc = exps[:,:,2]
    dd = exps[:,:,3]

    result = V_general(A,B,C,D,aa,bb,cc,dd,coeffs,'ssss')
    result = V_general(A,B,C,D,aa,bb,cc,dd,coeffs,'psss')
    result = V_general(A,B,C,D,aa,bb,cc,dd,coeffs,'ssps')
    result = V_general(A,B,C,D,aa,bb,cc,dd,coeffs,'ppss')
    result = V_general(A,B,C,D,aa,bb,cc,dd,coeffs,'sspp')
    result = V_general(A,B,C,D,aa,bb,cc,dd,coeffs,'pppp')


compute(geom, exps, coeffs, centers, am)
