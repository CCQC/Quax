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
#print(coeffs.shape)

junk, bounds =  onp.unique(am, return_index=True, axis=0)
unique_am = junk.tolist()
print(unique_am)
#bounds = bounds.tolist()
#print(bounds)

#print(len(centers))
#print(am.shape)

def general(centers,exps,coeff,am):
    A = centers[0,:]
    B = centers[1,:]
    C = centers[2,:]
    D = centers[3,:]
    aa = exps[:,0]
    bb = exps[:,1]
    cc = exps[:,2]
    dd = exps[:,3]

    # NOTE lax.map may be more efficient! vmap is more convenient since we dont have to map every argument 
    # although partial could probably be used with lax.map to mimic None behavior in vmap
    # Do not compute dummy padded value in contractions, set to 0
    ssss = jax.vmap(lambda A,B,C,D,aa,bb,cc,dd,coeff : 
                    np.where(coeff == 0, 0, eri_ssss(A,B,C,D,aa,bb,cc,dd,coeff)),(None,None,None,None,0,0,0,0,0))
    psss = jax.vmap(lambda A,B,C,D,aa,bb,cc,dd,coeff : 
                    np.where(coeff == 0, 0, eri_psss(A,B,C,D,aa,bb,cc,dd,coeff)),(None,None,None,None,0,0,0,0,0))
    ppss = jax.vmap(lambda A,B,C,D,aa,bb,cc,dd,coeff : 
                    np.where(coeff == 0, 0, eri_ppss(A,B,C,D,aa,bb,cc,dd,coeff)),(None,None,None,None,0,0,0,0,0))
    psps = jax.vmap(lambda A,B,C,D,aa,bb,cc,dd,coeff : 
                    np.where(coeff == 0, 0, eri_psps(A,B,C,D,aa,bb,cc,dd,coeff)),(None,None,None,None,0,0,0,0,0))
    ppps = jax.vmap(lambda A,B,C,D,aa,bb,cc,dd,coeff : 
                    np.where(coeff == 0, 0, eri_ppps(A,B,C,D,aa,bb,cc,dd,coeff)),(None,None,None,None,0,0,0,0,0))
    pppp = jax.vmap(lambda A,B,C,D,aa,bb,cc,dd,coeff : 
                    np.where(coeff == 0, 0, eri_pppp(A,B,C,D,aa,bb,cc,dd,coeff)),(None,None,None,None,0,0,0,0,0))

    if am == 'ssss':
        primitives = ssss(A,B,C,D,aa,bb,cc,dd,coeff)

    # NOTE TODO these permutation cases should just be taken care of in the preprocessing step. Would save a lot of time probably
    # issue: cannot take care of tranposes in preprocessing step
    if am == 'psss':
        primitives = psss(A,B,C,D,aa,bb,cc,dd,coeff)
    if am == 'spss':
        primitives = psss(B,A,C,D,bb,aa,cc,dd,coeff)
    if am == 'ssps':
        primitives = psss(C,D,A,B,cc,dd,aa,bb,coeff)
    if am == 'sssp':
        primitives = psss(D,C,B,A,dd,cc,bb,aa,coeff)

    if am == 'ppss':
        primitives = ppss(A,B,C,D,aa,bb,cc,dd,coeff)
    if am == 'sspp':
        primitives = ppss(C,D,A,B,cc,dd,aa,bb,coeff)

    if am == 'psps':
        primitives = psps(A,B,C,D,aa,bb,cc,dd,coeff)
    if am == 'spps':
        primitives = psps(B,A,C,D,bb,aa,cc,dd,coeff)
    if am == 'pssp':
        primitives = psps(A,B,D,C,aa,bb,dd,cc,coeff)
    if am == 'spsp':
        primitives = psps(B,A,D,C,bb,aa,dd,cc,coeff)

    # PPPS is a weird case, it returns a (3,3,3), where each dim is differentiation w.r.t. 0, then 1, then 2
    # You must transpose the result when you permute in some cases 
    if am == 'ppps':
        primitives = ppps(A,B,C,D,aa,bb,cc,dd,coeff)
    if am == 'ppsp': 
        primitives = ppps(A,B,D,C,aa,bb,dd,cc,coeff)
    if am == 'pspp': #TODO thes results need to be transposed somewhere
        #primitives = np.transpose(ppps(C,D,A,B,cc,dd,aa,bb,coeff), (0,1,4,2,3))
        primitives = ppps(C,D,A,B,cc,dd,aa,bb,coeff)
    if am == 'sppp': #TODO thes results need to be transposed somewhere
        #primitives = np.transpose(ppps(D,C,B,A,dd,cc,bb,aa,coeff), (0,1,4,3,2))
        primitives = ppps(D,C,B,A,dd,cc,bb,aa,coeff)
    if am == 'pppp':
        primitives = pppp(A,B,C,D,aa,bb,cc,dd,coeff)
    return np.sum(primitives, axis=0).reshape(-1) # contract and return flattened vector

# Map the general two electron integral function over a leading axis of shell quartets
#V_general = jax.vmap(general, (0,0,0,0,0,0,0,0,0,None))
V_general = jax.vmap(general, (0,0,0,None))

def compute(geom, exps, coeffs, centers, unique_am, b):
    ''' 

    unique_am : list of lists of size 4
        Represents the sorted angular momentum cases, 0000, 0001,..., 1111, etc
    bounds : array
        The bounds of angular momentum cases. It is the first index occurence of each new angular momentum case along the leading axis of exps and coeffs, which are sorted such
        that like-angular momentum cases are grouped together. 
        Used to tell which function to use for which data
    '''
    centers = np.take(geom, centers, axis=0)
    l,u = 0,1
    for case in unique_am:
        # convert list of AM [0101] to string spsp, etc   # must end in else
        tag = ''.join(['s' if j == 0 else 'p' if j == 1 else 'd' for j in case])
        if tag == 'pppp':
            result = V_general(centers[b[l]:],exps[b[l]:],coeffs[b[l]:],tag)
        else:
            result = V_general(centers[b[l]:b[u]],exps[b[l]:b[u]],coeffs[b[l]:b[u]],tag)
        print(result.shape)
        l += 1
        u += 1
    #print(result.shape)
    #result = V_general(centers[b[1]:b[2]],exps[b[1]:b[2]],coeffs[b[1]:b[2]],'psss')
    #print(result.shape)
    #result = V_general(centers[b[1]:b[2]],exps[b[1]:b[2]],coeffs[b[1]:b[2]],'spss')
    #print(result.shape)
    #result = V_general(centers[b[1]:b[2]],exps[b[1]:b[2]],coeffs[b[1]:b[2]],'ssps')
    #print(result.shape)
    #result = V_general(centers[b[1]:b[2]],exps[b[1]:b[2]],coeffs[b[1]:b[2]],'sssp')
    #print(result.shape)

    #result = V_general(centers[b[3]:b[4]],exps[b[3]:b[4]],coeffs[b[3]:b[4]],'ppss')
    #print(result.shape)
    #result = V_general(centers[b[4]:b[5]],exps[b[4]:b[5]],coeffs[b[4]:b[5]],'sspp')
    #print(result.shape)

    #result = V_general(centers[b[4]:b[5]],exps[b[4]:b[5]],coeffs[b[4]:b[5]],'psps')
    #print(result.shape)

    #result = V_general(centers[b[4]:b[5]],exps[b[4]:b[5]],coeffs[b[4]:b[5]],'psps')
    #print(result.shape)


    #result = V_general(centers[b[5]:b[6]],exps[b[5]:b[6]],coeffs[b[5]:b[6]],'pppp')
    #print(result.shape)


compute(geom, exps, coeffs, centers, unique_am, bounds)
