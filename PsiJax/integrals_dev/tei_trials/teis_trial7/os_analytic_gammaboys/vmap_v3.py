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
shell_quartets = old_cartesian_product(np.arange(nshells), np.arange(nshells), np.arange(nshells), np.arange(nshells))
print("Number of basis functions: ",nbf)
print("Number of shells: ", nshells)
print("Number of redundant shell quartets: ", shell_quartets.shape[0])
print("Max angular momentum: ", max_am)
print("Largest number of primitives: ", max_prim)
print("Largest contraction: ", biggest_K)

def homogenize_basisdict(basis_dict, max_prim):
    '''
    Make it so all contractions are the same size in the basis dict by padding exp and coef values to 1 and 0.
    Also create 'indices' key which says where along axis the integral should go, but padded with -1's to maximum angular momentum size
    This allows you to pack them neatly into an array, and then worry about redundant computation later.
    '''
    new_dict = basis_dict.copy()
    for i in range(len(basis_dict)):
        current_exp = onp.asarray(basis_dict[i]['exp'])
        new_dict[i]['exp'] = onp.asarray(onp.pad(current_exp, (0, max_prim - current_exp.shape[0]), constant_values=1))
        current_coef = onp.asarray(basis_dict[i]['coef'])
        new_dict[i]['coef'] = onp.asarray(onp.pad(current_coef, (0, max_prim - current_coef.shape[0])))
        #TODO what to do about indices?
        #idx, size = basis_dict[i]['idx'], basis_dict[i]['idx_stride']
        #indices = onp.repeat(idx, size) + onp.arange(size)
        #print(indices)
    return new_dict
new_basis_dict = homogenize_basisdict(basis_dict, max_prim)

def preprocess(basis):
    exps = []
    coeffs = []
    centers = []
    ams = []
    all_indices = []

    for quartet in shell_quartets:
        i,j,k,l = quartet
        c1, aa, atom1, am1, idx1, size1 = onp.asarray(basis[i]['coef']), onp.asarray(basis[i]['exp']), basis[i]['atom'], basis[i]['am'], basis[i]['idx'], basis[i]['idx_stride']
        c2, bb, atom2, am2, idx2, size2 = onp.asarray(basis[j]['coef']), onp.asarray(basis[j]['exp']), basis[j]['atom'], basis[j]['am'], basis[j]['idx'], basis[j]['idx_stride']
        c3, cc, atom3, am3, idx3, size3 = onp.asarray(basis[k]['coef']), onp.asarray(basis[k]['exp']), basis[k]['atom'], basis[k]['am'], basis[k]['idx'], basis[k]['idx_stride']
        c4, dd, atom4, am4, idx4, size4 = onp.asarray(basis[l]['coef']), onp.asarray(basis[l]['exp']), basis[l]['atom'], basis[l]['am'], basis[l]['idx'], basis[l]['idx_stride']

        exps.append([aa,bb,cc,dd])
        coeffs.append([c1,c2,c3,c4])
        centers.append([atom1,atom2,atom3,atom4])
        ams.append([am1,am2,am3,am4])

        indices1 = onp.repeat(idx1, size1) + onp.arange(size1)
        indices2 = onp.repeat(idx2, size2) + onp.arange(size2)
        indices3 = onp.repeat(idx3, size3) + onp.arange(size3)
        indices4 = onp.repeat(idx4, size4) + onp.arange(size4)
        indices = old_cartesian_product(indices1,indices2,indices3,indices4)
        indices = onp.pad(indices, ((0, 81-indices.shape[0]),(0,0)), constant_values=-1)
        all_indices.append(indices)
    print("loop done")
    am = onp.asarray(ams)
    sort_indx = onp.lexsort((am[:,3],am[:,2],am[:,1],am[:,0])) 
    exps = onp.asarray(exps)[sort_indx]
    coeffs = onp.asarray(coeffs)[sort_indx]
    centers = onp.asarray(centers)[sort_indx].tolist()
    am = am[sort_indx]
    all_indices = onp.asarray(all_indices)[sort_indx]
    return np.asarray(exps), np.asarray(coeffs), centers, np.asarray(am), np.asarray(all_indices, dtype=np.int16)

def general_tei(centers,exps,coeff,am,geom):
    coeff = np.prod(cartesian_product(coeff[0,:],coeff[1,:],coeff[2,:],coeff[3,:]), axis=1)
    exps = cartesian_product(exps[0,:],exps[1,:],exps[2,:],exps[3,:])
    A, B, C, D = geom[centers[0]], geom[centers[1]], geom[centers[2]], geom[centers[3]]
    aa, bb, cc, dd = exps[:,0], exps[:,1], exps[:,2], exps[:,3]
    # Note: This scheme may not be optimal, you're evaluating
    # uneeded data for coeff = 0, etc. however, np.where can blow up compilation time of derivatives so hmm
    tmp_func = jax.vmap(base,(None,None,None,None,0,0,0,0,0))
    ssss = jax.vmap(eri_ssss,(0,0,0,0,0,0,0,0,0,0,0,None))
    psss = jax.vmap(eri_psss,(0,0,0,0,0,0,0,0,0,0,0,None))
    ppss = jax.vmap(eri_ppss,(0,0,0,0,0,0,0,0,0,0,0,None))
    psps = jax.vmap(eri_psps,(0,0,0,0,0,0,0,0,0,0,0,None))
    ppps = jax.vmap(eri_ppps,(0,0,0,0,0,0,0,0,0,0,0,None))
    pppp = jax.vmap(eri_pppp,(0,0,0,0,0,0,0,0,0,0,0,None))

    if am == 'ssss':
        args = tmp_func(A,B,C,D,aa,bb,cc,dd,coeff)
        primitives = ssss(*args,0)
    elif am == 'psss':
        args = tmp_func(A,B,C,D,aa,bb,cc,dd,coeff)
        primitives = psss(*args,0)
    elif am == 'spss':
        args = tmp_func(B,A,C,D,bb,aa,cc,dd,coeff)
        primitives = psss(*args,0)
    elif am == 'ssps':
        args = tmp_func(C,D,A,B,cc,dd,aa,bb,coeff)
        primitives = psss(*args,0)
    elif am == 'sssp':
        args = tmp_func(D,C,B,A,dd,cc,bb,aa,coeff)
        primitives = psss(*args,0)

    elif am == 'ppss':
        args = tmp_func(A,B,C,D,aa,bb,cc,dd,coeff)
        primitives = ppss(*args,0)
    elif am == 'sspp':
        args = tmp_func(C,D,A,B,cc,dd,aa,bb,coeff)
        primitives = ppss(*args,0)

    elif am == 'psps':
        args = tmp_func(A,B,C,D,aa,bb,cc,dd,coeff)
        primitives = psps(*args,0)
    elif am == 'spps':
        args = tmp_func(B,A,C,D,bb,aa,cc,dd,coeff)
        primitives = psps(*args,0)
    elif am == 'pssp':
        args = tmp_func(A,B,D,C,aa,bb,dd,cc,coeff)
        primitives = psps(*args,0)
    elif am == 'spsp':
        args = tmp_func(B,A,D,C,bb,aa,dd,cc,coeff)
        primitives = psps(*args,0)

    # PPPS is a weird case, it returns a (3,3,3), where each dim is differentiation w.r.t. center 0, then 1, then 2
    # You must transpose the result when you permute in some cases 
    elif am == 'ppps':
        args = tmp_func(A,B,C,D,aa,bb,cc,dd,coeff)
        primitives = ppps(*args,0)
    elif am == 'ppsp': 
        args = tmp_func(A,B,D,C,aa,bb,dd,cc,coeff)
        primitives = ppps(*args,0)
    elif am == 'pspp': 
        args = tmp_func(C,D,A,B,cc,dd,aa,bb,coeff)
        primitives = np.transpose(ppps(*args,0), (0,3,1,2))
    elif am == 'sppp': 
        args = tmp_func(D,C,B,A,dd,cc,bb,aa,coeff)
        primitives = np.transpose(ppps(*args,0), (0,3,2,1))
    elif am == 'pppp':
        args = tmp_func(A,B,C,D,aa,bb,cc,dd,coeff)
        primitives = pppp(*args,0)
    return np.sum(primitives, axis=0).reshape(-1) # contract and return flatten

# Map the general two electron integral function over a leading axis of shell quartets
V_general = jax.vmap(general_tei, (0,0,0,None,None))

def compute(geom, basis_dict, shell_quartets):
#def compute(g1, g2, basis_dict, shell_quartets):
    ''' 

    unique_am : list of lists of size 4
        Represents the sorted angular momentum cases, 0000, 0001,..., 1111, etc
    bounds : list 
        The bounds of angular momentum cases. It is the first index occurence of each new angular momentum case along the leading axis of exps and coeffs, which are sorted such
        that like-angular momentum cases are grouped together. Used to tell which function to use for which data
    '''
#    geom = np.vstack((g1,g2))
    exps, coeffs, centers, am, indices = preprocess(basis_dict)
    print('preprocessing done')
    junk, bounds =  onp.unique(am, return_index=True, axis=0)
    unique_am = junk.tolist()
    bounds = bounds.tolist()
    print('uniqueness filtering done')

    centers = np.asarray(onp.asarray(centers))
    #centers = np.take(geom, centers, axis=0) # this guy is SO SLOW
    u = bounds[1:]
    u.append(-1) # upper (u) and lower (l) bounds of integral class indices 
    l = bounds   # used to slice data arrays before passing to integral class functions 

    G = np.zeros((nbf,nbf,nbf,nbf))
    # Compute each TEI class and place in G
    slices = []
    sizes = []
    for i in range(len(unique_am)):
        # Find size of this integral class (number of integrals)
        tmp = [(j + 1) * (j + 2) // 2 for j in unique_am[i]]
        size = np.prod(tmp)
        # convert list of AM [0101] to string spsp, etc
        am_class = ''.join(['s' if j == 0 else 'p' if j == 1 else 'd' for j in unique_am[i]])
        if u[i] == -1:
            s = slice(l[i], None)
        else:
            s = slice(l[i], u[i])
        slices.append(s)
        sizes.append(size)
        # Slicing the contraction would have to make slices which say what the maximum contraction size is for each case is, 
        # would only save a little bit i think 
        print(am_class)
        #eris = jax.lax.map(partial(general, am=am_class, geom=geom), (centers[s], exps[s], coeffs[s]))
        eris = V_general(centers[s],exps[s],coeffs[s],am_class, geom)
        G = jax.ops.index_update(G, (indices[s,:size,0],indices[s,:size,1],indices[s,:size,2],indices[s,:size,3]), eris)
    return G

G = compute(geom, basis_dict, shell_quartets)
mints = psi4.core.MintsHelper(basis_set)
psi_G = np.asarray(onp.asarray(mints.ao_eri()))
print(np.allclose(G, psi_G))

#grad = jax.jacfwd(compute)(geom,basis_dict, shell_quartets)
#print(grad.shape)
#hess = jax.jacfwd(jax.jacfwd(compute))(geom,basis_dict, shell_quartets)
#print(hess.shape)
#cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(compute)))(geom,basis_dict, shell_quartets)
#print(cube.shape)
#quar = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(compute))))(geom,basis_dict, shell_quartets)
#quar = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(compute))))(geom[0], geom[1],basis_dict, shell_quartets)
#print(quar.shape)



