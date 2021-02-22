import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product, find_unique_shells
from functools import partial
from jax.experimental import loops

from eri import *

np.set_printoptions(linewidth=800, suppress=True, threshold=100)


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
nbf = basis_set.nbf()
nshells = len(basis_dict)
unique_shell_quartets = find_unique_shells(nshells)
print("number of basis functions", nbf)
print("number of shells ", nshells)
print("number of shell quartets", nshells**4)
print("number of unique shell quartets", unique_shell_quartets.shape[0])

def preprocess(unique_shell_quartets, basis_dict):
    basis_data = [] # A list of every primitive exponent, coeff of 4 centers
    centers = []
    angular_momenta = [] # list of string identifiers '1010' etc
    count_primitives = []
    indices = []
    place = 0
    for quartet in unique_shell_quartets:
        i,j,k,l = quartet
        c1, exp1, atom1_idx, am1, dim1 = onp.asarray(basis_dict[i]['coef']), onp.asarray(basis_dict[i]['exp']), basis_dict[i]['atom'], basis_dict[i]['am'], basis_dict[i]['idx_stride']
        c2, exp2, atom2_idx, am2, dim2 = onp.asarray(basis_dict[j]['coef']), onp.asarray(basis_dict[j]['exp']), basis_dict[j]['atom'], basis_dict[j]['am'], basis_dict[j]['idx_stride']  
        c3, exp3, atom3_idx, am3, dim3 = onp.asarray(basis_dict[k]['coef']), onp.asarray(basis_dict[k]['exp']), basis_dict[k]['atom'], basis_dict[k]['am'], basis_dict[k]['idx_stride']
        c4, exp4, atom4_idx, am4, dim4 = onp.asarray(basis_dict[l]['coef']), onp.asarray(basis_dict[l]['exp']), basis_dict[l]['atom'], basis_dict[l]['am'], basis_dict[l]['idx_stride']
    
        # Compute all primitive combinations of exponents and contraction coefficients for this shell 
        # Each column is i, j, k, l
        exp_combos = old_cartesian_product(exp1,exp2,exp3,exp4)
        # fuse the already fused normalization/ contraction coefficients together
        coeff_combos = onp.prod(old_cartesian_product(c1,c2,c3,c4), axis=1)

        # For every primitive, gather data necessary for the ERI shell computation. 
        # Lots of redundnacy here, since each primitive has same geometry. But this is the only way to get things into arrays. 
        # could sort them by contraction size and compute each one at a time, or pad them to make them the same size

        # TODO make function for permuting exponent and center indice data into canonical function forms psss psps ppss ppps pppp
        # have it return exponent list, center index list, have it take in
        am = onp.array([am1,am2,am3,am4])
        atom_idx_list = [atom1_idx,atom2_idx,atom3_idx,atom4_idx]
    
        # index in unique TEIs vector
        size = dim1 * dim2 * dim3 * dim4
        index = np.pad(np.arange(size) + place, (0, 81-size), constant_values=-1)
#        print(index)

        for contraction in range(exp_combos.shape[0]):
            basis_data.append([exp_combos[contraction,0],
                               exp_combos[contraction,1],
                               exp_combos[contraction,2],
                               exp_combos[contraction,3],
                               coeff_combos[contraction]])
            centers.append(atom_idx_list)
            angular_momenta.append(am)

            indices.append(index)

            #TODO temp
            count_primitives.append(3**onp.sum(am)) 

        place += size
    print('NUMBER OF PRIMITIVES', sum(count_primitives))
    return np.asarray(onp.asarray(basis_data)), centers, np.asarray(onp.asarray(angular_momenta)), np.asarray(onp.asarray(indices, dtype=np.int32))

#basis_data, centers, am  = preprocess(unique_shell_quartets, basis_dict)
    # Try passing redundant shells
basis_data, centers, angular_momenta, indices  = preprocess(unique_shell_quartets, basis_dict)

print(basis_data.shape)
print(indices.shape)
print(indices)

final_centers = np.take(geom, centers, axis=0)


#def build_tei(geom, centers, basis_data, am):
    #G = np.zeros_like((nbf,nbf,nbf,nbf)) # how do i even generate a vector of size nunique?
    # leading axis is number of primitives
#    centers = np.take(geom, centers, axis=0)

#    return 0 

# This does not work since it is jit compiling based on index i,which is abstract, and the static_argnum 
#@partial(jax.jit, static_argnums=(2,))
def general(basis_data, centers, am, indices):
    #A, B, C, D = centers
    #aa, bb, cc, dd, coeff = basis_data
    #args = (A, B, C, D, aa, bb, cc, dd, coeff)

    nprimitives = 7287 + 1
    #nclasses = 1296

    with loops.Scope() as s:
        s.G = np.zeros(nprimitives)
        # Loop over all primitives, do index_add
        for i in s.range(indices.shape[0]): # number of integral function passes
            A, B, C, D = centers[i]
            aa, bb, cc, dd, coeff = basis_data[i]
            args = (A, B, C, D, aa, bb, cc, dd, coeff)
            am = angular_momenta[i]
            # NOTE this is how you do 'else' analgoue: do 1 big cond_range check here, then indent the rest below?
            # for _ in s.cond_range( big bool check):
                # for _ in s.cond_range(am 0000):
                #   ...
                # for _ in s.cond_range(am 1000):


            for _ in s.cond_range(np.allclose(am,np.array([0,0,0,0]))):
                # have to convert result to array or np.pad doesnt do anything
                val = np.pad(np.array([eri_ssss(*args)]), (0,80), constant_values=0)
                #val = np.pad(eri_ssss(*args), (0,80), constant_values=0)

            for _ in s.cond_range(np.allclose(am,np.array([1,0,0,0]))):
                val = np.pad(eri_psss(*args).reshape(-1), (0,78), constant_values=0)

            for _ in s.cond_range(np.allclose(am,np.array([1,0,1,0]))):
                val = np.pad(eri_psps(*args).reshape(-1), (0,72), constant_values=0)

            for _ in s.cond_range(np.allclose(am,np.array([1,1,0,0]))):
                val = np.pad(eri_ppss(*args).reshape(-1), (0,72), constant_values=0)

            for _ in s.cond_range(np.allclose(am,np.array([1,1,1,0]))):
                val = np.pad(eri_ppps(*args).reshape(-1), (0,54), constant_values=0)

            for _ in s.cond_range(np.allclose(am,np.array([1,1,1,1]))):
                val = eri_pppp(*args).reshape(-1)

            #TODO fix cases such as 1000 0100 0010 0001, can be done preprocessing side! 
            #Just rearrange data packed into 'basis_data' and 'centers' appropriately
            s.G = jax.ops.index_add(s.G, indices[i], val)

            # indices mimics [val0, val1, ..., -1, -1, -1] 
            #for j in range(81): 
            #    s.G = jax.ops.index_add(s.G, indices[i,j], val[j])

            # This shoudl definitiely be right, no funny business
            #for j in s.range(indices.shape[1]):
            #    s.G = jax.ops.index_add(s.G, jax.ops.index[indices[i,j]], val[j])
        return s.G


#new_general = jax.jit(general, static_argnums=(2,))

#def debug(basis_data, centers, am, indices):
#    for i in range(10):
#        A, B, C, D = centers[i]
#        aa, bb, cc, dd, coeff = basis_data[i]
#        args = (A, B, C, D, aa, bb, cc, dd, coeff)
#        am = angular_momenta[i]
#
#        if np.allclose(am,np.array([0,0,0,0])):
#            val = np.pad(np.array([eri_ssss(*args)]), (0,80), constant_values=0)
#            print('S!',val)
#        elif np.allclose(am,np.array([1,0,0,0])):
#            val = np.pad(eri_psss(*args).reshape(-1), (0,78), constant_values=0)
#        elif np.allclose(am,np.array([1,0,1,0])):
#            val = np.pad(eri_psps(*args).reshape(-1), (0,72), constant_values=0)
#        elif np.allclose(am,np.array([1,1,0,0])):
#            val = np.pad(eri_ppss(*args).reshape(-1), (0,72), constant_values=0)
#        elif np.allclose(am,np.array([1,1,1,0])):
#            val = np.pad(eri_ppps(*args).reshape(-1), (0,54), constant_values=0)
#        elif np.allclose(am,np.array([1,1,1,1])):
#            val = eri_pppp(*args).reshape(-1)
#        else:
#            val = np.zeros(81)
#
#        #print('indices', indices[i])
#        #print('values',  val[i])
#        #print('values',  val)
#        #    s.G = jax.ops.index_add(s.G, jax.ops.index[indices[i]], val)
#
#G = debug(basis_data, final_centers, angular_momenta, indices)

#TODO
G = general(basis_data, final_centers, angular_momenta, indices)
print(G[:20])
print('last element of G')
print(G[:-1])

#for i in range(1296):
#    print(angular_momenta[i], end=' ')
#    print(general(basis_data[i], final_centers[i], angular_momenta[i]))
#    new_general(basis_data[i], final_centers[i], np.array([0,0,0,0]))
#    print('integral result is', general(basis_data, final_centers, am, i))

# jit with static argnums on am
# note: will recompile everytime it changes, I don't *think* it caches the different cases
# can raise an issue to request this as an option, would make life super easy. 
# Can take care of ordering with a simple argsort so that all of the same class appear at the same time.
#def test( am):
#    ssss = np.array([0,0,0,0])
#    
#    if am == 'ssss':
#    if am == [0,0,0,0]:
#    if am == ssss:  # least array instantiation
#
