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
max_prim = basis_set.max_nprimitive()
biggest_K = max_prim**4
pprint(basis_dict)
nbf = basis_set.nbf()
nshells = len(basis_dict)
#unique_shell_quartets = find_unique_shells(nshells)

shell_quartets = old_cartesian_product(np.arange(nshells), np.arange(nshells), np.arange(nshells), np.arange(nshells))

print("number of basis functions", nbf)
print("number of shells ", nshells)
print("number of shell quartets", shell_quartets.shape[0])
print("Max primitives: ", max_prim)
print("Biggest contraction: ", biggest_K)

def preprocess(shell_quartets, basis_dict):
    basis_data = [] # A list of every primitive exponent, coeff of 4 centers
    centers = []
    angular_momenta = [] # list of string identifiers '1010' etc
    count_primitives = []
    indices = []
    count_unique_teis = []
    place = 0
    for quartet in shell_quartets:
        i,j,k,l = quartet
        c1, exp1, atom1_idx, am1, idx1, size1 = onp.asarray(basis_dict[i]['coef']), onp.asarray(basis_dict[i]['exp']), basis_dict[i]['atom'], basis_dict[i]['am'], basis_dict[i]['idx'], basis_dict[i]['idx_stride']
        c2, exp2, atom2_idx, am2, idx2, size2 = onp.asarray(basis_dict[j]['coef']), onp.asarray(basis_dict[j]['exp']), basis_dict[j]['atom'], basis_dict[j]['am'], basis_dict[j]['idx'], basis_dict[j]['idx_stride']  
        c3, exp3, atom3_idx, am3, idx3, size3 = onp.asarray(basis_dict[k]['coef']), onp.asarray(basis_dict[k]['exp']), basis_dict[k]['atom'], basis_dict[k]['am'], basis_dict[k]['idx'], basis_dict[k]['idx_stride']
        c4, exp4, atom4_idx, am4, idx4, size4 = onp.asarray(basis_dict[l]['coef']), onp.asarray(basis_dict[l]['exp']), basis_dict[l]['atom'], basis_dict[l]['am'], basis_dict[l]['idx'], basis_dict[l]['idx_stride']
    
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

        # NOTE
        # alright this ones a doozy. We compute all the indices of each dimension, pack them up, but then pad them so they can stack into an array 
        # and the junk values are added at the -1 index.

        am = onp.array([am1,am2,am3,am4])
        atom_idx_list = [atom1_idx,atom2_idx,atom3_idx,atom4_idx]

        indices1 = onp.repeat(idx1, size1) + onp.arange(size1)
        indices2 = onp.repeat(idx2, size2) + onp.arange(size2)
        indices3 = onp.repeat(idx3, size3) + onp.arange(size3)
        indices4 = onp.repeat(idx4, size4) + onp.arange(size4)
        index = old_cartesian_product(indices1, indices2, indices3, indices4)
        print(index)

        size = size1 * size2 * size3 * size4
        index = onp.pad(index, ((0,81-size),(0,0)), mode='constant', constant_values=-1)

        # Okay, now we have to do the padded contractions thing rather than looping over every primitive and blowing up our array
        # Remember to call np.where(not 0, compute_eri) down below
        current_K = exp_combos.shape[0] # Size of this contraction
        K = biggest_K - current_K
        exp_combos = onp.pad(exp_combos, ((0,K), (0,0)))
        coeff_combos = onp.pad(coeff_combos, (0,K))
        basis_data.append(np.hstack((exp_combos,coeff_combos.reshape(-1,1))))

        indices.append(index)
        angular_momenta.append(am)
        centers.append(atom_idx_list)

    return np.asarray(onp.asarray(basis_data)), centers, np.asarray(onp.asarray(angular_momenta)), np.asarray(onp.asarray(indices, dtype=np.int32))

#basis_data, centers, am  = preprocess(unique_shell_quartets, basis_dict)
basis_data, centers, angular_momenta, indices  = preprocess(shell_quartets, basis_dict)
print("SHAPES")
print('basis data',basis_data.shape)
print('centers',len(centers))
print('am',angular_momenta.shape)
print('indices',indices.shape)

# This does not work since it is jit compiling based on index i,which is abstract, and the static_argnum 
#@partial(jax.jit, static_argnums=(2,))
def general(geom, basis_data, center_indices, am, indices):
    centers = np.take(geom, center_indices, axis=0)

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

            for _ in s.cond_range(np.allclose(am,np.array([0,0,0,0]))):
                val = np.pad(np.array([eri_ssss(*args)]), (0,80), constant_values=0)

            for _ in s.cond_range(np.allclose(am,np.array([1,0,0,0]))):
                val = np.pad(eri_psss(*args).reshape(-1), (0,78), constant_values=0)
            for _ in s.cond_range(np.allclose(am,np.array([0,0,0,1]))): 
                val = np.pad(eri_psss(*args).reshape(-1), (0,78), constant_values=0)
            for _ in s.cond_range(np.allclose(am, np.array([0,0,1,0]))):
                val = np.pad(eri_psss(*args).reshape(-1), (0,78), constant_values=0)
            for _ in s.cond_range(np.allclose(am, np.array([0,1,0,0]))):
                val = np.pad(eri_psss(*args).reshape(-1), (0,78), constant_values=0)

            for _ in s.cond_range(np.allclose(am,np.array([1,0,1,0]))):
                val = np.pad(eri_psps(*args).reshape(-1), (0,72), constant_values=0)
            for _ in s.cond_range(np.allclose(am,np.array([0,1,0,1]))):
                val = np.pad(eri_psps(*args).reshape(-1), (0,72), constant_values=0)
            for _ in s.cond_range(np.allclose(am,np.array([0,1,1,0]))):
                val = np.pad(eri_psps(*args).reshape(-1), (0,72), constant_values=0)
            for _ in s.cond_range(np.allclose(am,np.array([1,0,0,1]))):
                val = np.pad(eri_psps(*args).reshape(-1), (0,72), constant_values=0)

            for _ in s.cond_range(np.allclose(am,np.array([1,1,0,0]))):
                val = np.pad(eri_ppss(*args).reshape(-1), (0,72), constant_values=0)
            for _ in s.cond_range(np.allclose(am,np.array([0,0,1,1]))):
                val = np.pad(eri_ppss(*args).reshape(-1), (0,72), constant_values=0)

            for _ in s.cond_range(np.allclose(am,np.array([1,1,1,0]))):
                val = np.pad(eri_ppps(*args).reshape(-1), (0,54), constant_values=0)
            for _ in s.cond_range(np.allclose(am, np.array([0,1,1,1]))):
                val = np.pad(eri_ppps(*args).reshape(-1), (0,54), constant_values=0)
            for _ in s.cond_range(np.allclose(am, np.array([1,0,1,1]))):
                val = np.pad(eri_ppps(*args).reshape(-1), (0,54), constant_values=0)
            for _ in s.cond_range(np.allclose(am, np.array([1,1,0,1]))):
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

##TODO
#G = general(geom, basis_data, centers, angular_momenta, indices)
#
#k = 0 
#for i in G:
#    print(i)
#
#print('last element of G')
#print(G[:-1])

#for i in range(1296):
#    print(angular_momenta[i], end=' ')
#    print(general(basis_data[i], final_centers[i], angular_momenta[i]))
#    new_general(basis_data[i], final_centers[i], np.array([0,0,0,0]))
#    print('integral result is', general(basis_data, final_centers, am, i))


def debug(basis_data, center_indices, am, indices):

    G = np.zeros(2024)
    centers = np.take(geom, center_indices, axis=0)
    for i in range(2000):
        A, B, C, D = centers[i]
        aa, bb, cc, dd, coeff = basis_data[i] # this is 81 x 5, so each thing is a vector of 81 
        args = (A, B, C, D, aa, bb, cc, dd, coeff)
        am = angular_momenta[i]

        if np.allclose(am,np.array([0,0,0,0])):
            val = np.pad(np.array([eri_ssss(*args)]), (0,80), constant_values=0)
            print('ssss')


        # ONE p
        elif np.allclose(am,np.array([1,0,0,0])):
            val = np.pad(eri_psss(*args).reshape(-1), (0,78), constant_values=0)
            print('psss')
        elif np.allclose(am, np.array([0,1,0,0])):
            print('spss')
            val = np.pad(eri_psss(*args).reshape(-1), (0,78), constant_values=0)
        elif np.allclose(am, np.array([0,0,1,0])):
            print('ssps')
            val = np.pad(eri_psss(*args).reshape(-1), (0,78), constant_values=0)
        elif np.allclose(am,np.array([0,0,0,1])): 
            print('sssp')
            val = np.pad(eri_psss(*args).reshape(-1), (0,78), constant_values=0)

        # TWO p's
        elif np.allclose(am,np.array([1,1,0,0])):
            print('ppss')
            val = np.pad(eri_ppss(*args).reshape(-1), (0,72), constant_values=0)
        elif np.allclose(am,np.array([1,0,1,0])):
            print('psps')
            val = np.pad(eri_psps(*args).reshape(-1), (0,72), constant_values=0)
        elif np.allclose(am,np.array([0,1,0,1])):
            print('spsp')
            val = np.pad(eri_psps(*args).reshape(-1), (0,72), constant_values=0)
        elif np.allclose(am,np.array([0,1,1,0])):
            print('spps')
            val = np.pad(eri_psps(*args).reshape(-1), (0,72), constant_values=0)
        elif np.allclose(am,np.array([1,0,0,1])):
            print('pssp')
            val = np.pad(eri_psps(*args).reshape(-1), (0,72), constant_values=0)
        elif np.allclose(am,np.array([0,0,1,1])):
            print('sspp')
            val = np.pad(eri_psps(*args).reshape(-1), (0,72), constant_values=0)

        # THREE p's
        elif np.allclose(am,np.array([1,1,1,0])):
            print('ppps')
            val = np.pad(eri_ppps(*args).reshape(-1), (0,54), constant_values=0)
        elif np.allclose(am, np.array([1,0,1,1])):
            print('pspp')
            val = np.pad(eri_ppps(*args).reshape(-1), (0,54), constant_values=0)
        elif np.allclose(am, np.array([1,1,0,1])):
            print('ppsp')
            val = np.pad(eri_ppps(*args).reshape(-1), (0,54), constant_values=0)
        elif np.allclose(am, np.array([0,1,1,1])):
            print('sppp')
            val = np.pad(eri_ppps(*args).reshape(-1), (0,54), constant_values=0)

        # FOUR p's
        elif np.allclose(am,np.array([1,1,1,1])):
            print('pppp')
            val = eri_pppp(*args).reshape(-1)
        else:
            print("NOPE!")
            val = np.zeros(81)

        #print('indices', indices[i])
        #print('values',  val[i])
        #print('values',  val)
        print(indices[i])
        G = jax.ops.index_add(G, jax.ops.index[indices[i]], val)
    return G

#G = debug(basis_data, centers, angular_momenta, indices)
#for i in G:
#    print(i)

mints = psi4.core.MintsHelper(basis_set)
psi_G = np.asarray(onp.asarray(mints.ao_eri()))
#print('Size of Psi G', psi_G.shape)
##print(np.allclose(S, psi_S))
## Print psi4 G values
#
#print(find_unique_shells(nbf).shape)

psi_G_vec = []
for i in range(nbf):
    for j in range(nbf):
        for k in range(nbf):
            for l in range(nbf):
                if i>=j and k>=l and (i*(i+1)/2 + j >= k*(k+1)/2 + l): # thanks Crawford
                    psi_G_vec.append(psi_G[i,j,k,l])

#for i in range(nbf):
#    for j in range(nbf):
#        for k in range(nbf):
#            for l in range(nbf):
#                print((psi_G[i,j,k,l]))
#

unique_tei_psi = onp.asarray(psi_G_vec)
#print(unique_tei_psi)
#for i in unique_tei_psi:
#    print(i)
print(unique_tei_psi.shape)

#print(G[0:100])
#print(G[-100:])
