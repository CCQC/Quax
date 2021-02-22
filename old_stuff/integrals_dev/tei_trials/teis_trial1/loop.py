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
# Pytree test
#value_flat, value_tree = jax.tree_util.tree_flatten(basis_dict)
#print(value_flat)


max_prim = basis_set.max_nprimitive()
print(max_prim)
biggest_K = max_prim**4
#pprint(basis_dict)
nbf = basis_set.nbf()
nshells = len(basis_dict)
#unique_shell_quartets = find_unique_shells(nshells)

shell_quartets = old_cartesian_product(np.arange(nshells), np.arange(nshells), np.arange(nshells), np.arange(nshells))


def transform_basisdict(basis_dict, max_prim):
    '''Make it so all contractions are the same size in the basis dict by padding exp and coef values to 0 and 0?
    This allows you to pack them neatly into an array, and then worry about redundant computation later.

    '''
    new_dict = basis_dict.copy()
    for i in range(len(basis_dict)):
        current_exp = onp.asarray(basis_dict[i]['exp'])
        new_dict[i]['exp'] = np.asarray(onp.pad(current_exp, (0, max_prim - current_exp.shape[0])))
        current_coef = onp.asarray(basis_dict[i]['coef'])
        new_dict[i]['coef'] = np.asarray(onp.pad(current_coef, (0, max_prim - current_coef.shape[0])))
    return new_dict


#TODO this is incorrect, mixes 0's and real values together, not what you want
basis_dict = transform_basisdict(basis_dict, max_prim)

#print("number of basis functions", nbf)
#print("number of shells ", nshells)
#print("number of shell quartets", shell_quartets.shape[0])
#print("Max primitives: ", max_prim)
#print("Biggest contraction: ", biggest_K)

def preprocess(shell_quartets, basis_dict):
    coeffs = []
    exps = []
    atoms = []
    ams = []
    indices = []
    sizes = []
    for i in range(nshells):
        c1, exp1, atom1_idx, am1, idx1, size1 = onp.asarray(basis_dict[i]['coef']), onp.asarray(basis_dict[i]['exp']), basis_dict[i]['atom'], basis_dict[i]['am'], basis_dict[i]['idx'], basis_dict[i]['idx_stride']
        coeffs.append(c1)
        exps.append(exp1)
        atoms.append(atom1_idx)
        ams.append(am1)
        indices.append(idx1)
        sizes.append(size1)

        
                                                          #TODO hard coded, needs to subtract from largest am size (idx_stride)
        #INDICES = onp.pad((onp.repeat(idx1, size1) + onp.arange(size1)), (0,3-size1), constant_values=-1)
        #print(INDICES)
        

    return np.asarray(coeffs), np.asarray(exps), np.asarray(atoms), np.asarray(ams), np.asarray(indices), np.asarray(sizes)

def get_indices(shell_quartets, basis_dict):
    '''
    Get all indices of ERIs in (nbf**4,4) array. 
    Record where each shell quartet starts and stops along the first axis of this index array.
    '''
    all_indices = []
    starts = []
    stops = []
    start = 0
    for i in range(nshells):
        idx1, size1 = basis_dict[i]['idx'], basis_dict[i]['idx_stride']
        indices1 = onp.repeat(idx1, size1) + onp.arange(size1)
        for j in range(nshells):
            idx2, size2 = basis_dict[j]['idx'], basis_dict[j]['idx_stride']
            indices2 = onp.repeat(idx2, size2) + onp.arange(size2)
            for k in range(nshells):
                idx3, size3 = basis_dict[k]['idx'], basis_dict[k]['idx_stride']
                indices3 = onp.repeat(idx3, size3) + onp.arange(size3)
                for l in range(nshells):
                    idx4, size4 = basis_dict[l]['idx'], basis_dict[l]['idx_stride']
                    indices4 = onp.repeat(idx4, size4) + onp.arange(size4)
                    indices = old_cartesian_product(indices1,indices2,indices3,indices4)
                    all_indices.append(indices)
                    
                    stop = start + indices.shape[0] #how much of the indices array this integral takes up
                    starts.append(start)
                    stops.append(stop)
                    start += indices.shape[0]
                    # this would be in the same order as that which appears in the JAX loop, 
                    # so theoretically could just stack them along the index axis?   

    # NOTE there may be an issue at the ending index 'stop' point, might complain about going out of range
    final_indices = np.asarray(onp.vstack(all_indices))
    starts = np.asarray(onp.asarray(starts))
    stops = np.asarray(onp.asarray(stops))
    return final_indices, starts, stops


        
coeffs, exps, atoms, am, indices, sizes = preprocess(shell_quartets, basis_dict)
print(indices.shape)

new_indices, starts, stops = get_indices(shell_quartets, basis_dict)
print(new_indices.shape)

starts = starts.reshape(nshells,nshells,nshells,nshells)
stops = stops.reshape(nshells,nshells,nshells,nshells)


#print(starts.shape)
#print(stops.shape)
#for i in range(starts.shape[0]):
#    print(new_indices[starts[i]:stops[i]].shape)
#print(starts)

#print(indices.dtype)
#print(sizes.dtype)
#print("coeffs", coeffs.shape)
#print("exps", exps.shape)
#print("atoms", atoms.shape)
#print("am", am.shape)
#print("indices", indices.shape)
#print("sizes", sizes.shape)
#print(am)

#print(exps[1])

#print(cartesian_product(exps[1],exps[1],exps[0],exps[0]))
#print(am)
#print('angular momentum of the 5th shell quartet')
#print(np.array([am[0,5], am[1,5], am[2,5], am[3,5]]))


def compute(geom, coeffs, exps, atoms, am, indices, sizes):
    #dim_indices = np.repeat(indices, sizes) + np.arange(sizes)

    with loops.Scope() as s:
        def primitive(A, B, C, D, aa, bb, cc, dd, coeff, am):
            '''Geometry parameters, exponents, coefficients, angular momentum identifier'''
            args = (A, B, C, D, aa, bb, cc, dd, coeff) 
            # Since we had to pad all coefficients and exponents to be the same size to use JAX functions,
            # we only compute the integral if the coefficient is not 0, otherwise we return 0, and it is effectively a 0-contraction component
            with loops.Scope() as S:
                #TODO
                for _ in S.cond_range(True):
                    primitive = np.where(coeff == 0,  0.0, eri_ssss(*args)).reshape(-1)
                #TODO


                for _ in S.cond_range(np.allclose(am,np.array([0,0,0,0]))):
                    primitive = np.where(coeff == 0,  0.0, eri_ssss(*args)).reshape(-1)


                #TEMP TODO
                for _ in S.cond_range(np.allclose(am,np.array([1,0,0,0]))):
                    primitive = np.where(coeff == 0,  0.0, eri_psss(*args)).reshape(-1)
                for _ in S.cond_range(np.allclose(am,np.array([0,1,0,0]))): # WRONG TODO
                    primitive = np.where(coeff == 0,  0.0, eri_psss(*args)).reshape(-1)
                for _ in S.cond_range(np.allclose(am,np.array([0,0,1,0]))): # WRONG TODO
                    primitive = np.where(coeff == 0,  0.0, eri_psss(*args)).reshape(-1)
                for _ in S.cond_range(np.allclose(am,np.array([0,0,0,1]))): # WRONG TODO
                    primitive = np.where(coeff == 0,  0.0, eri_psss(*args)).reshape(-1)

                for _ in S.cond_range(np.allclose(am,np.array([1,1,0,0]))):
                    primitive = np.where(coeff == 0,  0.0, eri_ppss(*args)).reshape(-1)
                for _ in S.cond_range(np.allclose(am,np.array([0,1,1,0]))): #WRONG TODO 
                    primitive = np.where(coeff == 0,  0.0, eri_ppss(*args)).reshape(-1)
                for _ in S.cond_range(np.allclose(am,np.array([0,0,1,1]))): #WRONG TODO 
                    primitive = np.where(coeff == 0,  0.0, eri_ppss(*args)).reshape(-1)

                for _ in S.cond_range(np.allclose(am,np.array([1,0,1,0]))):
                    primitive = np.where(coeff == 0,  0.0, eri_psps(*args)).reshape(-1)
                for _ in S.cond_range(np.allclose(am,np.array([0,1,0,1]))): #WRONG TODO
                    primitive = np.where(coeff == 0,  0.0, eri_psps(*args)).reshape(-1)
                for _ in S.cond_range(np.allclose(am,np.array([1,0,0,1]))): #WRONG TODO
                    primitive = np.where(coeff == 0,  0.0, eri_psps(*args)).reshape(-1)

                for _ in S.cond_range(np.allclose(am,np.array([1,1,1,0]))):
                    primitive = np.where(coeff == 0,  0.0, eri_ppps(*args)).reshape(-1)
                for _ in S.cond_range(np.allclose(am,np.array([1,1,0,1]))): #WRONG TODO
                    primitive = np.where(coeff == 0,  0.0, eri_ppps(*args)).reshape(-1)
                for _ in S.cond_range(np.allclose(am,np.array([1,0,1,1]))): #WRONG TODO
                    primitive = np.where(coeff == 0,  0.0, eri_ppps(*args)).reshape(-1)
                for _ in S.cond_range(np.allclose(am,np.array([0,1,1,1]))): #WRONG TODO
                    primitive = np.where(coeff == 0,  0.0, eri_ppps(*args)).reshape(-1)

                for _ in S.cond_range(np.allclose(am,np.array([1,1,1,1]))):
                    primitive = np.where(coeff == 0,  0.0, eri_pppp(*args)).reshape(-1)
                return primitive
    
        # Computes multiple primitives with same center, angular momentum 
        vectorized_primitive = jax.vmap(primitive, (None,None,None,None,0,0,0,0,0,None))

        ## Computes a contracted integral 
        #@jax.jit
        def contraction(A, B, C, D, aa, bb, cc, dd, coeff, am):
            primitives = vectorized_primitive(A, B, C, D, aa, bb, cc, dd, coeff, am)
            return np.sum(primitives, axis=0)

        #TEMP TODO until you figure out index shiz
        #s.G = np.zeros((nshells,nshells,nshells,nshells))
        # create with a 'dump' dimension because of index packing and creation issue

        indx_array = np.arange(nshells**4).reshape(nshells,nshells,nshells,nshells) 
        s.G = np.zeros((nbf,nbf,nbf,nbf))
        idx_vec = np.arange(nbf)
        for i in s.range(nshells):
            A = geom[atoms[i]]
            aa = exps[i]
            c1 = coeffs[i]
            ami = am[i]
            idx1 = indices[i]
            size1 = sizes[i]
            for j in s.range(nshells):
                B = geom[atoms[j]]
                bb = exps[j]
                c2 = coeffs[j]
                amj = am[j]
                idx2 = indices[j]
                size2 = sizes[j]
                for k in s.range(nshells):
                    C = geom[atoms[k]]
                    cc = exps[k]
                    c3 = coeffs[k]
                    amk = am[k]
                    idx3 = indices[k]
                    size3 = sizes[k]
                    for l in s.range(nshells):
                        D = geom[atoms[l]]
                        dd = exps[l]
                        c4 = coeffs[l]
                        aml = am[l]
                        idx4 = indices[l]
                        size4 = sizes[l]

                        exp_combos = cartesian_product(aa,bb,cc,dd)
                        coeff_combos = np.prod(cartesian_product(c1,c2,c3,c4), axis=1)
                        am_vec = np.array([ami, amj, amk, aml]) 
                        val = contraction(A,B,C,D, 
                                          exp_combos[:,0], 
                                          exp_combos[:,1], 
                                          exp_combos[:,2],
                                          exp_combos[:,3],
                                          coeff_combos, am_vec)

                        s.G = jax.ops.index_update(s.G, (index_combos[:,0], index_combos[:,1], index_combos[:,2], index_combos[:,3]), np.pad(val, (0,81 - val.shape[0])))

                        # no other way out, have to pad???

                        #s.G = np.concatenate((s.G, val.reshape(-1)))
                        #s.G = jax.ops.index_update(s.G, jax.ops.index[i,j,k,l], val.reshape(-1)[0])

                        # CAN GET COUNT WITHOUT COUNTING, USING RAVEL
                        #count = np.ravel_multi_index([i,j,k,l], (nbf,nbf,nbf,nbf))
                        #place = indx_array[i,j,k,l]
                        #start = starts[place]
                        #stop = stops[place]
                        #index = new_indices[start:stop] # replace with dynamic slice
                        #index = jax.lax.dynamic_slice(new_indices, [start,0], [val.shape[0],4])
                        # This only slices one row of new_indices
                        #index = jax.lax.dynamic_index_in_dim(new_indices, start)
                        #s.G = jax.ops.index_update(s.G, (index[:,0], index[:,1], index[:,2], index[:,3]), val.reshape(-1))


                        # These indices are fine, just val isn't matching the indices. THIS MAY WORK, shaping bug, but not yelling at you. I think you need to fix contraction function
                        # The count is porbably the issue, the computation is asynchronous so...
                        #index = new_indices[starts[count]:stops[count]]
                        #slic = jax.lax.dynamic_slice(new_indices, 
                        #s.G = jax.ops.index_update(s.G, (index[:,0], index[:,1], index[:,2], index[:,3]), val.reshape(-1))
                        #s.G = jax.ops.index_update(s.G, jax.ops.index[index[:,0], index[:,1], index[:,2], index[:,3]], val.reshape(-1))
                        #count += 1


                        # same error, static start stop
                        #index = new_indices[starts[i,j,k,l]:stops[i,j,k,l]]
                        #s.G = jax.ops.index_update(s.G, jax.ops.index[index[:,0], index[:,1], index[:,2], index[:,3]], val.reshape(-1))

                        # Most promising: USE SHAPE OF VAL TO INFORM INDICES???
                        #indices1 = np.repeat(idx1, val.shape[0]) + np.arange(val.shape[0])
                        #indices2 = np.repeat(idx2, val.shape[1]) + np.arange(val.shape[1])
                        #print(indices1)

                        #blah = np.repeat(np.array([idx1,idx2,idx3,idx4]), val.shape) + np.arange(val.shape[0])
                        #print(blah)
    

                        # test whether indices can be abstract 
                        # This works because all val's are broadcastable to the indices (3,3,3,3)
                        #fake = np.array([idx1,idx2,idx3])
                        #fake = np.array([counti,countj,countk])

                        #s.G = jax.ops.index_update(s.G, (fake,fake,fake,fake), val)

                        #s.G = jax.ops.index_update(s.G, (idx_vec[counti:counti+size1],idx_vec[countj:countj+size2],idx_vec[countk:countk+size3],idx_vec[countl:countl+size4]), val)
                        # IDK this may still work, getting a NAN for some reason

                        #i = 0
                        #for v in val.flatten():
                        #    s.G = jax.ops.index_update(s.G, (idx_vec[counti],idx_vec[countj],idx_vec[countk],idx_vec[countl]), v)
                        #    #s.G = jax.ops.index_update(s.G, (idx_vec[idx1:idx1+i],idx_vec[idx2],idx_vec[idx3],idx_vec[idx3]), v)
                        #    i+= 1
                        #    #counti += 1
                        #    #countj += 1
                        #    #countk += 1
                        #    #countl += 1
                        #for a1 in range(size1):
                        #    for a2 in range(size2):
                        #        for a3 in range(size3):
                        #            for a4 in range(size4):
                        #                i+=1
                


                        #counti += size1
                        #countj += size2
                        #countk += size3
                        #countl += size4
    
                        
                        #s.G = jax.ops.index_update(s.G, (fake,fake,fake,fake), val)

                        #print(val)
                        # index handling
                        #size = 3**np.sum(am_vec)
                        #indices1 = np.tile(np.array([idx1]), np.array([size1])) + np.arange(size1)
                        #indices2 = np.tile(np.array([idx2]), np.array([size2])) + np.arange(size2)
                        #indices3 = np.tile(np.array([idx3]), np.array([size3])) + np.arange(size3)
                        #indices4 = np.tile(np.array([idx4]), np.array([size4])) + np.arange(size4)
                        # Get indices 
                        #indices1 = np.arange(size1)
                        #indices2 = np.arange(size2)
                        #indices3 = np.arange(size3)
                        #indices4 = np.arange(size4)
                        #index = cartesian_product(indices1, indices2, indices3, indices4)

                        #s.G = jax.ops.index_update(s.G, (index[:,0],index[:,1],index[:,2],index[:,3]), val)

                        #s.G = jax.ops.index_update(s.G, jax.ops.index[i,j,k,l], val)


                        #val = np.sum(A) * np.sum(B) * np.sum(C) * np.sum(D)
                        #s.G = jax.ops.index_update(s.G, jax.ops.index[i,j,k,l], val)

        return s.G


G = compute(geom, coeffs, exps, atoms, am, indices, sizes)
#print(G)
for i in G.flatten()[:100]:
    print(i)
print(G[0,0,0,0])
#
#mints = psi4.core.MintsHelper(basis_set)
#psi_G = np.asarray(onp.asarray(mints.ao_eri()))
#print(psi_G)

