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


basis_dict = transform_basisdict(basis_dict, max_prim)

#print("number of basis functions", nbf)
#print("number of shells ", nshells)
#print("number of shell quartets", shell_quartets.shape[0])
#print("Max primitives: ", max_prim)
#print("Biggest contraction: ", biggest_K)

def preprocess(shell_quartets, basis_dict):

    ci, cj, ck, cl = [], [], [], []
    expi, expj, expk, expl, = [], [], [], []
    atomi, atomj, atomk, atoml = [], [], [], []
    ami, amj, amk, aml = [], [], [], []
    #indices = []

    for i in range(nshells):
        c1, exp1, atom1_idx, am1, idx1, size1 = onp.asarray(basis_dict[i]['coef']), onp.asarray(basis_dict[i]['exp']), basis_dict[i]['atom'], basis_dict[i]['am'], basis_dict[i]['idx'], basis_dict[i]['idx_stride']
        for j in range(nshells):
            c2, exp2, atom2_idx, am2, idx2, size2 = onp.asarray(basis_dict[j]['coef']), onp.asarray(basis_dict[j]['exp']), basis_dict[j]['atom'], basis_dict[j]['am'], basis_dict[j]['idx'], basis_dict[j]['idx_stride']  
            for k in range(nshells):
                c3, exp3, atom3_idx, am3, idx3, size3 = onp.asarray(basis_dict[k]['coef']), onp.asarray(basis_dict[k]['exp']), basis_dict[k]['atom'], basis_dict[k]['am'], basis_dict[k]['idx'], basis_dict[k]['idx_stride']
                for l in range(nshells):
                    c4, exp4, atom4_idx, am4, idx4, size4 = onp.asarray(basis_dict[l]['coef']), onp.asarray(basis_dict[l]['exp']), basis_dict[l]['atom'], basis_dict[l]['am'], basis_dict[l]['idx'], basis_dict[l]['idx_stride']

                    ci.append(c1)
                    cj.append(c2)
                    ck.append(c3)
                    cl.append(c4)
                    expi.append(exp1)
                    expj.append(exp2)
                    expk.append(exp3)
                    expl.append(exp4)
                    atomi.append(atom1_idx)
                    atomj.append(atom2_idx)
                    atomk.append(atom3_idx)
                    atoml.append(atom4_idx)
                    ami.append(am1)
                    amj.append(am2)
                    amk.append(am3)
                    aml.append(am4)


    # Each of these structures has shape (4, shell_quartets, max number of primitives in contraction)
    coeffs = onp.asarray([ci, cj, ck, cl])
    exps = onp.asarray([expi, expj, expk, expl])
    # atoms has shape (4, shell_quartets)
    atoms = onp.asarray([atomi, atomj, atomk, atoml])
    # am has shape (4, shell_quartets)
    am = onp.asarray([ami, amj, amk, aml])
    return np.asarray(coeffs), np.asarray(exps), np.asarray(atoms), np.asarray(am)

        
coeffs, exps, atoms, am = preprocess(shell_quartets, basis_dict)
print(am.shape)
print(am)


def compute(geom, coeffs, exps, atoms, am):
    with loops.Scope() as s:
        #def primitive(A, B, C, D, aa, bb, cc, dd, coeff, am):
        #    '''Geometry parameters, exponents, coefficients, angular momentum identifier'''
        #    args = (A, B, C, D, e1, e2, c1, c2)
        #    primitive =  np.where(e1 ==  0, 0.0,
        #                 np.where(am ==  0, overlap_ss(*args), 0.0))
        #    return primitive
        # Computes multiple primitive ss overlaps with same center, angular momentum 
        #vectorized_primitive = jax.vmap(primitive, (None,None,None,None,None,None,0,0,0,0,None))

        ## Computes a contracted ss overlap 
        #@jax.jit
        #def contraction(Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2):
        #    primitives = vectorized_primitive(Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2, 0)
        #    return np.sum(primitives)


        # Just collect 1d arrays for each shell's coefficient, exponent, am, index, size
        # Wouldh avet ocall a cartesian product within the loop
        # The indices would need to be padded i believe in order to get them into an array
        # or just compute indices in the loop? 

        #TEMP
        s.G = np.zeros((nshells,nshells,nshells,nshells))
        #s.G = np.zeros((nbf,nbf,nbf,nbf))
        for i in s.range(nshells):
            A = geom[atoms[0,i]]
            for j in s.range(nshells):
                B = geom[atoms[1,j]]
                for k in s.range(nshells):
                    C = geom[atoms[2,k]]
                    for l in s.range(nshells):
                        D = geom[atoms[3,l]]
                        val = np.sum(A) * np.sum(B) * np.sum(C) * np.sum(D)
                        s.G = jax.ops.index_update(s.G, jax.ops.index[i,j,k,l], val)

        return s.G


G = compute(geom, coeffs, exps, atoms, am)
print(G)




