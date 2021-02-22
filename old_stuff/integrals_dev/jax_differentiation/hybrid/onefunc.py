import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product
np.set_printoptions(linewidth=800, suppress=True, threshold=10000000)
from pprint import pprint
import time
from oei_s import * 
from oei_p import * 
from oei_d import * 
from oei_f import * 

molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         units bohr
                         """)

# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))
# Get Psi Basis Set and basis set dictionary objects
basis_name = 'cc-pvdz'
#basis_name = '6-31g'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
# Number of basis functions, number of shells
nbf = basis_set.nbf()
print("number of basis functions", nbf)
nshells = len(basis_dict)

key =  jax.random.PRNGKey(0)

# One function to rule them all. Super long compile time, hopefully quick execution time. 
# No padding, just redundant evaluations. Should affect computation time, but not gradient memory
# AIGHT this returns a parsing overflow error wtf. I guess parentheses nesting has a limit in Python/JAX
def primitive_map(basis_data, am):
    Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2 = basis_data
    args = (Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2)
    sgra = (Cx, Cy, Cz, Ax, Ay, Az, e2, e1, c2, c1)
    primitive =  np.where(am ==  0, overlap_ss(*args),
                 np.where(am ==  1, overlap_ps(*args)[0],
                 np.where(am ==  2, overlap_ps(*args)[1],
                 np.where(am ==  3, overlap_ps(*args)[2],
                 np.where(am ==  4, overlap_ps(*sgra)[0],
                 np.where(am ==  5, overlap_ps(*sgra)[1],
                 np.where(am ==  6, overlap_ps(*sgra)[2],
                 np.where(am ==  7, overlap_pp(*args)[0,0],
                 np.where(am ==  8, overlap_pp(*args)[0,1],
                 np.where(am ==  9, overlap_pp(*args)[0,2],
                 np.where(am == 10, overlap_pp(*args)[1,0],
                 np.where(am == 11, overlap_pp(*args)[1,1],
                 np.where(am == 12, overlap_pp(*args)[1,2],
                 np.where(am == 13, overlap_pp(*args)[2,0],
                 np.where(am == 14, overlap_pp(*args)[2,1],
                 np.where(am == 15, overlap_pp(*args)[2,2],0.0))))))))))))))))
    return primitive

def primitive_map2(basis_data,am):
    '''
    A general function to compute a single overlap primitive.

    Avoids parsing overflow error, while allow support for a buttload of integral types
    Can in prinicple map this over just basis_data, coef_data, and form contractions with it

    CAN THIS BE CONVERTED TO VECTOR FORMS SO THERES LESS FUNCTION CALLS?

    Once this is compiled, it takes the same amount of time as other function thats only over p's
    '''
    Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2 = basis_data
    args = (Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2)
    sgra = (Cx, Cy, Cz, Ax, Ay, Az, e2, e1, c2, c1)
    primitive1 = np.where(am ==  0, overlap_ss(*args), 0.0)

    primitive2 = np.where(am ==  1, overlap_ps(*args)[0],
                 np.where(am ==  2, overlap_ps(*args)[1],
                 np.where(am ==  3, overlap_ps(*args)[2],
                 np.where(am ==  4, overlap_ps(*sgra)[0],
                 np.where(am ==  5, overlap_ps(*sgra)[1],
                 np.where(am ==  6, overlap_ps(*sgra)[2], 0.0))))))

    primitive3 = np.where(am ==  7, overlap_pp(*args)[0,0],
                 np.where(am ==  8, overlap_pp(*args)[0,1],
                 np.where(am ==  9, overlap_pp(*args)[0,2],
                 np.where(am == 10, overlap_pp(*args)[1,0],
                 np.where(am == 11, overlap_pp(*args)[1,1],
                 np.where(am == 12, overlap_pp(*args)[1,2],
                 np.where(am == 13, overlap_pp(*args)[2,0],
                 np.where(am == 14, overlap_pp(*args)[2,1],
                 np.where(am == 15, overlap_pp(*args)[2,2],0.0)))))))))

    primitive4 = np.where(am == 16, overlap_ds(*args)[0],
                 np.where(am == 17, overlap_ds(*args)[1],
                 np.where(am == 18, overlap_ds(*args)[2],
                 np.where(am == 19, overlap_ds(*args)[3],
                 np.where(am == 20, overlap_ds(*args)[4],
                 np.where(am == 21, overlap_ds(*args)[5], 0.0))))))

    primitive4 = np.where(am == 22, overlap_ds(*sgra)[0],
                 np.where(am == 23, overlap_ds(*sgra)[1],
                 np.where(am == 24, overlap_ds(*sgra)[2],
                 np.where(am == 25, overlap_ds(*sgra)[3],
                 np.where(am == 26, overlap_ds(*sgra)[4],
                 np.where(am == 27, overlap_ds(*sgra)[5], 0.0))))))

    primitive5 = np.where(am == 28, overlap_dp(*args)[0,0],
                 np.where(am == 29, overlap_dp(*args)[0,1],
                 np.where(am == 30, overlap_dp(*args)[0,2],
                 np.where(am == 31, overlap_dp(*args)[1,0],
                 np.where(am == 32, overlap_dp(*args)[1,1],
                 np.where(am == 33, overlap_dp(*args)[1,2],
                 np.where(am == 34, overlap_dp(*args)[2,0],
                 np.where(am == 35, overlap_dp(*args)[2,1],
                 np.where(am == 36, overlap_dp(*args)[2,2],
                 np.where(am == 37, overlap_dp(*args)[3,0],
                 np.where(am == 38, overlap_dp(*args)[3,1],
                 np.where(am == 39, overlap_dp(*args)[3,2],
                 np.where(am == 40, overlap_dp(*args)[4,0],
                 np.where(am == 41, overlap_dp(*args)[4,1],
                 np.where(am == 42, overlap_dp(*args)[4,2],
                 np.where(am == 43, overlap_dp(*args)[5,0],
                 np.where(am == 44, overlap_dp(*args)[5,1],
                 np.where(am == 45, overlap_dp(*args)[5,2], 0.0))))))))))))))))))

    primitive6 = np.where(am == 46, overlap_dp(*sgra)[0,0],
                 np.where(am == 47, overlap_dp(*sgra)[0,1],
                 np.where(am == 48, overlap_dp(*sgra)[0,2],
                 np.where(am == 49, overlap_dp(*sgra)[1,0],
                 np.where(am == 50, overlap_dp(*sgra)[1,1],
                 np.where(am == 51, overlap_dp(*sgra)[1,2],
                 np.where(am == 52, overlap_dp(*sgra)[2,0],
                 np.where(am == 53, overlap_dp(*sgra)[2,1],
                 np.where(am == 54, overlap_dp(*sgra)[2,2],
                 np.where(am == 55, overlap_dp(*sgra)[3,0],
                 np.where(am == 56, overlap_dp(*sgra)[3,1],
                 np.where(am == 57, overlap_dp(*sgra)[3,2],
                 np.where(am == 58, overlap_dp(*sgra)[4,0],
                 np.where(am == 59, overlap_dp(*sgra)[4,1],
                 np.where(am == 60, overlap_dp(*sgra)[4,2],
                 np.where(am == 61, overlap_dp(*sgra)[5,0],
                 np.where(am == 62, overlap_dp(*sgra)[5,1],
                 np.where(am == 63, overlap_dp(*sgra)[5,2], 0.0))))))))))))))))))

    primitive = primitive1 + primitive2 + primitive3 + primitive4 + primitive5 + primitive6
    return primitive


# Map over the primitive axis so multiple primitives can be computed at once for a given angular momentum 
vectorized_primitives = jax.vmap(primitive_map, (0,None))
vectorized_primitives2 = jax.vmap(primitive_map2, (0,None))

# Create function which contracts a set of primitives
@jax.jit
def contraction_map(basis_data, am):
    '''basis_data = matrix of primitive data rows;  am = integer'''
    primitives = vectorized_primitives(basis_data, am)
    #primitives = vectorized_primitives2(basis_data, am)
    return np.sum(primitives)

# Create a function which maps over a set of many contractions (all contractions must be same size when input)
# So this function will be executed multiple times, once for each contraction size. 
# Suppose for all shell-pairs, contraction size K is 1,6,9,24,64. Then this function is run 5 times, once for each contraction size.
# You only need to sort by contraction size, not by integral type. Therefore, you should be able to just re-arrange the original dictionary
# so that the most contracted functions appear first. This would not match psi4 ordering. Also p-blocks and the like would be messed up.
# You would still need to index_update, yeah?
vectorized_contractions = jax.vmap(contraction_map, (0, 0))


bd = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0]])

bd2 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0]])

bd3 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0]])

prim = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,1.0,1.0,1.0])



# This loop could loop over basis data combos and return contracted integrals.
#for i in range(10000):
#    contraction_map(bd, 0)
for i in range(1):
    contraction_map(bd, 0)

arrs = [bd, bd2, bd3]
homo_arrs = [bd, bd, bd]

#import time

# Map over all contracted integrals of the same contraction order  

#a = time.time()
#print('generating arrays')
#big_data1 = np.asarray(onp.asarray([bd] * 1000000))
#big_data2 = np.asarray(onp.asarray([bd2] * 1000000))
#big_data3 = np.asarray(onp.asarray([bd3] * 1000000))
#indices = np.repeat(0, 1000000)
#print('arrays generated')
#b = time.time()
#print(b-a)
#
#
#print("compiling")
#indices = np.repeat(0, 2)
#big_data1 = np.asarray(onp.asarray([bd] * 2))
#res = vectorized_contractions(big_data1, indices)
#print("compiling done")
#res = vectorized_contractions(big_data2, indices)
#res = vectorized_contractions(big_data3, indices)
#c = time.time()
#print(c-b)

#res = jax.lax.map(contraction_map, (big_data, np.repeat(5,1500)))
#res = jax.lax.map(contraction_map, big_data)
#print(res.shape)


#for i in range(100000):
#    contraction_map(bd, 0)



#print('compiling')
#print(contraction_map(bd, 0))
#print(contraction_map(bd, 1))
#print(contraction_map(bd, 2))
#print(contraction_map(bd, 15))

#r = jax.random.randint(key, (1,), 0, 15)

#print(r[0])

#test = jax.vmap(contraction_map, (0,None))  

#test((bd,bd,bd), 2)




#for i in range(10000):
    #r = jax.random.randint(key, (1,), 0, 15)
    #contraction_map(arrs[0], r[0])
    #contraction_map(arrs[1], r[0])
    #contraction_map(arrs[2], r[0])

    #r = jax.random.randint(key, (1,), 0, 15)[0]
    #idx = jax.random.randint(key, (1,), 0, 2)[0]

    #contraction_map(arrs[0], 15)

#    contraction_map(arrs[2], 1)


#print('compiling done')



    
    




