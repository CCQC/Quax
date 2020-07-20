import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)

#def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
#    """
#    Computes and returns a (s|s) overlap integral
#    """
#    A = np.array([Ax, Ay, Az])
#    C = np.array([Cx, Cy, Cz])
#    #ss = ((np.pi / (alpha_bra + alpha_ket))**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(A-C, A-C)) / (alpha_bra + alpha_ket)))
#    #return ss * c1 * c2
#    alpha_sum = alpha_bra + alpha_ket
#    return c1 * c2 * (np.pi / alpha_sum)**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(A-C, A-C)) / alpha_sum)
#
##def contracted_overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
##    res = overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
##    return np.sum(res)
#
#vectorized_primitive_ss = jax.vmap(overlap_ss, (None,None,None,None,None,None,0,0,0,0))
#
#def contracted_ss(*args):
#    primitives = vectorized_primitive_ss(*args)
#    return np.sum(primitives)
#
##args = (0.5,0.5,0.5,1.5,1.5,1.5,0.5,0.5,0.5,0.5)
#vec_args = (0.5,0.5,0.5,1.5,1.5,1.5, np.array([0.5,0.5,0.5]), np.array([0.5,0.5,0.5]), np.array([0.5,0.5,0.5]), np.array([0.5,0.5,0.5]))
#res = contracted_ss(*vec_args)
#print(res)
#
#vectorized_contractions = jax.vmap(contracted_ss, (0,0,0,0,0,0,0,0,0,0))
#
#args = np.array([[0.5,0.5



def example_fun(length, val):
  return np.ones((length,)) * val
# un-jit'd works fine
print(example_fun(5, 4))

bad_example_jit = jax.jit(example_fun)
# this will fail:
try:
  print(bad_example_jit(10, 4))
except Exception as e:
  print("error!", e)
# static_argnums tells JAX to recompile on changes at these argument positions:
good_example_jit = jax.jit(example_fun, static_argnums=(0,))
# first compile
print(good_example_jit(10, 4))
# recompiles
print(good_example_jit(5, 4))
print(good_example_jit(5, 4))
print(good_example_jit(5, 4))
print(good_example_jit(5, 4))
print(good_example_jit(5, 4))
print(good_example_jit(5, 4))
