import jax
import jax.numpy as np
import numpy as onp
from jax.interpreters import batching
from jax import core


# Primitive is created
multiply_add_p = core.Primitive("multiply_add")

# Create function callable to call the primitive
def multiply_add_prim(x,y,z):
    return multiply_add_p.bind(x,y,z)


# Create primal evaluation rule
def multiply_add_impl(x,y,z):
    """ x * y + z """
    return onp.add(onp.multiply(x,y),z)

# Register the primal implemenation with JAX
multiply_add_p.def_impl(multiply_add_impl)

print(multiply_add_prim(1.0,2.0,3.0))

# Create JVP rule. You cannot reference e
def multiply_add_jvp(primals, tangents):
    x,y,z = primals
    x_dot, y_dot, z_dot = tangents
    primals_out = multiply_add_prim(x,y,z)
    def make_zero(tan):
        return jax.lax.zeros_like_array(x) if type(tan) is jax.ad.Zero else tan
    #tangents_out = mutiply_add_prim(make_zero(x_dot), y, mutiply_add_prim(x, make_zero(y_dot), make_zero(z_dot)))
    tangents_out = multiply_add_prim(make_zero(x_dot), y, multiply_add_prim(x, make_zero(y_dot), make_zero(z_dot)))

    return primals_out, tangents_out 
# Register the JVP rule
jax.ad.primitive_jvps[multiply_add_p] = multiply_add_jvp

jax.jvp(multiply_add_prim, (1.0,2.0,3.0), (np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.]),))

# Create batching rule, this is the weirdest one. Needed since vmap is used in jacfwd, not needed for using jvp directly.
# It is used in jacfwd to send through a batch of std basis vectors for the JVP's. 
def multiply_add_batch(vector_arg_values, batch_axes):
    #assert batch_axes[0] == batch_axes[1]
    #assert batch_axes[0] == batch_axes[2]
    res = multiply_add_prim(*vector_arg_values)
    return res, batch_axes[0]

# Register the batching rule
batching.primitive_batchers[multiply_add_p] = multiply_add_batch

what = jax.vmap(multiply_add_prim, in_axes=0, out_axes=0)(np.array([2., 3.]), np.array([10., 20.]), np.array([4.,5.]))
print(what)

what = jax.jacfwd(multiply_add_prim, 0)(1.0,2.0,3.0)
print(what)

# JACREV calls vjp, and the conversion of jvp to vjp requires abstract eval.
#what = jax.jacrev(multiply_add_prim, 0)(1.0,2.0,3.0)
#print(what)


# Uncertainties:
# Will my method of JVP even work for TEI's
