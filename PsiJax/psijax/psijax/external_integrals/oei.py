import jax 
import jax.numpy as np
import numpy as onp
from . import libint_interface
jax.config.update("jax_enable_x64", True)

# Create new JAX primitives for overlap, kinetic, potential evaluation and their derivatives 
overlap_p = jax.core.Primitive("overlap")
overlap_deriv_p = jax.core.Primitive("overlap_deriv")

kinetic_p = jax.core.Primitive("kinetic")
kinetic_deriv_p = jax.core.Primitive("kinetic_deriv")

potential_p = jax.core.Primitive("potential")
potential_deriv_p = jax.core.Primitive("potential_deriv")

# Create functions to call primitives
def overlap(geom):
    return overlap_p.bind(geom)

def overlap_deriv(geom, deriv_vec):
    return overlap_deriv_p.bind(geom, deriv_vec) 

def kinetic(geom):
    return kinetic_p.bind(geom)

def kinetic_deriv(geom, deriv_vec):
    return kinetic_deriv_p.bind(geom, deriv_vec) 

def potential(geom):
    return potential_p.bind(geom)

def potential_deriv(geom, deriv_vec):
    return potential_deriv_p.bind(geom, deriv_vec) 

# Create primitive evaluation rules 
def overlap_impl(geom):
    S = libint_interface.overlap()
    d = int(onp.sqrt(S.shape[0]))
    S = S.reshape(d,d)
    return np.asarray(S)

def kinetic_impl(geom):
    T = libint_interface.kinetic() 
    d = int(onp.sqrt(T.shape[0]))
    T = T.reshape(d,d)
    return np.asarray(T) 

def potential_impl(geom):
    V = libint_interface.potential()
    d = int(onp.sqrt(V.shape[0]))
    V = V.reshape(d,d)
    return np.asarray(V)

def overlap_deriv_impl(geom, deriv_vec):
    dS = libint_interface.overlap_deriv(onp.asarray(deriv_vec, int))
    dim = int(onp.sqrt(dS.shape[0]))
    return np.asarray(dS).reshape(dim,dim)

def kinetic_deriv_impl(geom, deriv_vec):
    dT = libint_interface.kinetic_deriv(onp.asarray(deriv_vec, int))
    dim = int(onp.sqrt(dT.shape[0]))
    return np.asarray(dT).reshape(dim,dim)

def potential_deriv_impl(geom, deriv_vec):
    dV = libint_interface.potential_deriv(onp.asarray(deriv_vec,int))
    dim = int(onp.sqrt(dV.shape[0]))
    return np.asarray(dV).reshape(dim,dim)

# Register primitive evaluation rules
overlap_p.def_impl(overlap_impl)
overlap_deriv_p.def_impl(overlap_deriv_impl)
kinetic_p.def_impl(kinetic_impl)
kinetic_deriv_p.def_impl(kinetic_deriv_impl)
potential_p.def_impl(potential_impl)
potential_deriv_p.def_impl(potential_deriv_impl)

# Next step: create Jacobian-vector product rules, which given some input args (primals)
# and a tangent std basis vector (tangent), returns the function evaluated at that point (primals_out)
# and the slice of the Jacobian (tangents_out)
def overlap_jvp(primals, tangents):
    geom, = primals
    primals_out = overlap(geom) 
    tangents_out = overlap_deriv(geom, tangents[0])
    return primals_out, tangents_out

def overlap_deriv_jvp(primals, tangents):
    geom, deriv_vec = primals
    primals_out = overlap_deriv(geom, deriv_vec)
    tangents_out = overlap_deriv(geom, deriv_vec + tangents[0])
    return primals_out, tangents_out

def kinetic_jvp(primals, tangents):
    geom, = primals
    primals_out = kinetic(geom) 
    tangents_out = kinetic_deriv(geom, tangents[0])
    return primals_out, tangents_out

def kinetic_deriv_jvp(primals, tangents):
    geom, deriv_vec = primals
    primals_out = kinetic_deriv(geom, deriv_vec)
    tangents_out = kinetic_deriv(geom, deriv_vec + tangents[0])
    return primals_out, tangents_out

def potential_jvp(primals, tangents):
    geom, = primals
    primals_out = potential(geom) 
    tangents_out = potential_deriv(geom, tangents[0])
    return primals_out, tangents_out

def potential_deriv_jvp(primals, tangents):
    geom, deriv_vec = primals
    primals_out = potential_deriv(geom, deriv_vec)
    tangents_out = potential_deriv(geom, deriv_vec + tangents[0])
    return primals_out, tangents_out

# Register the JVP rules with JAX
jax.ad.primitive_jvps[overlap_p] = overlap_jvp
jax.ad.primitive_jvps[overlap_deriv_p] = overlap_deriv_jvp
jax.ad.primitive_jvps[kinetic_p] = kinetic_jvp
jax.ad.primitive_jvps[kinetic_deriv_p] = kinetic_deriv_jvp
jax.ad.primitive_jvps[potential_p] = potential_jvp
jax.ad.primitive_jvps[potential_deriv_p] = potential_deriv_jvp

# Define Batching rules, this is only needed since jax.jacfwd will call vmap on the JVP's
# of each oei function
def overlap_deriv_batch(batched_args, batch_dims):
    # When the input argument of deriv_batch is batched along the 0'th axis
    # we want to evaluate every 2d slice, gather up a (ncart, n,n) array, 
    # (expand dims at 0 and concatenate at 0)
    # and then return the results, indicating the out batch axis 
    # is in the 0th position (return results, 0)
    geom_batch, deriv_batch = batched_args
    geom_dim, deriv_dim = batch_dims
    results = []
    for i in deriv_batch:
        tmp = overlap_deriv(geom_batch, i)
        results.append(np.expand_dims(tmp, axis=0))
    results = np.concatenate(results, axis=0)
    return results, 0

def kinetic_deriv_batch(batched_args, batch_dims):
    geom_batch, deriv_batch = batched_args
    geom_dim, deriv_dim = batch_dims
    results = []
    for i in deriv_batch:
        tmp = kinetic_deriv(geom_batch, i)
        results.append(np.expand_dims(tmp, axis=0))
    results = np.concatenate(results, axis=0)
    return results, 0

def potential_deriv_batch(batched_args, batch_dims):
    geom_batch, deriv_batch = batched_args
    geom_dim, deriv_dim = batch_dims
    results = []
    for i in deriv_batch:
        tmp = potential_deriv(geom_batch, i)
        results.append(np.expand_dims(tmp, axis=0))
    results = np.concatenate(results, axis=0)
    return results, 0

# Register the batching rules with JAX
jax.interpreters.batching.primitive_batchers[overlap_deriv_p] = overlap_deriv_batch
jax.interpreters.batching.primitive_batchers[kinetic_deriv_p] = kinetic_deriv_batch
jax.interpreters.batching.primitive_batchers[potential_deriv_p] = potential_deriv_batch

