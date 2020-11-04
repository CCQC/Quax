import jax 
import jax.numpy as np
import numpy as onp
from . import libint_interface
jax.config.update("jax_enable_x64", True)

def libint_init(xyz_path, basis_name):
    libint_interface.initialize(xyz_path, basis_name) 
    return 0

def libint_finalize():
    libint_interface.finalize()
    return 0

# Create new JAX primitives for overlap, kinetic, potential evaluation and their derivatives 
psi_overlap_p = jax.core.Primitive("psi_overlap")
psi_overlap_deriv_p = jax.core.Primitive("psi_overlap_deriv")

psi_kinetic_p = jax.core.Primitive("psi_kinetic")
psi_kinetic_deriv_p = jax.core.Primitive("psi_kinetic_deriv")

psi_potential_p = jax.core.Primitive("psi_potential")
psi_potential_deriv_p = jax.core.Primitive("psi_potential_deriv")

# Create functions to call primitives
def psi_overlap(geom):
    return psi_overlap_p.bind(geom)

def psi_overlap_deriv(geom, deriv_vec):
    return psi_overlap_deriv_p.bind(geom, deriv_vec) 

def psi_kinetic(geom):
    return psi_kinetic_p.bind(geom)

def psi_kinetic_deriv(geom, deriv_vec):
    return psi_kinetic_deriv_p.bind(geom, deriv_vec) 

def psi_potential(geom):
    return psi_potential_p.bind(geom)

def psi_potential_deriv(geom, deriv_vec):
    return psi_potential_deriv_p.bind(geom, deriv_vec) 

# Create primitive evaluation rules 
def psi_overlap_impl(geom):
    S = libint_interface.overlap()
    d = int(onp.sqrt(S.shape[0]))
    S = S.reshape(d,d)
    return np.asarray(S)

def psi_kinetic_impl(geom):
    T = libint_interface.kinetic() 
    d = int(onp.sqrt(T.shape[0]))
    T = T.reshape(d,d)
    return np.asarray(T) 

def psi_potential_impl(geom):
    V = libint_interface.potential()
    d = int(onp.sqrt(V.shape[0]))
    V = V.reshape(d,d)
    return np.asarray(V)

def psi_overlap_deriv_impl(geom, deriv_vec):
    dS = libint_interface.overlap_deriv(onp.asarray(deriv_vec, int))
    dim = int(onp.sqrt(dS.shape[0]))
    return np.asarray(dS).reshape(dim,dim)

def psi_kinetic_deriv_impl(geom, deriv_vec):
    dT = libint_interface.kinetic_deriv(onp.asarray(deriv_vec, int))
    dim = int(onp.sqrt(dT.shape[0]))
    return np.asarray(dT).reshape(dim,dim)

def psi_potential_deriv_impl(geom, deriv_vec):
    dV = libint_interface.potential_deriv(onp.asarray(deriv_vec,int))
    dim = int(onp.sqrt(dV.shape[0]))
    return np.asarray(dV).reshape(dim,dim)

# Register primitive evaluation rules
psi_overlap_p.def_impl(psi_overlap_impl)
psi_overlap_deriv_p.def_impl(psi_overlap_deriv_impl)
psi_kinetic_p.def_impl(psi_kinetic_impl)
psi_kinetic_deriv_p.def_impl(psi_kinetic_deriv_impl)
psi_potential_p.def_impl(psi_potential_impl)
psi_potential_deriv_p.def_impl(psi_potential_deriv_impl)

# Next step: create Jacobian-vector product rules, which given some input args (primals)
# and a tangent std basis vector (tangent), returns the function evaluated at that point (primals_out)
# and the slice of the Jacobian (tangents_out)
def psi_overlap_jvp(primals, tangents):
    geom, = primals
    primals_out = psi_overlap(geom) 
    tangents_out = psi_overlap_deriv(geom, tangents[0])
    return primals_out, tangents_out

def psi_overlap_deriv_jvp(primals, tangents):
    geom, deriv_vec = primals
    primals_out = psi_overlap_deriv(geom, deriv_vec)
    tangents_out = psi_overlap_deriv(geom, deriv_vec + tangents[0])
    return primals_out, tangents_out

def psi_kinetic_jvp(primals, tangents):
    geom, = primals
    primals_out = psi_kinetic(geom) 
    tangents_out = psi_kinetic_deriv(geom, tangents[0])
    return primals_out, tangents_out

def psi_kinetic_deriv_jvp(primals, tangents):
    geom, deriv_vec = primals
    primals_out = psi_kinetic_deriv(geom, deriv_vec)
    tangents_out = psi_kinetic_deriv(geom, deriv_vec + tangents[0])
    return primals_out, tangents_out

def psi_potential_jvp(primals, tangents):
    geom, = primals
    primals_out = psi_potential(geom) 
    tangents_out = psi_potential_deriv(geom, tangents[0])
    return primals_out, tangents_out

def psi_potential_deriv_jvp(primals, tangents):
    geom, deriv_vec = primals
    primals_out = psi_potential_deriv(geom, deriv_vec)
    tangents_out = psi_potential_deriv(geom, deriv_vec + tangents[0])
    return primals_out, tangents_out

# Register the JVP rules with JAX
jax.ad.primitive_jvps[psi_overlap_p] = psi_overlap_jvp
jax.ad.primitive_jvps[psi_overlap_deriv_p] = psi_overlap_deriv_jvp
jax.ad.primitive_jvps[psi_kinetic_p] = psi_kinetic_jvp
jax.ad.primitive_jvps[psi_kinetic_deriv_p] = psi_kinetic_deriv_jvp
jax.ad.primitive_jvps[psi_potential_p] = psi_potential_jvp
jax.ad.primitive_jvps[psi_potential_deriv_p] = psi_potential_deriv_jvp

# Define Batching rules, this is only needed since jax.jacfwd will call vmap on the JVP's
# of each oei function
def psi_overlap_deriv_batch(batched_args, batch_dims):
    # When the input argument of deriv_batch is batched along the 0'th axis
    # we want to evaluate every 2d slice, gather up a (ncart, n,n) array, 
    # (expand dims at 0 and concatenate at 0)
    # and then return the results, indicating the out batch axis 
    # is in the 0th position (return results, 0)
    geom_batch, deriv_batch = batched_args
    geom_dim, deriv_dim = batch_dims
    results = []
    for i in deriv_batch:
        tmp = psi_overlap_deriv(geom_batch, i)
        results.append(np.expand_dims(tmp, axis=0))
    results = np.concatenate(results, axis=0)
    return results, 0

def psi_kinetic_deriv_batch(batched_args, batch_dims):
    geom_batch, deriv_batch = batched_args
    geom_dim, deriv_dim = batch_dims
    results = []
    for i in deriv_batch:
        tmp = psi_kinetic_deriv(geom_batch, i)
        results.append(np.expand_dims(tmp, axis=0))
    results = np.concatenate(results, axis=0)
    return results, 0

def psi_potential_deriv_batch(batched_args, batch_dims):
    geom_batch, deriv_batch = batched_args
    geom_dim, deriv_dim = batch_dims
    results = []
    for i in deriv_batch:
        tmp = psi_potential_deriv(geom_batch, i)
        results.append(np.expand_dims(tmp, axis=0))
    results = np.concatenate(results, axis=0)
    return results, 0

# Register the batching rules with JAX
jax.interpreters.batching.primitive_batchers[psi_overlap_deriv_p] = psi_overlap_deriv_batch
jax.interpreters.batching.primitive_batchers[psi_kinetic_deriv_p] = psi_kinetic_deriv_batch
jax.interpreters.batching.primitive_batchers[psi_potential_deriv_p] = psi_potential_deriv_batch

