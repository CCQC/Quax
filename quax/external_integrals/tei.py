import jax 
import jax.numpy as jnp
import numpy as np
import h5py
import os
from . import libint_interface
from ..utils import get_deriv_vec_idx

jax.config.update("jax_enable_x64", True)
jax.config.enable_omnistaging()

# Create new JAX primitives for TEI evaluation and derivative evaluation
tei_p = jax.core.Primitive("tei")
tei_deriv_p = jax.core.Primitive("tei_deriv")

# Create functions to call primitives
def tei(geom):
    return tei_p.bind(geom)

def tei_deriv(geom, deriv_vec):
    return tei_deriv_p.bind(geom, deriv_vec) 

# Create primitive evaluation rules 
def tei_impl(geom):
    G = libint_interface.eri()
    d = int(np.sqrt(np.sqrt(G.shape[0])))
    G = G.reshape(d,d,d,d)
    return jnp.asarray(G)

def tei_deriv_impl(geom, deriv_vec):
    deriv_vec = np.asarray(deriv_vec, int)
    deriv_order = np.sum(deriv_vec)
    idx = get_deriv_vec_idx(deriv_vec)

    # By default, look for full derivative tensor file with datasets named (type)_deriv(order)
    # if not found, look for partial derivative tensor file with datasets named (type)_deriv(order)_(flattened_uppertri_idx)
    # if that also is not found, compute derivative on-the-fly and return
    if os.path.exists("eri_derivs.h5"):
        file_name = "eri_derivs.h5"
        dataset_name = "eri_deriv" + str(deriv_order)
    elif os.path.exists("eri_partials.h5"):
        file_name = "eri_partials.h5"
        dataset_name = "eri_deriv" + str(deriv_order) + "_" + str(idx)
    else:
        G = libint_interface.eri_deriv(np.asarray(deriv_vec, int))
        d = int(np.sqrt(np.sqrt(G.shape[0])))
        G = G.reshape(d,d,d,d)
        return jnp.asarray(G)

    with h5py.File(file_name, 'r') as f:
        data_set = f[dataset_name]
        if len(data_set.shape) == 5:
            G = data_set[:,:,:,:,idx]
        elif len(data_set.shape) == 4:
            G = data_set[:,:,:,:]
        else:
            raise Exception("Something went wrong reading integral derivative file")
    G = jnp.asarray(G)
    return G 
    
# Register primitive evaluation rules
tei_p.def_impl(tei_impl)
tei_deriv_p.def_impl(tei_deriv_impl)

# Next step: create Jacobian-vector product rules, which given some input args (primals)
# and a tangent std basis vector (tangent), returns the function evaluated at that point (primals_out)
# and the slice of the Jacobian (tangents_out)
def tei_jvp(primals, tangents):
    geom, = primals
    primals_out = tei(geom) 
    tangents_out = tei_deriv(geom, tangents[0])
    return primals_out, tangents_out

def tei_deriv_jvp(primals, tangents):
    geom, deriv_vec = primals
    primals_out = tei_deriv(geom, deriv_vec)
    # Here we add the current value of deriv_vec to the incoming tangent vector, 
    # so that nested higher order differentiation works
    tangents_out = tei_deriv(geom, deriv_vec + tangents[0])
    return primals_out, tangents_out

# Register the JVP rules with JAX
jax.ad.primitive_jvps[tei_p] = tei_jvp
jax.ad.primitive_jvps[tei_deriv_p] = tei_deriv_jvp

# Define Batching rules, this is only needed since jax.jacfwd will call vmap on the JVP of tei
def tei_deriv_batch(batched_args, batch_dims):
    # When the input argument of deriv_batch is batched along the 0'th axis
    # we want to evaluate every 4d slice, gather up a (ncart, n,n,n,n) array, 
    # (expand dims at 0 and concatenate at 0)
    # and then return the results, indicating the out batch axis 
    # is in the 0th position (return results, 0)
    geom_batch, deriv_batch = batched_args
    geom_dim, deriv_dim = batch_dims
    results = []
    for i in deriv_batch:
        tmp = tei_deriv(geom_batch, i)
        results.append(jnp.expand_dims(tmp, axis=0))
    results = jnp.concatenate(results, axis=0)
    return results, 0

# Register the batching rules with JAX
jax.interpreters.batching.primitive_batchers[tei_deriv_p] = tei_deriv_batch


