import jax 
import jax.numpy as np
import numpy as onp
import h5py
from . import libint_interface
from . import utils
jax.config.update("jax_enable_x64", True)

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
    d = int(onp.sqrt(onp.sqrt(G.shape[0])))
    G = G.reshape(d,d,d,d)
    return np.asarray(G)

def tei_deriv_impl(geom, deriv_vec):
#    deriv_vec = onp.asarray(deriv_vec, int)
#    #print("calling libint eri deriv with deriv vec ", deriv_vec)
#    G = libint_interface.eri_deriv(deriv_vec)
#    d = int(onp.sqrt(onp.sqrt(G.shape[0])))
#    G = G.reshape(d,d,d,d)

    # New disk-based implementation
    deriv_vec = onp.asarray(deriv_vec, int)
    deriv_order = onp.sum(deriv_vec)
    idx = utils.get_deriv_vec_idx(deriv_vec)
    dataset_name = "eri_deriv" + str(deriv_order)
    with h5py.File('eri_derivs.h5', 'r') as f:
        data_set = f[dataset_name]
        G = data_set[:,:,:,:,idx]
    return np.asarray(G)
    
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
        results.append(np.expand_dims(tmp, axis=0))
    results = np.concatenate(results, axis=0)
    return results, 0

# Register the batching rules with JAX
jax.interpreters.batching.primitive_batchers[tei_deriv_p] = tei_deriv_batch


