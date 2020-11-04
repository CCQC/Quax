import jax 
import jax.numpy as np
import numpy as onp
from . import libint_interface
jax.config.update("jax_enable_x64", True)

# Create new JAX primitives for TEI evaluation and derivative evaluation
psi_tei_p = jax.core.Primitive("psi_tei")
psi_tei_deriv_p = jax.core.Primitive("psi_tei_deriv")

# Create functions to call primitives
def psi_tei(geom):
    return psi_tei_p.bind(geom)

def psi_tei_deriv(geom, deriv_vec):
    return psi_tei_deriv_p.bind(geom, deriv_vec) 

# Create primitive evaluation rules 
def psi_tei_impl(geom):
    G = libint_interface.eri()
    d = int(onp.sqrt(onp.sqrt(G.shape[0])))
    G = G.reshape(d,d,d,d)
    return np.asarray(G)

def psi_tei_deriv_impl(geom, deriv_vec):
    deriv_vec = onp.asarray(deriv_vec, int)
    G = libint_interface.eri_deriv(deriv_vec)
    d = int(onp.sqrt(onp.sqrt(G.shape[0])))
    G = G.reshape(d,d,d,d)
    return np.asarray(G)
    
# Register primitive evaluation rules
psi_tei_p.def_impl(psi_tei_impl)
psi_tei_deriv_p.def_impl(psi_tei_deriv_impl)

# Next step: create Jacobian-vector product rules, which given some input args (primals)
# and a tangent std basis vector (tangent), returns the function evaluated at that point (primals_out)
# and the slice of the Jacobian (tangents_out)
def psi_tei_jvp(primals, tangents):
    geom, = primals
    primals_out = psi_tei(geom) 
    tangents_out = psi_tei_deriv(geom, tangents[0])
    return primals_out, tangents_out

def psi_tei_deriv_jvp(primals, tangents):
    geom, deriv_vec = primals
    # Here we add the current value of deriv_vec to the incoming tangent vector, 
    # so that nested higher order differentiation works
    #tangents_out = psi_tei_deriv(geom, deriv_vec + tangents[0], mints=mints)

    primals_out = psi_tei_deriv(geom, deriv_vec)
    # Here we add the current value of deriv_vec to the incoming tangent vector, 
    # so that nested higher order differentiation works
    tangents_out = psi_tei_deriv(geom, deriv_vec + tangents[0])
    return primals_out, tangents_out

# Register the JVP rules with JAX
jax.ad.primitive_jvps[psi_tei_p] = psi_tei_jvp
jax.ad.primitive_jvps[psi_tei_deriv_p] = psi_tei_deriv_jvp

# Define Batching rules, this is only needed since jax.jacfwd will call vmap on the JVP of psi_tei
def psi_tei_deriv_batch(batched_args, batch_dims):
    # When the input argument of deriv_batch is batched along the 0'th axis
    # we want to evaluate every 4d slice, gather up a (ncart, n,n,n,n) array, 
    # (expand dims at 0 and concatenate at 0)
    # and then return the results, indicating the out batch axis 
    # is in the 0th position (return results, 0)
    geom_batch, deriv_batch = batched_args
    geom_dim, deriv_dim = batch_dims
    results = []
    for i in deriv_batch:
        tmp = psi_tei_deriv(geom_batch, i)
        results.append(np.expand_dims(tmp, axis=0))
    results = np.concatenate(results, axis=0)
    return results, 0

# Register the batching rules with JAX
jax.interpreters.batching.primitive_batchers[psi_tei_deriv_p] = psi_tei_deriv_batch


