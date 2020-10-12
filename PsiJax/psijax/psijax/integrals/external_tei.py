import jax 
from jax import core
from jax.interpreters import batching
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import numpy as onp

# TEMP TODO: only needed to test psi_tei_deriv_impl
# On second derivatives
#def wrap(geomflat):
#    geom = geomflat.reshape(-1,3)
#    return tei_array(geom, basis_dict) 
#tmp_Hess = onp.asarray(jax.jacfwd(jax.jacfwd(wrap))(geom.reshape(-1)))

# Create new JAX primitives for TEI evaluation and derivative evaluation
psi_tei_p = core.Primitive("psi_tei")
psi_tei_deriv_p = core.Primitive("psi_tei_deriv")

# Create functions to call primitives
def psi_tei(geom, **params):
    return psi_tei_p.bind(geom, **params)

def psi_tei_deriv(geom, deriv_vec, **params):
    return psi_tei_deriv_p.bind(geom, deriv_vec, **params) 

# Create primitive evaluation rules 
def psi_tei_impl(geom, **params):
    mints = params['mints']
    psi_G = np.asarray(onp.asarray(mints.ao_eri()))
    return psi_G

def psi_tei_deriv_impl(geom, deriv_vec, **params):
    # TODO once Psi has TEI deriv:
    # mints = params['mints']
    # dG_di = mints.ao_tei_deriv(deriv_vec)
    # OR if tuple-like arguments:
    # indices = onp.nonzero(deriv_vec)[0]
    # args = onp.repeat(indices, deriv_vec[indices]))
    # dG_di = mints.ao_tei_deriv(tuple(args))
    
    # Quick, dirty, brainless TEI derivative code.
    # For first derivatives, use Psi, since its correct.
    # We will hardcode second derivatives as well, using PsiJax exact derivatives
    mints = params['mints']
    if onp.allclose(deriv_vec,onp.array([1.,0.,0.,0.,0.,0.])):
        dG_di = np.asarray(onp.asarray(mints.ao_tei_deriv1(0)[0]))
    if onp.allclose(deriv_vec,onp.array([0.,1.,0.,0.,0.,0.])):
        dG_di = np.asarray(onp.asarray(mints.ao_tei_deriv1(0)[1]))
    if onp.allclose(deriv_vec,onp.array([0.,0.,1.,0.,0.,0.])):
        dG_di = np.asarray(onp.asarray(mints.ao_tei_deriv1(0)[2]))
    if onp.allclose(deriv_vec,onp.array([0.,0.,0.,1.,0.,0.])):
        dG_di = np.asarray(onp.asarray(mints.ao_tei_deriv1(1)[0]))
    if onp.allclose(deriv_vec,onp.array([0.,0.,0.,0.,1.,0.])):
        dG_di = np.asarray(onp.asarray(mints.ao_tei_deriv1(1)[1]))
    if onp.allclose(deriv_vec,onp.array([0.,0.,0.,0.,0.,1.])):
        dG_di = np.asarray(onp.asarray(mints.ao_tei_deriv1(1)[2]))

    # For second derivs: use precompouted tmp_Hess from above 
    if onp.allclose(deriv_vec,onp.array([2.,0.,0.,0.,0.,0.])):
        dG_di = tmp_Hess[:,:,:,:,0,0]
    if onp.allclose(deriv_vec,onp.array([1.,1.,0.,0.,0.,0.])):
        dG_di = tmp_Hess[:,:,:,:,0,1]
    if onp.allclose(deriv_vec,onp.array([1.,0.,1.,0.,0.,0.])):
        dG_di = tmp_Hess[:,:,:,:,0,2]
    if onp.allclose(deriv_vec,onp.array([1.,0.,0.,1.,0.,0.])):
        dG_di = tmp_Hess[:,:,:,:,0,3]
    if onp.allclose(deriv_vec,onp.array([1.,0.,0.,0.,1.,0.])):
        dG_di = tmp_Hess[:,:,:,:,0,4]
    if onp.allclose(deriv_vec,onp.array([1.,0.,0.,0.,0.,1.])):
        dG_di = tmp_Hess[:,:,:,:,0,5]
    if onp.allclose(deriv_vec,onp.array([0.,2.,0.,0.,0.,0.])):
        dG_di = tmp_Hess[:,:,:,:,1,1]
    if onp.allclose(deriv_vec,onp.array([0.,1.,1.,0.,0.,0.])):
        dG_di = tmp_Hess[:,:,:,:,1,2]
    if onp.allclose(deriv_vec,onp.array([0.,1.,0.,1.,0.,0.])):
        dG_di = tmp_Hess[:,:,:,:,1,3]
    if onp.allclose(deriv_vec,onp.array([0.,1.,0.,0.,1.,0.])):
        dG_di = tmp_Hess[:,:,:,:,1,4]
    if onp.allclose(deriv_vec,onp.array([0.,1.,0.,0.,0.,1.])):
        dG_di = tmp_Hess[:,:,:,:,1,5]
    if onp.allclose(deriv_vec,onp.array([0.,0.,2.,0.,0.,0.])):
        dG_di = tmp_Hess[:,:,:,:,2,2]
    if onp.allclose(deriv_vec,onp.array([0.,0.,1.,1.,0.,0.])):
        dG_di = tmp_Hess[:,:,:,:,2,3]
    if onp.allclose(deriv_vec,onp.array([0.,0.,1.,0.,1.,0.])):
        dG_di = tmp_Hess[:,:,:,:,2,4]
    if onp.allclose(deriv_vec,onp.array([0.,0.,1.,0.,0.,1.])):
        dG_di = tmp_Hess[:,:,:,:,2,5]
    if onp.allclose(deriv_vec,onp.array([0.,0.,0.,2.,0.,0.])):
        dG_di = tmp_Hess[:,:,:,:,3,3]
    if onp.allclose(deriv_vec,onp.array([0.,0.,0.,1.,1.,0.])):
        dG_di = tmp_Hess[:,:,:,:,3,4]
    if onp.allclose(deriv_vec,onp.array([0.,0.,0.,1.,0.,1.])):
        dG_di = tmp_Hess[:,:,:,:,3,5]
    if onp.allclose(deriv_vec,onp.array([0.,0.,0.,0.,2.,0.])):
        dG_di = tmp_Hess[:,:,:,:,4,4]
    if onp.allclose(deriv_vec,onp.array([0.,0.,0.,0.,1.,1.])):
        dG_di = tmp_Hess[:,:,:,:,4,5]
    if onp.allclose(deriv_vec,onp.array([0.,0.,0.,0.,0.,2.])):
        dG_di = tmp_Hess[:,:,:,:,5,5]
    return np.asarray(dG_di)

# Register primitive evaluation rules
psi_tei_p.def_impl(psi_tei_impl)
psi_tei_deriv_p.def_impl(psi_tei_deriv_impl)

# Next step: create Jacobian-vector product rules, which given some input args (primals)
# and a tangent std basis vector (tangent), returns the function evaluated at that point (primals_out)
# and the slice of the Jacobian (tangents_out)
def psi_tei_jvp(primals, tangents, **params):
    mints = params['mints']
    geom, = primals
    primals_out = psi_tei(geom, **params) 
    tangents_out = psi_tei_deriv(geom, tangents[0], mints=mints)
    return primals_out, tangents_out

def psi_tei_deriv_jvp(primals, tangents, **params):
    geom, deriv_vec = primals
    primals_out = psi_tei_deriv(geom, deriv_vec, mints=mints)
    # Here we add the current value of deriv_vec to the incoming tangent vector, 
    # so that nested higher order differentiation works
    tangents_out = psi_tei_deriv(geom, deriv_vec + tangents[0], mints=mints)
    return primals_out, tangents_out

# Register the JVP rules with JAX
jax.ad.primitive_jvps[psi_tei_p] = psi_tei_jvp
jax.ad.primitive_jvps[psi_tei_deriv_p] = psi_tei_deriv_jvp

# Define Batching rules, this is only needed since jax.jacfwd will call vmap on the JVP of psi_tei
def psi_tei_deriv_batch(batched_args, batch_dims, **params):
    # When the input argument of deriv_batch is batched along the 0'th axis
    # we want to evaluate every 4d slice, gather up a (ncart, n,n,n,n) array, 
    # (expand dims at 0 and concatenate at 0)
    # and then return the results, indicating the out batch axis 
    # is in the 0th position (return results, 0)
    geom_batch, deriv_batch = batched_args
    geom_dim, deriv_dim = batch_dims
    results = []
    for i in deriv_batch:
        tmp = psi_tei_deriv(geom_batch, i, **params)
        results.append(np.expand_dims(tmp, axis=0))
    results = np.concatenate(results, axis=0)
    return results, 0

batching.primitive_batchers[psi_tei_deriv_p] = psi_tei_deriv_batch



