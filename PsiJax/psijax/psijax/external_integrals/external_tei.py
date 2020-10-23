import jax 
import jax.numpy as np
import numpy as onp
jax.config.update("jax_enable_x64", True)

# For testing higher order derivatives: load in analytic psijax derivatives for a given molecule and basis set
# Produce as needed for testing by evaluating jacfwd's on psijax.integrals.tei_array. 
# This effectively simulates referencing an external library for computation
#path ='/home/adabbott/Git/PsiTorch/PsiJax/psijax/psijax/external_integrals/saved_ints/'
#tmp_tei_hess = onp.asarray(onp.load(path + 'tei_hess_h2_dz_1p6.npy'))
#tmp_tei_cube = onp.asarray(onp.load(path + 'tei_cube_h2_dz_1p6.npy'))

# Create new JAX primitives for TEI evaluation and derivative evaluation
psi_tei_p = jax.core.Primitive("psi_tei")
psi_tei_deriv_p = jax.core.Primitive("psi_tei_deriv")

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
    
    # TODO Hard-coded for referencing Psi4 gradients, precomputed PsiJax hessians, cubics 
    mints = params['mints']
    if onp.allclose(onp.sum(deriv_vec), 1.):
        new_vec = deriv_vec.reshape(-1,3)
        indices = onp.nonzero(new_vec)
        atom_idx = indices[0][0]
        cart_idx = indices[1][0]
        dG_di = onp.asarray(mints.ao_tei_deriv1(atom_idx)[cart_idx])
    # For second derivs: use precomputed exact PsiJax TEI hessian, saved to disk, loaded in above
    elif onp.allclose(onp.sum(deriv_vec), 2.):
        deriv_vec = onp.asarray(deriv_vec, dtype=int)
        indices = onp.nonzero(deriv_vec)[0]
        args = onp.repeat(indices, deriv_vec[indices])
        i,j = args 
        dG_di = tmp_tei_hess[:,:,:,:,i,j]
    # Third derivs using precomputed exact PsiJax TEI third derivatives
    elif onp.allclose(onp.sum(deriv_vec), 3.):
        deriv_vec = onp.asarray(deriv_vec, dtype=int)
        indices = onp.nonzero(deriv_vec)[0]
        args = onp.repeat(indices, deriv_vec[indices])
        i,j,k = args 
        dG_di = tmp_tei_cube[:,:,:,:,i,j,k]
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
    mints = params['mints']
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

# Register the batching rules with JAX
jax.interpreters.batching.primitive_batchers[psi_tei_deriv_p] = psi_tei_deriv_batch


