import jax 
import jax.numpy as np
import numpy as onp
jax.config.update("jax_enable_x64", True)

# NOTE For testing higher order derivatives: load in analytic derivatives for given molecule and basis
# Produce as needed for testing by evaluating jacfwd's on psijax.integrals.oei_arrays
# This effectively simulates referencing an external library for computation
tmp_S_hess = onp.asarray(onp.load('overlap_hess_h2_dz_1p6.npy'))
tmp_T_hess = onp.asarray(onp.load('kinetic_hess_h2_dz_1p6.npy'))
tmp_V_hess = onp.asarray(onp.load('potential_hess_h2_dz_1p6.npy'))
tmp_S_cube = onp.asarray(onp.load('overlap_cube_h2_dz_1p6.npy'))
tmp_T_cube = onp.asarray(onp.load('kinetic_cube_h2_dz_1p6.npy'))
tmp_V_cube = onp.asarray(onp.load('potential_cube_h2_dz_1p6.npy'))

# Create new JAX primitives for overlap, kinetic, potential evaluation and their derivatives 
psi_overlap_p = jax.core.Primitive("psi_overlap")
psi_overlap_deriv_p = jax.core.Primitive("psi_overlap_deriv")

psi_kinetic_p = jax.core.Primitive("psi_kinetic")
psi_kinetic_deriv_p = jax.core.Primitive("psi_kinetic_deriv")

psi_potential_p = jax.core.Primitive("psi_potential")
psi_potential_deriv_p = jax.core.Primitive("psi_potential_deriv")

# Create functions to call primitives
def psi_overlap(geom, **params):
    return psi_overlap_p.bind(geom, **params)

def psi_overlap_deriv(geom, deriv_vec, **params):
    return psi_overlap_deriv_p.bind(geom, deriv_vec, **params) 

def psi_kinetic(geom, **params):
    return psi_kinetic_p.bind(geom, **params)

def psi_kinetic_deriv(geom, deriv_vec, **params):
    return psi_kinetic_deriv_p.bind(geom, deriv_vec, **params) 

def psi_potential(geom, **params):
    return psi_potential_p.bind(geom, **params)

def psi_potential_deriv(geom, deriv_vec, **params):
    return psi_potential_deriv_p.bind(geom, deriv_vec, **params) 

# Create primitive evaluation rules 
def psi_overlap_impl(geom, **params):
    mints = params['mints']
    psi_S = np.asarray(onp.asarray(mints.ao_overlap()))
    return psi_S

def psi_kinetic_impl(geom, **params):
    mints = params['mints']
    psi_T = np.asarray(onp.asarray(mints.ao_kinetic()))
    return psi_T

def psi_potential_impl(geom, **params):
    mints = params['mints']
    psi_V = np.asarray(onp.asarray(mints.ao_potential()))
    return psi_V

# TODO Following are hard-coded for gradients and precomputed hessians, precomputed third derivatives
# Change to general form once integral derivative API is complete
def psi_overlap_deriv_impl(geom, deriv_vec, **params):
    mints = params['mints']
    # TODO Hard-coded for referencing Psi4 gradients, precomputed PsiJax hessians, cubics 
    if onp.allclose(onp.sum(deriv_vec), 1.):
        # General way to convert one-hot flattened basis vector
        # into atom index, cart index as psi prefers
        deriv_vec = deriv_vec.reshape(-1,3)
        indices = onp.nonzero(deriv_vec)
        atom_idx = indices[0][0]
        cart_idx = indices[1][0]
        dG_di = mints.ao_oei_deriv1("OVERLAP",atom_idx)[cart_idx]
    # For second derivatives: use precomputed exact PsiJax overlap hessian, saved to disk, loaded in above
    if onp.allclose(onp.sum(deriv_vec), 2.):
        deriv_vec = onp.asarray(deriv_vec, dtype=int)
        indices = onp.nonzero(deriv_vec)[0]
        args = onp.repeat(indices, deriv_vec[indices])
        i,j = args
        dG_di = tmp_S_hess[:,:,i,j]
    # For third derivatives: use precomputed exact PsiJax overlap cubic, saved to disk, loaded in above
    if onp.allclose(onp.sum(deriv_vec), 3.):
        deriv_vec = onp.asarray(deriv_vec, dtype=int)
        indices = onp.nonzero(deriv_vec)[0]
        args = onp.repeat(indices, deriv_vec[indices])
        i,j,k = args
        dG_di = tmp_S_cube[:,:,i,j,k]
    return np.asarray(onp.asarray(dG_di))

def psi_kinetic_deriv_impl(geom, deriv_vec, **params):
    mints = params['mints']
    # First derivatives: Use Psi4
    if onp.allclose(onp.sum(deriv_vec), 1.):
        deriv_vec = deriv_vec.reshape(-1,3)
        indices = onp.nonzero(deriv_vec)
        atom_idx = indices[0][0]
        cart_idx = indices[1][0]
        dG_di = mints.ao_oei_deriv1("KINETIC",atom_idx)[cart_idx]
    # For second derivatives: use precomputed exact PsiJax kinetic hessian, saved to disk, loaded in above
    if onp.allclose(onp.sum(deriv_vec), 2.):
        deriv_vec = onp.asarray(deriv_vec, dtype=int)
        indices = onp.nonzero(deriv_vec)[0]
        args = onp.repeat(indices, deriv_vec[indices])
        i,j = args
        dG_di = tmp_T_hess[:,:,i,j]
    # For third derivatives: use precomputed exact PsiJax kinetic cubic, saved to disk, loaded in above
    if onp.allclose(onp.sum(deriv_vec), 3.):
        deriv_vec = onp.asarray(deriv_vec, dtype=int)
        indices = onp.nonzero(deriv_vec)[0]
        args = onp.repeat(indices, deriv_vec[indices])
        i,j,k = args
        dG_di = tmp_T_cube[:,:,i,j,k]
    return np.asarray(onp.asarray(dG_di))

def psi_potential_deriv_impl(geom, deriv_vec, **params):
    mints = params['mints']
    # First derivatives: Use Psi4
    if onp.allclose(onp.sum(deriv_vec), 1.):
        deriv_vec = deriv_vec.reshape(-1,3)
        indices = onp.nonzero(deriv_vec)
        atom_idx = indices[0][0]
        cart_idx = indices[1][0]
        dG_di = mints.ao_oei_deriv1("POTENTIAL",atom_idx)[cart_idx]
    # For second derivatives: use precomputed exact PsiJax potential hessian, saved to disk, loaded in above
    if onp.allclose(onp.sum(deriv_vec), 2.):
        deriv_vec = onp.asarray(deriv_vec, dtype=int)
        indices = onp.nonzero(deriv_vec)[0]
        args = onp.repeat(indices, deriv_vec[indices])
        i,j = args
        dG_di = tmp_V_hess[:,:,i,j]
    # For third derivatives: use precomputed exact PsiJax potential cubic, saved to disk, loaded in above
    if onp.allclose(onp.sum(deriv_vec), 3.):
        deriv_vec = onp.asarray(deriv_vec, dtype=int)
        indices = onp.nonzero(deriv_vec)[0]
        args = onp.repeat(indices, deriv_vec[indices])
        i,j,k = args
        dG_di = tmp_V_cube[:,:,i,j,k]
    return np.asarray(onp.asarray(dG_di))

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
def psi_overlap_jvp(primals, tangents, **params):
    mints = params['mints']
    geom, = primals
    primals_out = psi_overlap(geom, **params) 
    tangents_out = psi_overlap_deriv(geom, tangents[0], mints=mints)
    return primals_out, tangents_out

def psi_overlap_deriv_jvp(primals, tangents, **params):
    mints = params['mints']
    geom, deriv_vec = primals
    primals_out = psi_overlap_deriv(geom, deriv_vec, mints=mints)
    tangents_out = psi_overlap_deriv(geom, deriv_vec + tangents[0], mints=mints)
    return primals_out, tangents_out

def psi_kinetic_jvp(primals, tangents, **params):
    mints = params['mints']
    geom, = primals
    primals_out = psi_kinetic(geom, **params) 
    tangents_out = psi_kinetic_deriv(geom, tangents[0], mints=mints)
    return primals_out, tangents_out

def psi_kinetic_deriv_jvp(primals, tangents, **params):
    mints = params['mints']
    geom, deriv_vec = primals
    primals_out = psi_kinetic_deriv(geom, deriv_vec, mints=mints)
    tangents_out = psi_kinetic_deriv(geom, deriv_vec + tangents[0], mints=mints)
    return primals_out, tangents_out

def psi_potential_jvp(primals, tangents, **params):
    mints = params['mints']
    geom, = primals
    primals_out = psi_potential(geom, **params) 
    tangents_out = psi_potential_deriv(geom, tangents[0], mints=mints)
    return primals_out, tangents_out

def psi_potential_deriv_jvp(primals, tangents, **params):
    mints = params['mints']
    geom, deriv_vec = primals
    primals_out = psi_potential_deriv(geom, deriv_vec, mints=mints)
    tangents_out = psi_potential_deriv(geom, deriv_vec + tangents[0], mints=mints)
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
def psi_overlap_deriv_batch(batched_args, batch_dims, **params):
    # When the input argument of deriv_batch is batched along the 0'th axis
    # we want to evaluate every 2d slice, gather up a (ncart, n,n) array, 
    # (expand dims at 0 and concatenate at 0)
    # and then return the results, indicating the out batch axis 
    # is in the 0th position (return results, 0)
    geom_batch, deriv_batch = batched_args
    geom_dim, deriv_dim = batch_dims
    results = []
    for i in deriv_batch:
        tmp = psi_overlap_deriv(geom_batch, i, **params)
        results.append(np.expand_dims(tmp, axis=0))
    results = np.concatenate(results, axis=0)
    return results, 0

def psi_kinetic_deriv_batch(batched_args, batch_dims, **params):
    geom_batch, deriv_batch = batched_args
    geom_dim, deriv_dim = batch_dims
    results = []
    for i in deriv_batch:
        tmp = psi_kinetic_deriv(geom_batch, i, **params)
        results.append(np.expand_dims(tmp, axis=0))
    results = np.concatenate(results, axis=0)
    return results, 0

def psi_potential_deriv_batch(batched_args, batch_dims, **params):
    geom_batch, deriv_batch = batched_args
    geom_dim, deriv_dim = batch_dims
    results = []
    for i in deriv_batch:
        tmp = psi_potential_deriv(geom_batch, i, **params)
        results.append(np.expand_dims(tmp, axis=0))
    results = np.concatenate(results, axis=0)
    return results, 0

# Register the batching rules with JAX
jax.interpreters.batching.primitive_batchers[psi_overlap_deriv_p] = psi_overlap_deriv_batch
jax.interpreters.batching.primitive_batchers[psi_kinetic_deriv_p] = psi_kinetic_deriv_batch
jax.interpreters.batching.primitive_batchers[psi_potential_deriv_p] = psi_potential_deriv_batch
