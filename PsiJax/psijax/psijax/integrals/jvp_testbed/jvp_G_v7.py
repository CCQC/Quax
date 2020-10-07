import psi4
psi4.core.be_quiet()
import jax 
from jax.interpreters import batching
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp
from jax import core
np.set_printoptions(linewidth=500)

from psijax.integrals.basis_utils import build_basis_set
from psijax.integrals.tei import tei_array

# Now we can evaluate, let's try it
molecule = psi4.geometry("""
                         0 1
                         H  1.0  2.0 3.0
                         H -1.0 -2.0 -3.0
                         no_reorient
                         no_com
                         units bohr
                         """)
geom = np.asarray(onp.asarray(molecule.geometry())).reshape(-1)
basis_name = 'sto-3g'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
mints = psi4.core.MintsHelper(basis_set)
basis_dict = build_basis_set(molecule, basis_name)
params = {'mints': mints}

def wrap(geomflat):
    geom = geomflat.reshape(-1,3)
    return tei_array(geom, basis_dict) 

#TODO uncomment if running second order derivatives
#tmp_Hess = onp.asarray(jax.jacfwd(jax.jacfwd(wrap))(geom.reshape(-1)))

# Create primitives
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

# Register primitive evaluation rule
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
    # Here we add the current value of deriv_vec to the incoming tangent vector, so that nested higher order differentiation works
    tangents_out = psi_tei_deriv(geom, deriv_vec + tangents[0], mints=mints)
    return primals_out, tangents_out

# Register the JVP rules with JAX
jax.ad.primitive_jvps[psi_tei_p] = psi_tei_jvp
jax.ad.primitive_jvps[psi_tei_deriv_p] = psi_tei_deriv_jvp

#def psi_tei_batch(batched_args, batch_dims, **params):
#    print(batched_args)
#    print(batch_dims)
#    operand, = batched_args
#    bdim, = batch_dims
#    # We will never batch this such that geom will be changed, so just take one of them
#    result = psi_tei(operand[0], **params)
#    return result, -1

# Define Batching rules, this is only needed since jax.jacfwd will call vmap on the JVP of psi_tei
def psi_tei_deriv_batch(batched_args, batch_dims, **params):
    # When the input argument of deriv_batch is batched along the 0'th axis
    # we want to evaluate every 4d slice, gather up a (ncart, n,n,n,n) array, 
    # (expand dims at 0 and concatenate at 0)
    # and then return the results, indicating the out batch axis is in the 0th position (return results, 0)
    geom_batch, deriv_batch = batched_args
    geom_dim, deriv_dim = batch_dims
    results = []
    for i in deriv_batch:
        tmp = psi_tei_deriv(geom_batch, i, **params)
        #print(tmp.shape)
        results.append(np.expand_dims(tmp, axis=0))
    results = np.concatenate(results, axis=0)
    #print(results.shape)
    return results, 0

batching.primitive_batchers[psi_tei_deriv_p] = psi_tei_deriv_batch

# Compute gradients with above psi_tei function, compare to Psi4's gradients.
# We are, of course, using ao_tei_deriv1 in our psi_tei_deriv function, and therefore in the JVP of psi_tei.
# This is just making sure all the shaping and batching and tangent vector handling is correct.
#G_grad = jax.jacfwd(psi_tei)(geom, **params)
#x1, y1, z1 = mints.ao_tei_deriv1(0)
#x2, y2, z2 = mints.ao_tei_deriv1(1)
#print(onp.allclose(G_grad[:,:,:,:,0], x1))
#print(onp.allclose(G_grad[:,:,:,:,1], y1))
#print(onp.allclose(G_grad[:,:,:,:,2], z1))
#print(onp.allclose(G_grad[:,:,:,:,3], x2))
#print(onp.allclose(G_grad[:,:,:,:,4], y2))
#print(onp.allclose(G_grad[:,:,:,:,5], z2))

# Now the real test:
# Does nested differentiation work? psi_tei is using exact gradient and hessian slices to build
# full TEI hessian. Is it working? Compare to real psijax hessian.
#G_hess = jax.jacfwd(jax.jacfwd(psi_tei))(geom, **params)
#real_hess = jax.jacfwd(jax.jacfwd(wrap))(geom.reshape(-1))
#print(onp.allclose(G_hess, real_hess))
# The above returns true. You got it! Now we just need the new psi api to support fast arbitrary order differentiation.

# One last thing: simulate an energy computation. This revealed a batching bug, but it works now. 
def dummy_energy(geom):
    G = psi_tei(geom, **params)
    energy = np.sum(G) 
    return energy

test = jax.jacfwd(dummy_energy)(geom)
print(test)



