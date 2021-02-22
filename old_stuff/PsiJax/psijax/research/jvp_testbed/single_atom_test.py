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

# Here, we try to complete the implementation, adding a batching rule for the two primitives.
# We also simplify the implementation by using a mints object. Mostly want to see hte performance impact of this
# Okay, instantiating mints once is wayyy faster.

# Strategy: define a TEI primitive, and TEI derivative primitive.
# The JVP rule of TEI primitive will call TEI derivative primitive
# The TEI derivative JVP rule will call TEI derivative itself 

# Create primitives
psi_tei_p = core.Primitive("psi_tei")
psi_tei_deriv_p = core.Primitive("psi_tei_deriv")

# Create functions to call primitives
# here params holds a mints object 
def psi_tei(geom, **params):
    return psi_tei_p.bind(geom, **params)

# here params holds mints object 
def psi_tei_deriv(geom, deriv_vec, **params):
    return psi_tei_deriv_p.bind(geom, deriv_vec, **params) 

# Create primitive evaluation rules 
def psi_tei_impl(geom, **params):
    mints = params['mints']
    psi_G = np.asarray(onp.asarray(mints.ao_eri()))
    return psi_G

def psi_tei_deriv_impl(geom, deriv_vec, **params):
    # Quick, dirty, brainless TEI derivative code.
    # Only supports up to second order due to Psi4 API limitations.
    mints = params['mints']
    if onp.allclose(deriv_vec,onp.array([1.,0.,0.])):
        dG_di = np.asarray(onp.asarray(mints.ao_tei_deriv1(0)[0]))
    if onp.allclose(deriv_vec,onp.array([0.,1.,0.])):
        dG_di = np.asarray(onp.asarray(mints.ao_tei_deriv1(0)[1]))
    if onp.allclose(deriv_vec,onp.array([0.,0.,1.])):
        dG_di = np.asarray(onp.asarray(mints.ao_tei_deriv1(0)[2]))

    if onp.allclose(onp.sum(deriv_vec), 2.):
        x1x1, x1y1, x1z1, y1x1, y1y1, y1z1, z1x1, z1y1, z1z1  = mints.ao_tei_deriv2(0,0)
    if onp.allclose(deriv_vec,onp.array([2.,0.,0.])):
        dG_di = np.asarray(onp.asarray(x1x1))
    if onp.allclose(deriv_vec,onp.array([1.,1.,0.])):
        dG_di = np.asarray(onp.asarray(x1y1))
    if onp.allclose(deriv_vec,onp.array([1.,0.,1.])):
        dG_di = np.asarray(onp.asarray(x1z1))
    if onp.allclose(deriv_vec,onp.array([0.,2.,0.])):
        dG_di = np.asarray(onp.asarray(y1y1))
    if onp.allclose(deriv_vec,onp.array([0.,1.,1.])):
        dG_di = np.asarray(onp.asarray(y1z1))
    if onp.allclose(deriv_vec,onp.array([0.,0.,2.])):
        dG_di = np.asarray(onp.asarray(z1z1))
    return dG_di

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
    # I really thought it would be more complicated than this... hmmm.
    #tei_derivatives = []
    #for g_dot in tangents:
    #    new_deriv = g_dot + deriv_vec
    #    tei_deriv = psi_tei_deriv(geom, new_deriv, mol=mol, basis_name=basis_name)
    #    tei_derivatives.append(tei_deriv)
    # need tangents for each input, but molecule and basis should not be modified
    #tangents_out = np.concatenate(tei_derivatives, axis=-1)
    # Hmm.. some random Zero(6) object is in tangents, messing everything up... Can I just ignore it?
    tangents_out = psi_tei_deriv(geom, deriv_vec + tangents[0], mints=mints)
    return primals_out, tangents_out

# Register the JVP rules with JAX
jax.ad.primitive_jvps[psi_tei_p] = psi_tei_jvp
jax.ad.primitive_jvps[psi_tei_deriv_p] = psi_tei_deriv_jvp

# Define Batching rules:
# FIRST ATTEMPT: just... evaluate it normally
def psi_tei_batch(batched_args, batch_dims, **params):
    operand, = batched_args
    bdim, = batch_dims
    # We will never batch this such that geom will be changed, so just take one of them
    result = psi_tei(operand[0], **params)
    return result, batch_dims[0]

def psi_tei_deriv_batch(batched_args, batch_dims, **params):
    # This will actually get batched, several deriv_dim values will be sent through
    # I have no clue man
    geom_batch, deriv_batch = batched_args
    geom_dim, deriv_dim = batch_dims
    results = []
    for i in deriv_batch:
        tmp = psi_tei_deriv(geom_batch, i, **params)
        #results.append(np.expand_dims(tmp, axis=-1))
        results.append(np.expand_dims(tmp, axis=-1))
    results = np.concatenate(results, axis=-1)
    #return results, batch_dims[1]
    #return results, 0  
    return results, -1

# Okay this is not even involved at all
#batching.primitive_batchers[psi_tei_p] = psi_tei_batch
batching.primitive_batchers[psi_tei_deriv_p] = psi_tei_deriv_batch

# Now we can evaluate, let's try it
molecule = psi4.geometry("""
                         0 1
                         He  1.0  2.0 3.0
                         no_reorient
                         no_com
                         units bohr
                         """)
geom = np.asarray(onp.asarray(molecule.geometry())).reshape(-1)
basis_name = 'sto-3g'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
mints = psi4.core.MintsHelper(basis_set)
params = {'mints': mints}

# Compute gradient
G = psi_tei(geom, **params)
# Compute gradient
gradient = jax.jacfwd(psi_tei)(geom, **params)
# Compute hessian 
hessian = jax.jacfwd(jax.jacfwd(psi_tei))(geom, **params)

print(G)
#print(gradient)
#print(hessian)

psi_G = onp.asarray(mints.ao_eri())
print("psijax G matches psi4 G", onp.allclose(G, psi_G))

dG_d0 = np.asarray(onp.asarray(mints.ao_tei_deriv1(0)[0]))
print(dG_d0)
dG_d1 = np.asarray(onp.asarray(mints.ao_tei_deriv1(0)[1]))
print(dG_d1)
dG_d2 = np.asarray(onp.asarray(mints.ao_tei_deriv1(0)[2]))
print(dG_d2)

