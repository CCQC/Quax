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

    if onp.allclose(onp.sum(deriv_vec), 2.):
        #NOTE the first call is correctly ordered. Check for typos
        x1x1, x1y1, x1z1, y1x1, y1y1, y1z1, z1x1, z1y1, z1z1  = mints.ao_tei_deriv2(0,0)
        x1x2, x1y2, x1z2, y1x2, y1y2, y1z2, z1x2, z1y2, z1z2  = mints.ao_tei_deriv2(0,1)
        x2x1, x2y1, x2z1, y2x1, y2y1, y2z1, z2x1, z2y1, z2z1  = mints.ao_tei_deriv2(1,0)
        x2x2, x2y2, x2z2, y2x2, y2y2, y2z2, z2x2, z2y2, z2z2  = mints.ao_tei_deriv2(1,1)
    if onp.allclose(deriv_vec,onp.array([2.,0.,0.,0.,0.,0.])):
        dG_di = np.asarray(onp.asarray(x1x1))
    if onp.allclose(deriv_vec,onp.array([1.,1.,0.,0.,0.,0.])):
        dG_di = np.asarray(onp.asarray(x1y1))
    if onp.allclose(deriv_vec,onp.array([1.,0.,1.,0.,0.,0.])):
        dG_di = np.asarray(onp.asarray(x1z1))
    if onp.allclose(deriv_vec,onp.array([1.,0.,0.,1.,0.,0.])):
        dG_di = np.asarray(onp.asarray(x1x2))
    if onp.allclose(deriv_vec,onp.array([1.,0.,0.,0.,1.,0.])):
        dG_di = np.asarray(onp.asarray(x1y2))
    if onp.allclose(deriv_vec,onp.array([1.,0.,0.,0.,0.,1.])):
        dG_di = np.asarray(onp.asarray(x1z2))
    if onp.allclose(deriv_vec,onp.array([0.,2.,0.,0.,0.,0.])):
        dG_di = np.asarray(onp.asarray(y1y1))
    if onp.allclose(deriv_vec,onp.array([0.,1.,1.,0.,0.,0.])):
        dG_di = np.asarray(onp.asarray(y1z1))
    if onp.allclose(deriv_vec,onp.array([0.,1.,0.,1.,0.,0.])):
        dG_di = np.asarray(onp.asarray(y1x2))
    if onp.allclose(deriv_vec,onp.array([0.,1.,0.,0.,1.,0.])):
        dG_di = np.asarray(onp.asarray(y1y2))
    if onp.allclose(deriv_vec,onp.array([0.,1.,0.,0.,0.,1.])):
        dG_di = np.asarray(onp.asarray(y1z2))
    if onp.allclose(deriv_vec,onp.array([0.,0.,2.,0.,0.,0.])):
        dG_di = np.asarray(onp.asarray(z1z1))
    if onp.allclose(deriv_vec,onp.array([0.,0.,1.,1.,0.,0.])):
        dG_di = np.asarray(onp.asarray(z1x2))
    if onp.allclose(deriv_vec,onp.array([0.,0.,1.,0.,1.,0.])):
        dG_di = np.asarray(onp.asarray(z1y2))
    if onp.allclose(deriv_vec,onp.array([0.,0.,1.,0.,0.,1.])):
        dG_di = np.asarray(onp.asarray(z1z2))
    if onp.allclose(deriv_vec,onp.array([0.,0.,0.,2.,0.,0.])):
        dG_di = np.asarray(onp.asarray(x2x2))
    if onp.allclose(deriv_vec,onp.array([0.,0.,0.,1.,1.,0.])):
        dG_di = np.asarray(onp.asarray(x2y2))
    if onp.allclose(deriv_vec,onp.array([0.,0.,0.,1.,0.,1.])):
        dG_di = np.asarray(onp.asarray(x2z2))
    if onp.allclose(deriv_vec,onp.array([0.,0.,0.,0.,2.,0.])):
        dG_di = np.asarray(onp.asarray(y2y2))
    if onp.allclose(deriv_vec,onp.array([0.,0.,0.,0.,1.,1.])):
        dG_di = np.asarray(onp.asarray(y2z2))
    if onp.allclose(deriv_vec,onp.array([0.,0.,0.,0.,0.,2.])):
        dG_di = np.asarray(onp.asarray(z2z2))
    return dG_di

# Register primitive evaluation rule
psi_tei_p.def_impl(psi_tei_impl)
psi_tei_deriv_p.def_impl(psi_tei_deriv_impl)

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
#basis_name = 'sto-3g'
#basis_name = '3-21g'
basis_name = 'cc-pvdz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
mints = psi4.core.MintsHelper(basis_set)
params = {'mints': mints}

deriv_test = np.array([1.,0.,0.,0.,0.,0.])
test = psi_tei(geom, **params)
test2 = psi_tei_deriv(geom, deriv_test, **params) 

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

# We can only call JVP on the partial. I think jacfwd handles this automatically...
#partial_psi_tei = partial(psi_tei, mol=molecule,basis_name=basis_name)
partial_psi_tei = partial(psi_tei, mints=mints)
#base_vec = np.array([1.,0.,0.,0.,0.,0.])
#p, t = jax.jvp(partial_psi_tei, (geom,), (base_vec,))

#partial_psi_tei_deriv = partial(psi_tei_deriv, mol=molecule,basis_name=basis_name)

# Let's skip to the real test.
def my_jacfwd_novmap(f):
    """A basic version of jax.jacfwd, with no vmap. assumes only one argument, no static args, etc"""
    def jacfun(x, **params):
        # create little function that grabs tangents (second arg returned, hence [1])
        _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
        # evaluate tangents on standard basis. Note we are only mapping over tangents arg of jvp
        #Jt = jax.vmap(_jvp, in_axes=1)(np.eye(len(x)))
        Jt = np.asarray([_jvp(i) for i in np.eye(len(x))])
        #print(Jt.shape)
        #return np.transpose(Jt)
        return np.moveaxis(Jt, 0, -1)
    return jacfun

# Currently I'm using partial here, but the real jax.jacfwd should work with explicit params for molecule and basis name, since you can pick which arg to differentiate
#gradient = my_jacfwd_novmap(partial_psi_tei)(geom)
#print(gradient.shape)
#hessian = my_jacfwd_novmap(my_jacfwd_novmap(partial_psi_tei))(geom)
#cubic = my_jacfwd_novmap(my_jacfwd_novmap(my_jacfwd_novmap(partial_psi_tei)))(geom)

# Define Batching rules:
# FIRST ATTEMPT: just... evaluate it normally
def psi_tei_batch(batched_args, batch_dims, **params):
    operand, = batched_args
    bdim, = batch_dims
    print(operand)
    print(bdim)
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

#NOPE!
#batching.defvectorized(psi_tei_p) 
#batching.defvectorized(psi_tei_deriv_p)
#NOPE!
#batching.defbroadcasting(psi_tei_p) 
#batching.defbroadcasting(psi_tei_deriv_p) 

# Okay, theres an error here with dimensions...
#g = jax.jacfwd(partial_psi_tei)(geom)
#print(g)

#hess = jax.jacfwd(jax.jacfwd(partial_psi_tei))(geom)
#print(hess)
#print(hess)
#hessian = my_jacfwd_novmap(my_jacfwd_novmap(partial_psi_tei))(geom)
#print(onp.allclose(hess,hessian))


# Does not partial work?
#hess2 = jax.jacfwd(jax.jacfwd(psi_tei))(geom, **params)
#print(np.allclose(hess,hess2))

G_grad = jax.jacfwd(psi_tei)(geom, **params)
x1, y1, z1 = mints.ao_tei_deriv1(0)
x2, y2, z2 = mints.ao_tei_deriv1(1)
print(onp.allclose(G_grad[:,:,:,:,0], x1))
print(onp.allclose(G_grad[:,:,:,:,1], y1))
print(onp.allclose(G_grad[:,:,:,:,2], z1))
print(onp.allclose(G_grad[:,:,:,:,3], x2))
print(onp.allclose(G_grad[:,:,:,:,4], y2))
print(onp.allclose(G_grad[:,:,:,:,5], z2))


# Well, we can check the same atom derivs
G_hess = jax.jacfwd(jax.jacfwd(psi_tei))(geom, **params)
#x1x1, x1y1, x1z1, y1x1, y1y1, y1z1, z1x1, z1y1, z1z1  = mints.ao_tei_deriv2(0,0)
#x2x2, x2y2, x2z2, y2x2, y2y2, y2z2, z2x2, z2y2, z2z2  = mints.ao_tei_deriv2(1,1)

x1x1, x1y1, x1z1, y1x1, y1y1, y1z1, z1x1, z1y1, z1z1  = mints.ao_tei_deriv2(0,0)
#x1x2, x1y2, x1z2, y1x2, y1y2, y1z2, z1x2, z1y2, z1z2  = mints.ao_tei_deriv2(0,1)
#x2x1, x2y1, x2z1, y2x1, y2y1, y2z1, z2x1, z2y1, z2z1  = mints.ao_tei_deriv2(1,0)
x2x2, x2y2, x2z2, y2x2, y2y2, y2z2, z2x2, z2y2, z2z2  = mints.ao_tei_deriv2(1,1)

print('x1x1', onp.allclose(x1x1, G_hess[:,:,:,:,0,0]))
print('x1y1', onp.allclose(x1y1, G_hess[:,:,:,:,0,1]))
print('x1z1', onp.allclose(x1z1, G_hess[:,:,:,:,0,2]))
print('y1x1', onp.allclose(y1x1, G_hess[:,:,:,:,1,0]))
print('y1y1', onp.allclose(y1y1, G_hess[:,:,:,:,1,1]))
print('y1z1', onp.allclose(y1z1, G_hess[:,:,:,:,1,2]))
print('z1x1', onp.allclose(z1x1, G_hess[:,:,:,:,2,0]))
print('z1y1', onp.allclose(z1y1, G_hess[:,:,:,:,2,1]))
print('z1z1', onp.allclose(z1z1, G_hess[:,:,:,:,2,2]))
# atom 2 second derivatives: these are correct
print('x2x2', onp.allclose(x2x2, G_hess[:,:,:,:,3,3]))
print('x2y2', onp.allclose(x2y2, G_hess[:,:,:,:,3,4]))
print('x2z2', onp.allclose(x2z2, G_hess[:,:,:,:,3,5]))
print('y2x2', onp.allclose(y2x2, G_hess[:,:,:,:,4,3]))
print('y2y2', onp.allclose(y2y2, G_hess[:,:,:,:,4,4]))
print('y2z2', onp.allclose(y2z2, G_hess[:,:,:,:,4,5]))
print('z2x2', onp.allclose(z2x2, G_hess[:,:,:,:,5,3]))
print('z2y2', onp.allclose(z2y2, G_hess[:,:,:,:,5,4]))
print('z2z2', onp.allclose(z2z2, G_hess[:,:,:,:,5,5]))


# THIS ATOM BLOCK BELOW IS CORRECT
#print('x1x1', hess[:,:,:,:,0,0])
#print('x1y1', hess[:,:,:,:,0,1])
#print('x1z1', hess[:,:,:,:,0,2])
#print('y1x1', hess[:,:,:,:,1,0])
#print('y1y1', hess[:,:,:,:,1,1])
#print('y1z1', hess[:,:,:,:,1,2])
#print('z1x1', hess[:,:,:,:,2,0])
#print('z1y1', hess[:,:,:,:,2,1])
#print('z1z1', hess[:,:,:,:,2,2])
# THIS ATOM BLOCK ABOVE IS CORRECT

#print(g.shape)
#print(np.allclose(g, gradient))
#print(huh.shape)
#print(np.allclose(huh,gradient))

#what = jax.jacfwd(jax.jacfwd(partial_psi_tei))(geom)
#print(what.shape)

#print(np.allclose(what,hessian))


