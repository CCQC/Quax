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

# Strategy: define a TEI primitive, and TEI derivative primitive.
# The JVP rule of TEI primitive will call TEI derivative primitive
# The TEI derivative JVP rule will call TEI derivative itself 

# TODO eventually, you can replace mol and basis with a mints object
# This is whats gonna be doing everything: mints.ao_eri() mints.ao_eri_partial_derivative()

# Create primitives
psi_tei_p = core.Primitive("psi_tei")
psi_tei_deriv_p = core.Primitive("psi_tei_deriv")

# Create functions to call primitives
# here params holds the molecule object and basis name string
def psi_tei(geom, **params):
    return psi_tei_p.bind(geom, **params)

# here params holds molecule object and basis name string
def psi_tei_deriv(geom, deriv_vec, **params):
    return psi_tei_deriv_p.bind(geom, deriv_vec, **params) 

# Create primitive evaluation rules 
def psi_tei_impl(geom, **params):
    mol = params['mol']
    basis_name = params['basis_name']
    basis_set = psi4.core.BasisSet.build(mol, 'BASIS', basis_name, puream=0)
    mints = psi4.core.MintsHelper(basis_set)
    psi_G = np.asarray(onp.asarray(mints.ao_eri()))
    return psi_G

def psi_tei_deriv_impl(geom, deriv_vec, **params):
    # For now, we use a dummy partial derivative code
    # and just assume the derivative of G wrt a cartesian coordinate 
    # is G plus G * cartesian coordinate. 
    # The 'deriv' vector says which coords to differentiate wrt to
    psi_G = psi_tei(geom, **params)
    dummy = deriv_vec * geom
    G = np.kron(dummy, np.expand_dims(psi_G, axis=-1))
    dG_di = psi_G + np.sum(G, axis=-1)
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
basis_name = 'sto-3g'
deriv_test = np.array([1.,0.,0.,0.,0.,0.])
params = {'mol': molecule, 'basis_name': basis_name}
test = psi_tei(geom, **params)
print(test)
test2 = psi_tei_deriv(geom, deriv_test, **params) 
print(test2)

# Next step: create Jacobian-vector product rules, which given some input args
# and a tangent std basis vector, returns the function evaluated at that point
# and the slice of the Jacobian

# I guess this is working? A bit trippy though.
def psi_tei_jvp(primals, tangents, **params):
    mol = params['mol']
    basis_name = params['basis_name']
    geom, = primals
    primals_out = psi_tei(geom, **params) 

    tei_derivatives = []
    for g_dot in tangents:
        tei_deriv = psi_tei_deriv(geom, g_dot, mol=mol, basis_name=basis_name)
        tei_derivatives.append(tei_deriv)
    tangents_out = np.concatenate(tei_derivatives, axis=-1)
    return primals_out, tangents_out

def psi_tei_deriv_jvp(primals, tangents, **params):
    mol = params['mol']
    basis_name = params['basis_name']
    #deriv = params['deriv']
    geom, deriv_vec = primals
    primals_out = psi_tei_deriv(geom, deriv_vec, mol=mol,basis_name=basis_name)
    #tei_derivatives = []
    #for g_dot in tangents:
    #    new_deriv = g_dot + deriv_vec
    #    tei_deriv = psi_tei_deriv(geom, new_deriv, mol=mol, basis_name=basis_name)
    #    tei_derivatives.append(tei_deriv)
    # need tangents for each input, but molecule and basis should not be modified
    #tangents_out = np.concatenate(tei_derivatives, axis=-1)

    # Hmm.. some random Zero(6) object is in tangents, messing everything up... Can I just ignore it?
    tangents_out = psi_tei_deriv(geom, deriv_vec + tangents[0], mol=mol, basis_name=basis_name)
    #tangents_out = derivative
    return primals_out, tangents_out


# Register the JVP rules with JAX
jax.ad.primitive_jvps[psi_tei_p] = psi_tei_jvp
jax.ad.primitive_jvps[psi_tei_deriv_p] = psi_tei_deriv_jvp

# We can only call JVP on the partial. I think jacfwd handles this automatically...
partial_psi_tei = partial(psi_tei, mol=molecule,basis_name=basis_name)
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

#what = my_jacfwd_novmap(psi_tei)(geom, **params)
#print(what)

what = my_jacfwd_novmap(partial_psi_tei)(geom)
print(what.shape)

huh = my_jacfwd_novmap(my_jacfwd_novmap(partial_psi_tei))(geom)
print(huh.shape)

# Okay, lets check it. Since the first argument
dAx_dAx = huh[:,:,:,:,0,0]

test = psi_tei(geom, **params)




