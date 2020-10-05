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
    # Let the partial derivative wrt parameters just equal
    # G times a coefficient equal to prod(geom ^ deriv_vec), so you just multiply
    # by each geom coordinate 'deriv_vec' times.
    G = psi_tei(geom, **params)
    dG_di = G * onp.prod(onp.power(geom, deriv_vec)) 
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
test2 = psi_tei_deriv(geom, deriv_test, **params) 

# Next step: create Jacobian-vector product rules, which given some input args (primals)
# and a tangent std basis vector (tangent), returns the function evaluated at that point (primals_out)
# and the slice of the Jacobian (tangents_out)

def psi_tei_jvp(primals, tangents, **params):
    mol = params['mol']
    basis_name = params['basis_name']
    geom, = primals
    primals_out = psi_tei(geom, **params) 
    #print(tangents)
    #tei_derivatives = []
    #for g_dot in tangents:
    #    tei_deriv = psi_tei_deriv(geom, g_dot, mol=mol, basis_name=basis_name)
    #    tei_derivatives.append(tei_deriv)
    #tangents_out = np.concatenate(tei_derivatives, axis=-1)
    tangents_out = psi_tei_deriv(geom, tangents[0], mol=mol, basis_name=basis_name)
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

# Currently I'm using partial here, but the real jax.jacfwd should work with explicit params for molecule and basis name, since you can pick which arg to differentiate
gradient = my_jacfwd_novmap(partial_psi_tei)(geom)
hessian = my_jacfwd_novmap(my_jacfwd_novmap(partial_psi_tei))(geom)
#cubic = my_jacfwd_novmap(my_jacfwd_novmap(my_jacfwd_novmap(partial_psi_tei)))(geom)

# Okay, lets check it. Gradients should just be the TEI array times the coordinate factor that you're differentiating wrt
TEI = psi_tei(geom, **params)
print(onp.allclose(gradient[...,0], TEI * geom[0]))
print(onp.allclose(gradient[...,1], TEI * geom[1]))
print(onp.allclose(gradient[...,2], TEI * geom[2]))
print(onp.allclose(gradient[...,3], TEI * geom[3]))
print(onp.allclose(gradient[...,4], TEI * geom[4]))
print(onp.allclose(gradient[...,5], TEI * geom[5]))

print(onp.allclose(hessian[...,0,0], TEI * geom[0] * geom[0]))
print(onp.allclose(hessian[...,0,1], TEI * geom[0] * geom[1]))
print(onp.allclose(hessian[...,0,2], TEI * geom[0] * geom[2]))
print(onp.allclose(hessian[...,0,3], TEI * geom[0] * geom[3]))
print(onp.allclose(hessian[...,0,4], TEI * geom[0] * geom[4]))
print(onp.allclose(hessian[...,0,5], TEI * geom[0] * geom[5]))

print(onp.allclose(hessian[...,1,0], TEI * geom[1] * geom[0]))
print(onp.allclose(hessian[...,1,1], TEI * geom[1] * geom[1]))
print(onp.allclose(hessian[...,1,2], TEI * geom[1] * geom[2]))
print(onp.allclose(hessian[...,1,3], TEI * geom[1] * geom[3]))
print(onp.allclose(hessian[...,1,4], TEI * geom[1] * geom[4]))
print(onp.allclose(hessian[...,1,5], TEI * geom[1] * geom[5]))

print(onp.allclose(hessian[...,2,0], TEI * geom[2] * geom[0]))
print(onp.allclose(hessian[...,2,1], TEI * geom[2] * geom[1]))
print(onp.allclose(hessian[...,2,2], TEI * geom[2] * geom[2]))
print(onp.allclose(hessian[...,2,3], TEI * geom[2] * geom[3]))
print(onp.allclose(hessian[...,2,4], TEI * geom[2] * geom[4]))
print(onp.allclose(hessian[...,2,5], TEI * geom[2] * geom[5]))

print(onp.allclose(hessian[...,3,0], TEI * geom[3] * geom[0]))
print(onp.allclose(hessian[...,3,1], TEI * geom[3] * geom[1]))
print(onp.allclose(hessian[...,3,2], TEI * geom[3] * geom[2]))
print(onp.allclose(hessian[...,3,3], TEI * geom[3] * geom[3]))
print(onp.allclose(hessian[...,3,4], TEI * geom[3] * geom[4]))
print(onp.allclose(hessian[...,3,5], TEI * geom[3] * geom[5]))

print(onp.allclose(hessian[...,4,0], TEI * geom[4] * geom[0]))
print(onp.allclose(hessian[...,4,1], TEI * geom[4] * geom[1]))
print(onp.allclose(hessian[...,4,2], TEI * geom[4] * geom[2]))
print(onp.allclose(hessian[...,4,3], TEI * geom[4] * geom[3]))
print(onp.allclose(hessian[...,4,4], TEI * geom[4] * geom[4]))
print(onp.allclose(hessian[...,4,5], TEI * geom[4] * geom[5]))

print(onp.allclose(hessian[...,5,0], TEI * geom[5] * geom[0]))
print(onp.allclose(hessian[...,5,1], TEI * geom[5] * geom[1]))
print(onp.allclose(hessian[...,5,2], TEI * geom[5] * geom[2]))
print(onp.allclose(hessian[...,5,3], TEI * geom[5] * geom[3]))
print(onp.allclose(hessian[...,5,4], TEI * geom[5] * geom[4]))
print(onp.allclose(hessian[...,5,5], TEI * geom[5] * geom[5]))


#print(onp.allclose(cubic[...,5,5,5], TEI * geom[5] * geom[5] * geom[5]))


