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

# Create primitives
psi_tei_p = core.Primitive("psi_tei")
psi_tei_deriv_p = core.Primitive("psi_tei_deriv")

# Create functions to call primitives
def psi_tei(geom, molstring, basis_name):
    return psi_tei_p.bind(geom, molstring, basis_name)

def psi_tei_deriv(geom, deriv, molstring, basis_name):
    return psi_tei_deriv_p.bind(geom, deriv, molstring, basis_name)

# Create primitive evaluation rules 
def psi_tei_impl(geom, molstring, basis_name):
    mol = psi4.core.Molecule.from_string(molstring)
    basis_set = psi4.core.BasisSet.build(mol, 'BASIS', basis_name, puream=0)
    mints = psi4.core.MintsHelper(basis_set)
    psi_G = np.asarray(onp.asarray(mints.ao_eri()))
    return psi_G

def psi_tei_deriv_impl(geom, deriv, molstring, basis_name):
    # For now, we use a dummy partial derivative code
    # and just assume the derivative of G wrt a cartesian coordinate 
    # is G plus G * cartesian coordinate. 
    # The 'deriv' vector says which coords to differentiate wrt to
    psi_G = psi_tei(geom, molstring, basis_name)  
    dummy = deriv * geom
    G = np.kron(dummy, np.expand_dims(psi_G, axis=-1))
    dG_di = psi_G + np.sum(G, axis=-1)
    return dG_di

# Register primitive evaluation rule
psi_tei_p.def_impl(psi_tei_impl)
psi_tei_deriv_p.def_impl(psi_tei_deriv_impl)

# Now we can evaluate 
molstring = """
            0 1
            H  1.0  2.0 3.0
            H -1.0 -2.0 -3.0
            no_reorient
            no_com
            units bohr
            """
molecule = psi4.core.Molecule.from_string(molstring)
geom = np.asarray(onp.asarray(molecule.geometry())).reshape(-1)
basis_name = 'sto-3g'
deriv_test = np.array([1.,0.,0.,0.,0.,0.])

test = psi_tei(geom, molstring, basis_name)
print(test)
test2 = psi_tei_deriv(geom, deriv_test, molstring, basis_name)
print(test2)

# Create Jacobian-vector product rule, which given some input args
# and a tangent std basis vector, returns the function evaluated at that point
# and the slice of the Jacobian

def psi_tei_jvp(primals, tangents):
    geom, mol, basis_name = primals
    primals_out = psi_tei(geom, mol, basis_name) 

    tei_derivatives = []
    for g_dot in tangents:
        tei_deriv = psi_tei_deriv(geom, g_dot, mol, basis_name)
        tei_derivatives.append(tei_deriv)

    # need tangents for each input, but molecule and basis should not be modified
    geom_tangent = np.concatenate(tei_derivatives, axis=-1)
    deriv_tangent = tangents
    # what the heck to do for deriv tangent?
    tangents_out = (geom_tangent, deriv_tangent, None, None)
    return primals_out, tangents_out


jax.ad.primitive_jvps[psi_tei_p] = psi_tei_jvp

base_vec = np.array([1.,0.,0.,0.,0.,0.])
#p, t = jax.jvp(psi_tei, (geom, molstring, basis_name), (base_vec, molstring, basis_name))

# Cant differentiate wrt 
p, t  = jax.jvp(partial(psi_tei, geom=geom, molstring=molstring, basis_name=basis_name), (geom,), (base_vec,))



