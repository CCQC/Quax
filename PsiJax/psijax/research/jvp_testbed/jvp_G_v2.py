import psi4
psi4.core.be_quiet()
import jax 
from jax import custom_jvp
from jax.interpreters import batching
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)
#config.enable_omnistaging()
import jax.numpy as np
from jax.experimental import loops
from basis_utils import flatten_basis_data, get_nbf
from jax import core
import numpy as onp

# Here I am trying to avoid funny business with arguments by using partial
molecule = psi4.geometry("""
                         0 1
                         H  1.0  2.0 3.0
                         H -1.0 -2.0 -3.0
                         no_reorient
                         no_com
                         units bohr
                         """)
# Flatten geometry and deriv
geom = np.asarray(onp.asarray(molecule.geometry())).reshape(-1)
deriv = np.zeros_like(geom)
basis_name = 'sto-3g'


# Create Primitive 
psi_tei_p = core.Primitive("psi_tei")

# Create function to call the primitive
def psi_tei(geom, deriv):
    return psi_tei_p.bind(geom, deriv)

# Create primal evaluation rule, which knows how to evaluate two electron integrals and any cartesian coordinate partial derivatives.
def psi_tei_impl(geom, deriv, mol, basis_name):
    """
    geom : flattened geometry array
    deriv : flattened array of integers indicating which cartesian coordinates to take partial deriv of 
    mol : psi4 molecule object corresponding to geometry
    basis_name: a string of basis set name, cc-pvdz, sto-3g, etc
    """
    basis_set = psi4.core.BasisSet.build(mol, 'BASIS', basis_name, puream=0)
    mints = psi4.core.MintsHelper(basis_set)
    psi_G = np.asarray(onp.asarray(mints.ao_eri()))
    # FOR NOW: Dummy partial derivative code: derivative of G wrt a cartesian coordinate 
    # is G plus G * geom parameter. 
    dummy = deriv * geom
    G = np.kron(dummy, np.expand_dims(psi_G, axis=-1))
    G = psi_G + np.sum(G, axis=-1)
    return G

# Register the primal implementation with JAX, but use partial to block out funky non jax stuff, like molecule and basis string
new = partial(psi_tei_impl, mol=molecule, basis_name=basis_name)  
psi_tei_p.def_impl(new)

# Now we can try out our primitive which has a fixed molecule, basis and deriv level
G = psi_tei(geom, deriv)
print(G)

# Now the JVP only needs to refer to the geometry and deriv objects, which makes life easier for now.
# Create JVP rule. For now lets assume the derivative wrt
# a cartesian coordinate is that coordinate times the whole G array
def psi_tei_jvp(primals, tangents):
    geom, deriv = primals
    primals_out = psi_tei(geom, np.zeros_like(geom)) 
    # Here the two components of tangents should be the same;
    # after all, of course the basis tangent vector of geom should be the same as the deriv level
    # focus on fisrt derivatives for now, then think about it generalizing to higher order 

    # Tangents is a standard basis of vectors of size 3N
    # For every geometry cartesian component, use this tangent vector and compute derivative of G wrt
    tei_derivatives = []
    for g_dot in tangents:
        #TODO here is the key: how to setup for higher order diff
        #tei_deriv = psi_tei(geom, g_dot + deriv, mol, basis_name)
        tei_deriv = psi_tei(geom, g_dot + deriv)
        tei_derivatives.append(tei_deriv)

    # need tangents for each input, but molecule and basis should not be modified
    geom_tangent = np.concatenate(tei_derivatives, axis=-1)
    # what the heck to do for deriv tangent?
    deriv_tangent = deriv
    tangents_out = (geom_tangent, deriv_tangent)
    return primals_out, tangents_out

jax.ad.primitive_jvps[psi_tei_p] = psi_tei_jvp
#batching.primitive_batchers[multiply_add_p] = multiply_add_batch

base_vec = np.array([1.,0.,0.,0.,0.,0.])
deriv = np.array([1.,0.,0.,0.,0.,0.])
blah = jax.jvp(psi_tei, (geom, deriv), (base_vec, deriv))
print(blah)

# I need to figure out a way for nested jvp's to pass along 'tangents' that really just compute the appropriate derivative

# Cheating batching rule, this is general but may not be appropriate for this case
# This dont work
#jax.interpreters.batching.defvectorized(psi_tei_p)

#what = jax.jacfwd(psi_tei, 0)(geom, deriv)
#print(what)

