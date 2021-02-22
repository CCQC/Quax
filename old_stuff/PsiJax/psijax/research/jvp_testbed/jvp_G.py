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

# Create Primitive 
psi_tei_p = core.Primitive("psi_tei")

# Create function to call the primitive
def psi_tei(geom, deriv, mol, basis_name):
    return psi_tei_p.bind(geom, deriv, mol, basis_name)

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

# Register the primal implementation with JAX
psi_tei_p.def_impl(psi_tei_impl)

# TODO you may need two primitives: TEI and TEI_DERIV
#TODO you can replace molecule object with a multiline string

# Create JVP rule. For now lets assume the derivative wrt
# a cartesian coordinate is that coordinate times the whole G array
def psi_tei_jvp(primals, tangents):
    geom, deriv, mol, basis_name = primals
    primals_out = psi_tei(geom, np.zeros_like(geom), mol, basis_name) 
    print(tangents)

    # Tangents is a standard basis of vectors of size 3N
    # For every geometry cartesian component, use this tangent vector and compute derivative of G wrt
    tei_derivatives = []
    for g_dot in tangents:
        tei_deriv = psi_tei(geom, g_dot + deriv, mol, basis_name)
        tei_derivatives.append(tei_deriv)

    # need tangents for each input, but molecule and basis should not be modified
    geom_tangent = np.concatenate(tei_derivatives, axis=-1)
    # what the heck to do for deriv tangent?
    #deriv_tangent = ?? 
    tangents_out = (geom_tangent, deriv_tangent, mol, basis_name)
    return primals_out, tangents_out

jax.ad.primitive_jvps[psi_tei_p] = psi_tei_jvp
#batching.primitive_batchers[multiply_add_p] = multiply_add_batch

# We now should be able to get TEI's. Let's try it.
molecule = psi4.geometry("""
                         0 1
                         H -0.1 -0.2  -0.849220457955
                         H  0.1  0.2  -1.849220457955
                         no_reorient
                         no_com
                         units bohr
                         """)
# Flatten geometry and deriv
geom = np.asarray(onp.asarray(molecule.geometry())).reshape(-1)
deriv = np.zeros_like(geom)
basis_name = 'sto-3g'
G = psi_tei(geom, deriv, molecule, basis_name)
print(G.shape)
print(G)

# Big problem: need to limit the differentiable arguments.

base_vec = np.array([1.,0.,0.,0.,0.,0.])
jax.jvp(psi_tei, (geom, deriv, molecule, basis_name), (base_vec, deriv, molecule, basis_name))

## reference 
#def psi_tei(geom, deriv_level, mol, basis_name):
#    basis_set = psi4.core.BasisSet.build(mol, 'BASIS', basis_name, puream=0)
#    mints = psi4.core.MintsHelper(basis_set)
#    psi_G = np.asarray(onp.asarray(mints.ao_eri()))
#    return psi_G

# Set up primitive
#ext_tei_p = core.Primitive("ext_tei")
#def ext_tei(geom, basis, deriv):
#    return ext_tei_p.bind(geom, basis, deriv)


def ext_tei(geom,basis,deriv):
    """
    Call 'external library' (TODO: actually just use your code for starters) for TEI array and TEI partial derivatives 
    Parameters:
    -----------
    geom : Flattened? N x 3 array of cartesian coordinates in bohr
    basis : dictionary of basis 
    deriv : an array the same shape as geom, with an integer in each cartesian coordinate position to indicate which partial derivative to take
            Example [1,0,0,0,...,0] would be to take the TEI gradient wrt first cartesian coordinate
                    [0,1,0,1,...,0] would be to take the TEI second derivative wrt 2nd and 4th cartesian coordinates
                    [0,0,0,0,...,0] would be the usual TEI array 
    Returns:
    -------
    An array of shape (nbf,nbf,nbf,nbf), either is a partial derivative of original TEI array depending on contents of 'deriv'
    """
    # Prepare geom, basis 
    # Pass data to libint
    # omggivemeintslibintplz.jpg
    return G 


def ext_tei_jvp(primals, tangents):
    #TODO just a sketch, not sure if this is right
    geom, basis, deriv = primals
    primals_out = ext_tei(geom,basis,np.zeros_like(geom))
    
    tei_derivatives = []
    for g_dot in tangents:
        tei_deriv = ext_tei(geom, basis, g_dot + deriv)
        tei_derivatives.append(tei_deriv)
    
    tangents_out = np.concatenate(tei_derivatives, axis=-1)
    return primals_out, tangents_out

