import psijax
import psi4
import jax
from jax.config import config; config.update("jax_enable_x64", True)
from psijax.integrals.basis_utils import build_basis_set
from psijax.integrals.tei import tei_array
from psijax.integrals.oei import oei_arrays
from psijax.methods.hartree_fock import restricted_hartree_fock
import jax.numpy as np
import numpy as onp
import os
np.set_printoptions(linewidth=800)

molecule = psi4.geometry("""
                         0 1
                         H  0.0  1.0  0.0 
                         H  0.0  0.0  1.0
                         units ang 
                         """)
# NOTE flattened geometry
geom = onp.asarray(molecule.geometry())
geomflat = np.asarray(geom.flatten())
basis_name = 'cc-pvdz'
xyz_file_name = "geom.xyz"
# Save xyz file, get path
molecule.save_xyz_file(xyz_file_name, True)
xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
mints = psi4.core.MintsHelper(basis_set)
basis_dict = build_basis_set(molecule, basis_name)
charge = molecule.molecular_charge()
nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])

# Old integrals 
def wrap_oeis(geomflat):
    geom = geomflat.reshape(-1,3)
    S, T, V = oei_arrays(geom,basis_dict,nuclear_charges)
    return S, T, V

# TEI's
def wrap(geomflat):
    geom = geomflat.reshape(-1,3)
    return tei_array(geom, basis_dict) 

# New libint interface gradients and hessians
# Initialize Libint
psijax.external_integrals.libint_initialize(xyz_path, basis_name)
#
#new_overlap_grad = jax.jacfwd(psijax.external_integrals.overlap)(geomflat)
#new_kinetic_grad = jax.jacfwd(psijax.external_integrals.kinetic)(geomflat)
#new_potential_grad = jax.jacfwd(psijax.external_integrals.potential)(geomflat)
#
#new_overlap_hess = jax.jacfwd(jax.jacfwd(psijax.external_integrals.overlap))(geomflat)
#new_kinetic_hess = jax.jacfwd(jax.jacfwd(psijax.external_integrals.kinetic))(geomflat)
#new_potential_hess = jax.jacfwd(jax.jacfwd(psijax.external_integrals.potential))(geomflat)
#
#overlap_grad, kinetic_grad, potential_grad = jax.jacfwd(wrap_oeis)(geomflat)
#
#print("overlap gradients match ", onp.allclose(overlap_grad, new_overlap_grad))
#print("kinetic gradients match ", onp.allclose(kinetic_grad, new_kinetic_grad))
#print("potential gradients match ", onp.allclose(potential_grad, new_potential_grad))
#
#overlap_hess, kinetic_hess, potential_hess = jax.jacfwd(jax.jacfwd(wrap_oeis))(geomflat)
#
#print("overlap hessians match ", onp.allclose(overlap_hess, new_overlap_hess))
#print("kinetic hessians match ", onp.allclose(kinetic_hess, new_kinetic_hess))
#print("potential hessians match ", onp.allclose(potential_hess, new_potential_hess))


new_tei_grad = jax.jacfwd(psijax.external_integrals.tei)(geomflat)
new_tei_hess = jax.jacfwd(jax.jacfwd(psijax.external_integrals.tei))(geomflat)

tei_grad = jax.jacfwd(wrap)(geomflat)
print("ERI gradients match ", onp.allclose(tei_grad, new_tei_grad))
tei_hess = jax.jacfwd(jax.jacfwd(wrap))(geomflat)
print("ERI hessians match ", onp.allclose(tei_hess, new_tei_hess))

## Cubics 
## OKAY LIBINT ENGINE DOES NOT SUPPORT POTENTIALS
## New code
#new_overlap_cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(psijax.external_integrals.overlap)))(geomflat)
#new_kinetic_cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(psijax.external_integrals.kinetic)))(geomflat)
##new_potential_cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(psijax.external_integrals.potential)))(geomflat)
new_tei_cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(psijax.external_integrals.tei)))(geomflat)
#
### Old code
#overlap_cube, kinetic_cube, potential_cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(wrap_oeis)))(geomflat)
tei_cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(wrap)))(geomflat)
#
#print("overlap cubics match ", onp.allclose(overlap_cube, new_overlap_cube))
#print("kinetic cubics match ", onp.allclose(kinetic_cube, new_kinetic_cube))
##print("potential cubics match ", onp.allclose(potential_cube, new_potential_cube))
print("ERI cubics match ", onp.allclose(tei_cube, new_tei_cube))
#
### Quartics 
### New code
#new_overlap_quartic = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(psijax.external_integrals.overlap))))(geomflat)
#new_kinetic_quartic = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(psijax.external_integrals.kinetic))))(geomflat)
##new_potential_quartic = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(psijax.external_integrals.potential))))(geomflat)
new_tei_quartic = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(psijax.external_integrals.tei))))(geomflat)
#
### Old code
#overlap_quartic, kinetic_quartic, potential_quartic = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(wrap_oeis))))(geomflat)
tei_quartic = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(wrap))))(geomflat)

#print("overlap quartics match ", onp.allclose(overlap_quartic, new_overlap_quartic))
#print("kinetic quartics match ", onp.allclose(kinetic_quartic, new_kinetic_quartic))
#print("potential quartics match ", onp.allclose(potential_quartic, new_potential_quartic))
print("ERI quartics match ", onp.allclose(tei_quartic, new_tei_quartic))
#
## Finalize Libint
psijax.external_integrals.libint_finalize()


