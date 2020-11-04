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
                         O  0.0  0.0  0.0
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

# New libint interface gradients and hessians
# Initialize Libint
psijax.external_integrals.external_oei.libint_init(xyz_path, basis_name)

new_overlap_grad = jax.jacfwd(psijax.external_integrals.external_oei.psi_overlap)(geomflat)
new_kinetic_grad = jax.jacfwd(psijax.external_integrals.external_oei.psi_kinetic)(geomflat)
new_potential_grad = jax.jacfwd(psijax.external_integrals.external_oei.psi_potential)(geomflat)

new_overlap_hess = jax.jacfwd(jax.jacfwd(psijax.external_integrals.external_oei.psi_overlap))(geomflat)
new_kinetic_hess = jax.jacfwd(jax.jacfwd(psijax.external_integrals.external_oei.psi_kinetic))(geomflat)
new_potential_hess = jax.jacfwd(jax.jacfwd(psijax.external_integrals.external_oei.psi_potential))(geomflat)

# Old integrals 
def wrap_oeis(geomflat):
    geom = geomflat.reshape(-1,3)
    S, T, V = oei_arrays(geom,basis_dict,nuclear_charges)
    return S, T, V
overlap_grad, kinetic_grad, potential_grad = jax.jacfwd(wrap_oeis)(geomflat)

print("overlap gradients match ", onp.allclose(overlap_grad, new_overlap_grad))
print("kinetic gradients match ", onp.allclose(kinetic_grad, new_kinetic_grad))
print("potential gradients match ", onp.allclose(potential_grad, new_potential_grad))

overlap_hess, kinetic_hess, potential_hess = jax.jacfwd(jax.jacfwd(wrap_oeis))(geomflat)

print("overlap hessians match ", onp.allclose(overlap_hess, new_overlap_hess))
print("kinetic hessians match ", onp.allclose(kinetic_hess, new_kinetic_hess))
print("potential hessians match ", onp.allclose(potential_hess, new_potential_hess))

# TEI's
def wrap(geomflat):
    geom = geomflat.reshape(-1,3)
    return tei_array(geom, basis_dict) 

new_tei_grad = jax.jacfwd(psijax.external_integrals.external_tei.psi_tei)(geomflat)
new_tei_hess = jax.jacfwd(jax.jacfwd(psijax.external_integrals.external_tei.psi_tei))(geomflat)

tei_grad = jax.jacfwd(wrap)(geomflat)
print("ERI gradients match ", onp.allclose(tei_grad, new_tei_grad))
tei_hess = jax.jacfwd(jax.jacfwd(wrap))(geomflat)
print("ERI hessians match ", onp.allclose(tei_hess, new_tei_hess))

# Finalize Libint
psijax.external_integrals.external_oei.libint_finalize()

