import psi4
import psijax
import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from psijax.integrals.basis_utils import build_basis_set
from psijax.integrals.tei import tei_array
from psijax.integrals.oei import oei_arrays
from psijax.methods.hartree_fock import restricted_hartree_fock
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
basis_name = 'sto-3g'
xyz_file_name = "geom.xyz"

# Save xyz file, get path
molecule.save_xyz_file(xyz_file_name, True)
xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
mints = psi4.core.MintsHelper(basis_set)
basis_dict = build_basis_set(molecule, basis_name)
charge = molecule.molecular_charge()
nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])


# Iinitilzie
psijax.external_integrals.external_oei.libint_init(xyz_path, basis_name)

# Build integral arrays
overlap = psijax.external_integrals.external_oei.psi_overlap(geomflat)
print(overlap)
kinetic = psijax.external_integrals.external_oei.psi_kinetic(geomflat)
print(kinetic)

new_overlap_grad = jax.jacfwd(psijax.external_integrals.external_oei.psi_overlap)(geomflat)
new_potential_grad = jax.jacfwd(psijax.external_integrals.external_oei.psi_potential)(geomflat)
new_kinetic_grad = jax.jacfwd(psijax.external_integrals.external_oei.psi_kinetic)(geomflat)

#print(new_overlap_grad)

# Finalize 
psijax.external_integrals.external_oei.libint_finalize()

