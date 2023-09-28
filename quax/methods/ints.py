import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jacfwd
import numpy as np
import h5py
import psi4
import os

from ..utils import get_deriv_vec_idx, get_required_deriv_vecs

# Check for Libint interface
from ..integrals import TEI
from ..integrals import OEI
from ..integrals import libint_interface
     

def compute_integrals(geom, basis_name, xyz_path, nuclear_charges, charge, deriv_order, options):
    # Load integral algo, decides to compute integrals in memory or use disk 
    algo = options['integral_algo']
    libint_interface.initialize(xyz_path, basis_name, basis_name, basis_name, basis_name)

    if algo == 'libint_disk':
        # Check disk for currently existing integral derivatives
        check = check_disk(geom, basis_name, xyz_path, deriv_order)

        tei_obj = TEI(basis_name, basis_name, basis_name, basis_name, xyz_path, deriv_order, 'disk')
        oei_obj = OEI(basis_name, basis_name, xyz_path, deriv_order, 'disk')
        # If disk integral derivs are right, nothing to do
        if check:
            S = oei_obj.overlap(geom)
            T = oei_obj.kinetic(geom)
            V = oei_obj.potential(geom)
            G = tei_obj.eri(geom)
        else:
            libint_interface.oei_deriv_disk(deriv_order)
            libint_interface.eri_deriv_disk(deriv_order)
            S = oei_obj.overlap(geom)
            T = oei_obj.kinetic(geom)
            V = oei_obj.potential(geom)
            G = tei_obj.eri(geom)

    else:
        # Precompute TEI derivatives
        tei_obj = TEI(basis_name, basis_name, basis_name, basis_name, xyz_path, deriv_order, 'core')
        oei_obj = OEI(basis_name, basis_name, xyz_path, deriv_order, 'core')
        # Compute integrals
        S = oei_obj.overlap(geom)
        T = oei_obj.kinetic(geom)
        V = oei_obj.potential(geom)
        G = tei_obj.eri(geom)

    libint_interface.finalize()
    return S, T, V, G

def check_disk(geom, basis_name, xyz_path, deriv_order, address=None):
    # TODO need to check geometry and basis set name in addition to nbf
    # First check TEI's, then OEI's, return separately, check separately in compute_integrals
    correct_int_derivs = False

    if ((os.path.exists("eri_derivs.h5") and os.path.exists("oei_derivs.h5"))):
        print("Found currently existing integral derivatives in your working directory. Trying to use them.")
        oeifile = h5py.File('oei_derivs.h5', 'r')
        erifile = h5py.File('eri_derivs.h5', 'r')
        with open(xyz_path, 'r') as f:
            tmp = f.read()
        molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
        basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
        nbf = basis_set.nbf()
        # Check if there are `deriv_order` datasets in the eri file
        correct_deriv_order = len(erifile) == deriv_order
        # Check nbf dimension of integral arrays
        sample_dataset_name = list(oeifile.keys())[0]
        correct_nbf = oeifile[sample_dataset_name].shape[0] == nbf
        oeifile.close()
        erifile.close()
        correct_int_derivs = correct_deriv_order and correct_nbf
        if correct_int_derivs:
            print("Integral derivatives appear to be correct. Avoiding recomputation.")

    # TODO flesh out this logic for determining if partials file contains all integrals needed
    # for particular address
    elif ((os.path.exists("eri_partials.h5") and os.path.exists("oei_partials.h5"))):
        print("Found currently existing partial derivatives in working directory. Assuming they are correct.") 
        oeifile = h5py.File('oei_partials.h5', 'r')
        erifile = h5py.File('eri_partials.h5', 'r')
        with open(xyz_path, 'r') as f:
            tmp = f.read()
        molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
        basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
        nbf = basis_set.nbf()
        sample_dataset_name = list(oeifile.keys())[0]
        correct_nbf = oeifile[sample_dataset_name].shape[0] == nbf
        correct_int_derivs = correct_nbf
    return correct_int_derivs
