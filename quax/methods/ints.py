import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import h5py
import psi4
import os

from ..integrals.basis_utils import build_basis_set
from ..integrals.tei import tei_array 
from ..integrals.oei import oei_arrays

from ..external_integrals import overlap
from ..external_integrals import kinetic
from ..external_integrals import potential
from ..external_integrals import tei
from ..external_integrals import tmp_potential

from ..external_integrals import libint_initialize
from ..external_integrals import libint_finalize

def compute_integrals(geom, basis_name, xyz_path, nuclear_charges, charge, deriv_order):
    if deriv_order > 0:
        # TODO assumes libint is installed
        libint = True
        if libint:
            # First check if integral derivatives are already available
            # TODO this currently does not check geometry 
            if ((os.path.exists("eri_derivs.h5") and os.path.exists("oei_derivs.h5"))):
                print("Found currently existing integral derivatives in your working directory. Trying to use them.")
                oeifile = h5py.File('oei_derivs.h5', 'r')
                erifile = h5py.File('eri_derivs.h5', 'r')
                with open(xyz_path, 'r') as f:
                    tmp = f.read()
                molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
                basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
                nbf = basis_set.nbf()
                # Check if there are `deriv_order` datatsets in the eri file
                # TODO this should be >= right?
                correct_deriv_order = len(erifile) == deriv_order
                # Check nbf dimension of integral arrays
                sample_dataset_name = list(oeifile.keys())[0]
                correct_nbf = oeifile[sample_dataset_name].shape[0] == nbf
                oeifile.close()
                erifile.close()
                correct_int_derivs = correct_deriv_order and correct_nbf
                if correct_int_derivs:
                    print("Integral derivatives appear to be correct. Avoiding recomputation.")
                    libint_initialize(xyz_path, basis_name)
                    S = overlap(geom)
                    T = kinetic(geom)
                    V = potential(geom)
                    G = tei(geom)
                    libint_finalize()
                #TODO this is an absolute MESS
                else:
                    print("Integral derivatives dimensions do not match requested derivative order and/or basis set. Recomputing integral derivatives")
                    if os.path.exists("eri_derivs.h5"):
                        print("Deleting two electron integral derivatives...")
                        os.remove("eri_derivs.h5")
                    if os.path.exists("oei_derivs.h5"):
                        print("Deleting one electron integral derivatives...")
                        os.remove("oei_derivs.h5")
                    if deriv_order <= 2:
                        libint_initialize(xyz_path, basis_name, deriv_order)
                        S = overlap(geom)
                        T = kinetic(geom)
                        V = potential(geom)
                        G = tei(geom)
                        libint_finalize()
                    else:
                        # If higher order, LIBINT api does not support potentials
                        # In this case, use Libint to write TEI's to disk, and do OEI's manually
                        libint_initialize(xyz_path, basis_name)
                        libint_interface.eri_deriv_disk(max_deriv_order)
                        G = tei(geom)
                        libint_finalize()

                        with open(xyz_path, 'r') as f:
                            tmp = f.read()
                        molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
                        basis_dict = build_basis_set(molecule, basis_name)
                        S, T, V = oei_arrays(geom.reshape(-1,3),basis_dict,nuclear_charges)
                    
            else:
                # TODO this is only required since libint API does not expose potential integrals > 2nd derivs
                if deriv_order <= 2:
                    libint_initialize(xyz_path, basis_name, deriv_order)
                    S = overlap(geom)
                    T = kinetic(geom)
                    V = potential(geom)
                    G = tei(geom)
                    libint_finalize()
                else:
                    # If higher order, LIBINT api does not support potentials
                    # In this case, use Libint to write TEI's to disk, and do OEI's manually
                    libint_initialize(xyz_path, basis_name)
                    libint_interface.eri_deriv_disk(max_deriv_order)
                    G = tei(geom)
                    libint_finalize()

                    with open(xyz_path, 'r') as f:
                        tmp = f.read()
                    molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
                    basis_dict = build_basis_set(molecule, basis_name)
                    S, T, V = oei_arrays(geom.reshape(-1,3),basis_dict,nuclear_charges)

        # Use local integrals implementation
        else:
            with open(xyz_path, 'r') as f:
                tmp = f.read()
            molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
            basis_dict = build_basis_set(molecule, basis_name)
            S, T, V = oei_arrays(geom.reshape(-1,3),basis_dict,nuclear_charges)
            G = tei_array(geom.reshape(-1,3),basis_dict)

    # If deriv_order = 0, it is an energy computation.
    else:
        libint_initialize(xyz_path, basis_name)
        S = overlap(geom)
        T = kinetic(geom)
        V = potential(geom)
        G = tei(geom)
        libint_finalize()
    return S, T, V, G
        
        


