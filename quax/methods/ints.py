import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jacfwd
import numpy as np
import h5py
import psi4
import os

from ..integrals.basis_utils import build_basis_set
from ..integrals.tei import tei_array 
from ..integrals.oei import oei_arrays

from ..utils import get_deriv_vec_idx, get_required_deriv_vecs

# Check for Libint interface 
from ..constants import libint_imported
if libint_imported:
    from ..external_integrals import TEI 
    from ..external_integrals import OEI 
    from ..external_integrals import libint_interface
     

def compute_integrals(geom, basis_name, xyz_path, nuclear_charges, charge, deriv_order, options):
    # Load integral algo, decides to compute integrals in memory or use disk 
    algo = options['integral_algo']

    if libint_imported and libint_interface.LIBINT2_MAX_DERIV_ORDER >= deriv_order:
        if algo == 'libint_core':
            libint_interface.initialize(xyz_path, basis_name)
            # Precompute TEI derivatives 
            tei_obj = TEI(basis_name, xyz_path, deriv_order, 'core')
            oei_obj = OEI(basis_name, xyz_path, deriv_order, 'core')
            # Compute integrals
            S = oei_obj.overlap(geom)
            T = oei_obj.kinetic(geom)
            # TODO add hotfix for Libint not supporting > 2nd order
            V = oei_obj.potential(geom)
            G = tei_obj.tei(geom)
            libint_interface.finalize()
            return S, T, V, G

        elif algo == 'libint_disk' and deriv_order > 0:
            # Check disk for currently existing integral derivatives 
            check = check_disk(geom,basis_name,xyz_path,deriv_order)

            tei_obj = TEI(basis_name, xyz_path, deriv_order, 'disk')
            oei_obj = OEI(basis_name, xyz_path, deriv_order, 'disk')
            # If disk integral derivs are right, nothing to do
            if check:
                libint_interface.initialize(xyz_path, basis_name)
                S = oei_obj.overlap(geom)
                T = oei_obj.kinetic(geom)
                V = oei_obj.potential(geom)
                G = tei_obj.tei(geom)
                libint_interface.finalize()
            else:
                # Else write integral derivs to disk
                if deriv_order <= 2:
                    libint_interface.initialize(xyz_path, basis_name)
                    libint_interface.oei_deriv_disk(deriv_order)
                    libint_interface.eri_deriv_disk(deriv_order)
                    S = oei_obj.overlap(geom)
                    T = oei_obj.kinetic(geom)
                    V = oei_obj.potential(geom)
                    G = tei_obj.tei(geom)
                    libint_interface.finalize()
                else:
                    # If higher order than 2, LIBINT api does not support potentials 
                    # In this case, use Libint to write TEI's to disk, and do OEI's with Quax
                    libint_interface.initialize(xyz_path, basis_name)
                    libint_interface.eri_deriv_disk(deriv_order)
                    G = tei_obj.tei(geom)
                    libint_interface.finalize()
    
                    with open(xyz_path, 'r') as f:
                        tmp = f.read()
                    molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
                    basis_dict = build_basis_set(molecule, basis_name)
                    S, T, V = oei_arrays(geom.reshape(-1,3),basis_dict,nuclear_charges)
        elif deriv_order == 0:
            libint_interface.initialize(xyz_path, basis_name)
            tei_obj = TEI(basis_name, xyz_path, deriv_order, 'core')
            oei_obj = OEI(basis_name, xyz_path, deriv_order, 'core')
            # Compute integrals
            S = oei_obj.overlap(geom)
            T = oei_obj.kinetic(geom)
            V = oei_obj.potential(geom)
            G = tei_obj.tei(geom)
            libint_interface.finalize()

        # TODO
        #elif algo == 'quax_disk':

        elif algo == 'quax_core':
            with open(xyz_path, 'r') as f:
                tmp = f.read()
            molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
            basis_dict = build_basis_set(molecule, basis_name)
            S, T, V = oei_arrays(geom.reshape(-1,3),basis_dict,nuclear_charges)
            G = tei_array(geom.reshape(-1,3),basis_dict)

    # If Libint not imported or Libint version doesnt support requested deriv order, use Quax integrals
    else:
        with open(xyz_path, 'r') as f:
            tmp = f.read()
        molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
        basis_dict = build_basis_set(molecule, basis_name)
        S, T, V = oei_arrays(geom.reshape(-1,3),basis_dict,nuclear_charges)
        G = tei_array(geom.reshape(-1,3),basis_dict)
    return S, T, V, G

def check_disk(geom,basis_name,xyz_path,deriv_order,address=None):
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

def write_integrals(molecule, basis_name, deriv_order, address):
    geom = jnp.asarray(np.asarray(molecule.geometry()))
    natoms = geom.shape[0]
    geom_list = np.asarray(molecule.geometry()).reshape(-1).tolist()
    charge = molecule.molecular_charge()
    nuclear_charges = jnp.asarray([molecule.charge(i) for i in range(geom.shape[0])])
    basis_dict = build_basis_set(molecule,basis_name)
    kwargs = {"basis_dict":basis_dict,"nuclear_charges":nuclear_charges}

    # Define wrapper functions for computing partial derivatives
    def oei_wrapper(*args, **kwargs):
        geom = jnp.asarray(args)
        basis_dict = kwargs['basis_dict']
        nuclear_charges = kwargs['nuclear_charges']
        S, T, V = oei_arrays(geom.reshape(-1,3),basis_dict,nuclear_charges)
        return S, T, V

    def tei_wrapper(*args, **kwargs):
        geom = jnp.asarray(args)
        basis_dict = kwargs['basis_dict']
        G = tei_array(geom.reshape(-1,3),basis_dict)
        return G

    # Determine the set of all integral derivatives that need to be written 
    # to disk for this computation
    deriv_vecs = get_required_deriv_vecs(natoms, deriv_order, address)
    for deriv_vec in deriv_vecs:
        flat_idx = get_deriv_vec_idx(deriv_vec)
        order = np.sum(deriv_vec)
        # Compute partial derivative integral arrays corresponding to this deriv vec
        if order == 1:
            i = address[0]
            dS, dT, dV = jacfwd(oei_wrapper, i)(*geom_list, **kwargs)
            dG = jacfwd(tei_wrapper, i)(*geom_list, **kwargs)
        elif order == 2:
            i,j = address[0], address[1]
            dS, dT, dV = jacfwd(jacfwd(oei_wrapper, i), j)(*geom_list, **kwargs)
            dG = jacfwd(jacfwd(tei_wrapper, i), j)(*geom_list, **kwargs)
        elif order == 3:
            i,j,k = address[0], address[1], address[2]
            dS, dT, dV = jacfwd(jacfwd(jacfwd(oei_wrapper, i), j), k)(*geom_list, **kwargs)
            dG = jacfwd(jacfwd(jacfwd(tei_wrapper, i), j), k)(*geom_list, **kwargs)
        elif order == 4:
            i,j,k,l = address[0], address[1], address[2], address[3]
            dS, dT, dV= jacfwd(jacfwd(jacfwd(jacfwd(oei_wrapper, i), j), k), l)(*geom_list, **kwargs)
            dG = jacfwd(jacfwd(jacfwd(jacfwd(tei_wrapper, i), j), k), l)(*geom_list, **kwargs)
        elif order == 5:
            i,j,k,l,m = address[0], address[1], address[2], address[3], address[4]
            dS, dT, dV= jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(oei_wrapper, i), j), k), l), m)(*geom_list, **kwargs)
            dG = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(tei_wrapper, i), j), k), l), m)(*geom_list, **kwargs)
        elif order == 6:
            i,j,k,l,m,n = address[0], address[1], address[2], address[3], address[4], address[5]
            dS, dT, dV= jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(oei_wrapper, i), j), k), l), m), n)(*geom_list, **kwargs)
            dG = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(tei_wrapper, i), j), k), l), m), n)(*geom_list, **kwargs)
        # Save partial derivative arrays to disk
        f = h5py.File("oei_partials.h5","a")
        f.create_dataset("overlap_deriv"+str(order)+"_"+str(flat_idx), data=dS)
        f.create_dataset("kinetic_deriv"+str(order)+"_"+str(flat_idx), data=dT)
        f.create_dataset("potential_deriv"+str(order)+"_"+str(flat_idx), data=dV)
        f.close()

        f = h5py.File("eri_partials.h5","a")
        f.create_dataset("eri_deriv"+str(order)+"_"+str(flat_idx), data=dG)
        f.close()



              



    


