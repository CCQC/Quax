import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jacfwd
import numpy as np
import h5py
import psi4
import os

# Check for Libint interface
from ..integrals import TEI
from ..integrals import OEI
from ..integrals import libint_interface
     

def compute_integrals(geom, basis_set, xyz_path, deriv_order, options):
    # Load integral algo, decides to compute integrals in memory or use disk 
    algo = options['integral_algo']
    basis_name = basis_set.name()
    libint_interface.initialize(xyz_path, basis_name, basis_name, basis_name, basis_name, options['ints_tolerance'])

    if algo == 'libint_disk':
        # Check disk for currently existing integral derivatives
        check_oei = check_oei_disk("all", basis_set, basis_set, deriv_order)
        check_tei = check_tei_disk("eri", basis_set, basis_set, basis_set, basis_set, deriv_order)

        oei_obj = OEI(basis_set, basis_set, xyz_path, deriv_order, 'disk')
        tei_obj = TEI(basis_set, basis_set, basis_set, basis_set, xyz_path, deriv_order, options, 'disk')
        # If disk integral derivs are right, nothing to do
        if check_oei:
            S = oei_obj.overlap(geom)
            T = oei_obj.kinetic(geom)
            V = oei_obj.potential(geom)
        else:
            libint_interface.oei_deriv_disk(deriv_order)
            S = oei_obj.overlap(geom)
            T = oei_obj.kinetic(geom)
            V = oei_obj.potential(geom)

        if check_tei:
            G = tei_obj.eri(geom)
        else:
            libint_interface.compute_2e_deriv_disk("eri", 0., deriv_order)
            G = tei_obj.eri(geom)

    else:
        # Precompute TEI derivatives
        oei_obj = OEI(basis_set, basis_set, xyz_path, deriv_order, 'core')
        tei_obj = TEI(basis_set, basis_set, basis_set, basis_set, xyz_path, deriv_order, options, 'core')
        # Compute integrals
        S = oei_obj.overlap(geom)
        T = oei_obj.kinetic(geom)
        V = oei_obj.potential(geom)
        G = tei_obj.eri(geom)

    libint_interface.finalize()
    return S, T, V, G

def compute_f12_oeints(geom, basis1, basis2, xyz_path, deriv_order, options, cabs):
    # Load integral algo, decides to compute integrals in memory or use disk
    algo = options['integral_algo']
    basis1_name = basis1.name()
    basis2_name = basis2.name()
    libint_interface.initialize(xyz_path, basis1_name, basis2_name, basis1_name, basis2_name, options['ints_tolerance'])

    if cabs:
        if algo == 'libint_disk':
            # Check disk for currently existing integral derivatives
            check = check_oei_disk("overlap", basis1, basis2, deriv_order)
    
            oei_obj = OEI(basis1, basis2, xyz_path, deriv_order, 'disk')
            # If disk integral derivs are right, nothing to do
            if check:
                S = oei_obj.overlap(geom)
            else:
                libint_interface.compute_1e_deriv_disk("overlap", deriv_order)
                S = oei_obj.overlap(geom)

        else:
            # Precompute OEI derivatives
            oei_obj = OEI(basis1, basis2, xyz_path, deriv_order, 'f12')
            # Compute integrals
            S = oei_obj.overlap(geom)
        
        libint_interface.finalize()
        return S

    else:
        if algo == 'libint_disk':
            # Check disk for currently existing integral derivatives
            check_T = check_oei_disk("kinetic", basis1, basis2, deriv_order)
            check_V = check_oei_disk("potential", basis1, basis2, deriv_order)

            oei_obj = OEI(basis1, basis2, xyz_path, deriv_order, 'disk')
            # If disk integral derivs are right, nothing to do
            if check_T:
                T = oei_obj.kinetic(geom)
            else:
                libint_interface.compute_1e_deriv_disk("kinetic",deriv_order)
                T = oei_obj.kinetic(geom)

            if check_V:
                V = oei_obj.potential(geom)
            else:
                libint_interface.compute_1e_deriv_disk("potential", deriv_order)
                V = oei_obj.potential(geom)

        else:
            # Precompute OEI derivatives
            oei_obj = OEI(basis1, basis2, xyz_path, deriv_order, 'f12')
            # Compute integrals
            T = oei_obj.kinetic(geom)
            V = oei_obj.potential(geom)
        
        libint_interface.finalize()
        return T, V

def compute_f12_teints(geom, basis1, basis2, basis3, basis4, int_type, xyz_path, deriv_order, options):
    # Load integral algo, decides to compute integrals in memory or use disk
    algo = options['integral_algo']
    beta = options['beta']
    basis1_name = basis1.name()
    basis2_name = basis2.name()
    basis3_name = basis3.name()
    basis4_name = basis4.name()
    libint_interface.initialize(xyz_path, basis1_name, basis2_name, basis3_name, basis4_name, options['ints_tolerance'])

    if algo == 'libint_disk':
        # Check disk for currently existing integral derivatives
        check = check_tei_disk(int_type, basis1, basis2, basis3, basis4, deriv_order)

        tei_obj = TEI(basis1, basis2, basis3, basis4, xyz_path, deriv_order, options, 'disk')
        # If disk integral derivs are right, nothing to do
        if check:
            match int_type:
                case "f12":
                    F = tei_obj.f12(geom, beta)
                case "f12_squared":
                    F = tei_obj.f12_squared(geom, beta)
                case "f12g12":
                    F = tei_obj.f12g12(geom, beta)
                case "f12_double_commutator":
                    F = tei_obj.f12_double_commutator(geom, beta)
                case "eri":
                    F = tei_obj.eri(geom)
        else:
            match int_type:
                case "f12":
                    libint_interface.compute_2e_deriv_disk(int_type, beta, deriv_order)
                    F = tei_obj.f12(geom, beta)
                case "f12_squared":
                    libint_interface.compute_2e_deriv_disk(int_type, beta, deriv_order)
                    F = tei_obj.f12_squared(geom, beta)
                case "f12g12":
                    libint_interface.compute_2e_deriv_disk(int_type, beta, deriv_order)
                    F = tei_obj.f12g12(geom, beta)
                case "f12_double_commutator":
                    libint_interface.compute_2e_deriv_disk(int_type, beta, deriv_order)
                    F = tei_obj.f12_double_commutator(geom, beta)
                case "eri":
                    libint_interface.compute_2e_deriv_disk(int_type, 0., deriv_order)
                    F = tei_obj.eri(geom)

    else:
        # Precompute TEI derivatives
        tei_obj = TEI(basis1, basis2, basis3, basis4, xyz_path, deriv_order, options, 'f12')
        # Compute integrals
        match int_type:
            case "f12":
                F = tei_obj.f12(geom, beta)
            case "f12_squared":
                F = tei_obj.f12_squared(geom, beta)
            case "f12g12":
                F = tei_obj.f12g12(geom, beta)
            case "f12_double_commutator":
                F = tei_obj.f12_double_commutator(geom, beta)
            case "eri":
                F = tei_obj.eri(geom)

    libint_interface.finalize()
    return F

def check_oei_disk(int_type, basis1, basis2, deriv_order, address=None):
    # Check OEI's in compute_integrals
    correct_int_derivs = False
    correct_nbf1 = correct_nbf2 = correct_deriv_order = False

    if ((os.path.exists("oei_derivs.h5"))):
        print("Found currently existing one-electron integral derivatives in your working directory. Trying to use them.")
        oeifile = h5py.File('oei_derivs.h5', 'r')
        nbf1 = basis1.nbf()
        nbf2 = basis2.nbf()

        if int_type == "all":
            oei_name = ["overlap_" + str(nbf1) + "_" + str(nbf2) + "_deriv" + str(deriv_order),\
                        "kinetic_" + str(nbf1) + "_" + str(nbf2) + "_deriv" + str(deriv_order),\
                        "potential_" + str(nbf1) + "_" + str(nbf2) + "_deriv" + str(deriv_order)]
        else:
            oei_name = int_type + "_" + str(nbf1) + "_" + str(nbf2) + "_deriv" + str(deriv_order)

        for name in list(oeifile.keys()):
            if name in oei_name:
                correct_nbf1 = oeifile[name].shape[0] == nbf1
                correct_nbf2 = oeifile[name].shape[1] == nbf2
                correct_deriv_order = True
        oeifile.close()

        correct_int_derivs = correct_deriv_order and correct_nbf1 and correct_nbf2

    if correct_int_derivs:
        print("Integral derivatives appear to be correct. Avoiding recomputation.")
    return correct_int_derivs

"""     # TODO flesh out this logic for determining if partials file contains all integrals needed
    # for particular address
    elif (os.path.exists("oei_partials.h5")):
        print("Found currently existing partial oei derivatives in working directory. Assuming they are correct.")
        oeifile = h5py.File('oei_partials.h5', 'r')
        with open(xyz_path, 'r') as f:
        nbf1 = basis1.nbf()
        nbf2 = basis2.nbf()
        # Check if there are `deriv_order` datasets in the eri file
        correct_deriv_order = len(oeifile) == deriv_order
        # Check nbf dimension of integral arrays
        sample_dataset_name = list(oeifile.keys())[0]
        correct_nbf1 = oeifile[sample_dataset_name].shape[0] == nbf1
        correct_nbf2 = oeifile[sample_dataset_name].shape[1] == nbf2
        oeifile.close()
        correct_int_derivs = correct_deriv_order and correct_nbf1 and correct_nbf2 """

def check_tei_disk(int_type, basis1, basis2, basis3, basis4, deriv_order, address=None):
    # Check TEI's in compute_integrals
    correct_int_derivs = False
    correct_nbf1 = correct_nbf2 = correct_nbf3 = correct_nbf4 = correct_deriv_order = False

    if ((os.path.exists(int_type + "_derivs.h5"))):
        print("Found currently existing " + int_type + " integral derivatives in your working directory. Trying to use them.")
        erifile = h5py.File(int_type + '_derivs.h5', 'r')
        nbf1 = basis1.nbf()
        nbf2 = basis2.nbf()
        nbf3 = basis3.nbf()
        nbf4 = basis4.nbf()

        tei_name = int_type + "_" + str(nbf1) + "_" + str(nbf2)\
                            + "_" + str(nbf3) + "_" + str(nbf4) + "_deriv" + str(deriv_order)
        
        # Check nbf dimension of integral arrays
        for name in list(erifile.keys()):
            if name in tei_name:
                correct_nbf1 = erifile[name].shape[0] == nbf1
                correct_nbf2 = erifile[name].shape[1] == nbf2
                correct_nbf3 = erifile[name].shape[2] == nbf3
                correct_nbf4 = erifile[name].shape[3] == nbf4
                correct_deriv_order = True
        erifile.close()
        correct_int_derivs = correct_deriv_order and correct_nbf1 and correct_nbf2 and correct_nbf3 and correct_nbf4
    
    if correct_int_derivs:
        print("Integral derivatives appear to be correct. Avoiding recomputation.")
    return correct_int_derivs

"""     # TODO flesh out this logic for determining if partials file contains all integrals needed
    # for particular address
    elif ((os.path.exists("eri_partials.h5"))):
        print("Found currently existing partial tei derivatives in working directory. Assuming they are correct.")
        erifile = h5py.File('eri_partials.h5', 'r')
        nbf1 = basis1.nbf()
        nbf2 = basis2.nbf()
        nbf3 = basis3.nbf()
        nbf4 = basis4.nbf()
        sample_dataset_name = list(erifile.keys())[0]
        correct_nbf1 = erifile[sample_dataset_name].shape[0] == nbf1
        correct_nbf2 = erifile[sample_dataset_name].shape[1] == nbf2
        correct_nbf3 = erifile[sample_dataset_name].shape[2] == nbf3
        correct_nbf4 = erifile[sample_dataset_name].shape[3] == nbf4
        erifile.close()
        correct_int_derivs = correct_deriv_order and correct_nbf1 and correct_nbf2 and correct_nbf3 and correct_nbf4
        if correct_int_derivs:
            print("Integral derivatives appear to be correct. Avoiding recomputation.")
        return correct_int_derivs
 """