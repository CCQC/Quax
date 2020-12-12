import jax 
from jax import jacfwd
from jax.config import config
config.update("jax_enable_x64", True)
config.enable_omnistaging()
import jax.numpy as jnp
import psi4
import numpy as np
import os
import h5py

from .external_integrals import libint_initialize, libint_finalize
from .external_integrals.utils import get_deriv_vec_idx

from .integrals import oei
from .integrals import tei

from .integrals.basis_utils import build_basis_set
from .methods.energy_utils import nuclear_repulsion, cholesky_orthogonalization
from .methods.hartree_fock import restricted_hartree_fock
from .methods.mp2 import restricted_mp2
from .methods.ccsd import rccsd
from .methods.ccsd_t import rccsd_t

psi4.core.be_quiet()

def energy(molecule, basis_name, method='scf'):
    """
    Call an energy method on a molecule and basis set.

    Parameters
    ----------
    molecule : psi4.Molecule
        A Psi4 Molecule object containing geometry, charge, multiplicity in a multiline string. 
        Examples:
        molecule = psi4.geometry('''
                                 0 1
                                 H 0.0 0.0 -0.55000000000
                                 H 0.0 0.0  0.55000000000
                                 units bohr
                                 ''')

        molecule = psi4.geometry('''
                                 0 1
                                 O
                                 H 1 r1
                                 H 1 r2 2 a1
                        
                                 r1 = 1.0
                                 r2 = 1.0
                                 a1 = 104.5
                                 units ang
                                 ''')

    basis_name : str
        A string representing a Gaussian basis set available in Psi4's basis set library.

    method : str
        A string representing a quantum chemistry method supported in PsiJax
        method = 'scf', method = 'mp2', method = 'ccd'

    Returns
    -------
    The electronic energy in a.u.
    """
    geom2d = np.asarray(molecule.geometry())
    geom = jnp.asarray(geom2d.flatten())
    xyz_file_name = "geom.xyz"
    # Save xyz file, get path
    molecule.save_xyz_file(xyz_file_name, True)
    xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name
    mult = molecule.multiplicity()
    charge = molecule.molecular_charge()
    nuclear_charges = jnp.asarray([molecule.charge(i) for i in range(geom2d.shape[0])])

    if method == 'scf' or method == 'hf' or method == 'rhf':
        E_scf = restricted_hartree_fock(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False)  
        return E_scf

    if method == 'mp2':
        E_mp2 = restricted_mp2(geom, basis_name, xyz_path, nuclear_charges, charge)
        return E_mp2

    if method == 'ccsd':
        E_ccsd = rccsd(geom, basis_name, xyz_path, nuclear_charges, charge) 
        return E_ccsd

    if method == 'ccsd(t)':
        E_ccsd_t = rccsd_t(geom, basis_name, xyz_path, nuclear_charges, charge) 
        return E_ccsd_t

def derivative(molecule, basis_name, method, order=1):
    """
    Convenience function for computing the full nuclear derivative tensor at some order
    for a particular energy method, molecule, and basis set.
    May be memory-intensive.
    For gradients, choose order=1, hessian order=2, cubic derivative tensor order=3, quartic order = 4.
    Anything higher order derivatives should use the partial derivative utility.
    """
    #geom = jnp.asarray(np.asarray(molecule.geometry()))
    geom2d = np.asarray(molecule.geometry())
    geom = jnp.asarray(geom2d.flatten())
    mult = molecule.multiplicity()
    charge = molecule.molecular_charge()
    nuclear_charges = jnp.asarray([molecule.charge(i) for i in range(geom2d.shape[0])])
    xyz_file_name = "geom.xyz"
    # Save xyz file, get path
    molecule.save_xyz_file(xyz_file_name, True)
    xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name
    #basis_dict = build_basis_set(molecule, basis_name)
    dim = geom.reshape(-1).shape[0]

    # Get number of basis functions
    basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
    nbf = basis_set.nbf()

    #TODO TODO TODO: support internal coordinate wrapper function.
    # This will take in internal coordinates, transform them into cartesians, and then compute integrals, energy
    # JAX will then collect the internal coordinate derivative tensor instead. 

    # TODO Can make this safer by including info HDF5 file with rounded geometry, atom labels, etc.
    if ((os.path.exists("eri_derivs.h5") and os.path.exists("oei_derivs.h5"))):
        print("Found currently existing integral derivatives in your working directory. Trying to use them.")
        oeifile = h5py.File('oei_derivs.h5', 'r')
        erifile = h5py.File('eri_derivs.h5', 'r')
        # Check if there are `deriv_order` datatsets in the eri file
        correct_deriv_order = len(erifile) == order
        # Check nbf dimension of integral arrays
        sample_dataset_name = list(oeifile.keys())[0]
        correct_nbf = oeifile[sample_dataset_name].shape[0] == nbf
        oeifile.close()
        erifile.close()
        if correct_deriv_order and correct_nbf:
            print("Integral derivatives appear to be correct. Avoiding recomputation.")
        else:
            print("Integral derivatives dimensions do not match requested derivative order and/or basis set. Recomputing integral derivatives")
            if os.path.exists("eri_derivs.h5"):
                print("Deleting two electron integral derivatives...")
                os.remove("eri_derivs.h5")
            if os.path.exists("oei_derivs.h5"):
                print("Deleting one electron integral derivatives...")
                os.remove("oei_derivs.h5")
            libint_initialize(xyz_path, basis_name, order)
            libint_finalize()
    else:
        libint_initialize(xyz_path, basis_name, order)
        libint_finalize()

    # Define function 'energy' depending on requested method
    if method == 'scf' or method == 'hf' or method == 'rhf':
        def electronic_energy(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False):
            return restricted_hartree_fock(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False)
    elif method =='mp2':
        def electronic_energy(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False):
            return restricted_mp2(geom, basis_name, xyz_path, nuclear_charges, charge)
    elif method =='ccsd':
        def electronic_energy(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False):
            return rccsd(geom, basis_name, xyz_path, nuclear_charges, charge)
    elif method =='ccsd(t)':
        def electronic_energy(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False):
            return rccsd_t(geom, basis_name, xyz_path, nuclear_charges, charge)
    else:
        print("Desired electronic structure method not understood. Use 'scf' 'hf' 'mp2' 'ccsd' or 'ccsd(t)' ")

    # Now compile and compute differentiated energy function
    if order == 1:
        grad = jacfwd(electronic_energy, 0)(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False)
        deriv = jnp.round(grad, 10)
    elif order == 2:
        hess = jacfwd(jacfwd(electronic_energy, 0))(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False)
        deriv = jnp.round(hess.reshape(dim,dim), 10)
    elif order == 3:
        cubic = jacfwd(jacfwd(jacfwd(electronic_energy, 0)))(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False)
        deriv = jnp.round(cubic.reshape(dim,dim,dim), 10)
    elif order == 4:
        quartic = jacfwd(jacfwd(jacfwd(jacfwd(electronic_energy, 0))))(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False)
        deriv = jnp.round(quartic.reshape(dim,dim,dim,dim), 10)

    return np.asarray(deriv)

def partial_derivative(molecule, basis_name, method, order, address):
    """
    Computes one particular nth-order partial derivative of the energy of an electronic structure method
    w.r.t. a set of cartesian coordinates. If you have N cartesian coordinates in your molecule, the nuclear derivative tensor
    is N x N x N ... however many orders of differentiation. This function computes one element of that tensor, depending
    on the address of the derivative you supply.
    If you have 9 cartesian coordinates x1,y1,z1,x2,y2,z2,x3,y3,z3 and you want the quartic derivative d^4E/dx1dy2(dz3)^2
    the 'address' of this derivative in the quartic derivative tensor would be (0, 4, 8, 8).
    Note that this is the same derivative as, say, (4, 8, 0, 8), or any other permutation of that tuple.
    Also note this is dependent upon the order in which you supply the cartesian coordinates in the molecule object,
    because that will determine the indices of the coordinates.

    Parameters
    ----------
    Call an energy method on a molecule and basis set.

    Parameters
    ----------
    molecule : psi4.Molecule
        A Psi4 Molecule object containing geometry, charge, multiplicity in a multiline string. 
        Examples:
        molecule = psi4.geometry('''
                                 0 1
                                 H 0.0 0.0 -0.55000000000
                                 H 0.0 0.0  0.55000000000
                                 units bohr
                                 ''')

        molecule = psi4.geometry('''
                                 0 1
                                 O
                                 H 1 r1
                                 H 1 r2 2 a1
                        
                                 r1 = 1.0
                                 r2 = 1.0
                                 a1 = 104.5
                                 units ang
                                 ''')

    basis_name : str
        A string representing a Gaussian basis set available in Psi4's basis set library.
    method : str
        A string representing a quantum chemistry method supported in PsiJax
        method = 'scf', method = 'mp2', method = 'ccd'
    order : int
        The order of the derivative. order = 1 -> gradient ; order = 2 --> hessian ; order = 3 --> cubic ...
    address : tuple
       The index at which the desired derivative appears in the derivative tensor.

    Returns
    -------
    partial_deriv : float
        The requested partial derivative of the energy for the given geometry and basis set.
    """
    if len(address) != order:
        raise Exception("The length of the index coordinates given by 'address' arguments should be the same as the order of differentiation")
    geom = jnp.asarray(np.asarray(molecule.geometry()))
    geom_list = np.asarray(molecule.geometry()).reshape(-1).tolist()
    mult = molecule.multiplicity()
    charge = molecule.molecular_charge()
    nuclear_charges = jnp.asarray([molecule.charge(i) for i in range(geom.shape[0])])

    # Get number of basis functions
    basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
    nbf = basis_set.nbf()

    # Save xyz file, get path
    xyz_file_name = "geom.xyz"
    molecule.save_xyz_file(xyz_file_name, True)
    xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name

    #basis_dict = build_basis_set(molecule, basis_name
    kwargs = {"basis_name":basis_name,"xyz_path":xyz_path, "nuclear_charges":nuclear_charges, "charge":charge}

    #TODO TODO TODO: support internal coordinate wrapper function.
    # This will take in internal coordinates, transform them into cartesians, and then compute integrals, energy
    # JAX will then collect the internal coordinate partial derivative instead. 

    # If integrals already exist in the working directory and they are correct shape, reuse them.
    # TODO Can make this safer by including info HDF5 file with rounded geometry, atom labels, etc.
    if ((os.path.exists("eri_derivs.h5") and os.path.exists("oei_derivs.h5"))):
        print("Found currently existing integral derivatives in your working directory. Trying to use them.")
        oeifile = h5py.File('oei_derivs.h5', 'r')
        erifile = h5py.File('eri_derivs.h5', 'r')
        # Check if there are `deriv_order` datatsets in the eri file
        correct_deriv_order = len(erifile) == order
        # Check nbf dimension of integral arrays
        sample_dataset_name = list(oeifile.keys())[0]
        correct_nbf = oeifile[sample_dataset_name].shape[0] == nbf
        oeifile.close()
        erifile.close()
        if correct_deriv_order and correct_nbf:
            print("Integral derivatives appear to be correct. Avoiding recomputation.")
        else:
            print("Integral derivatives dimensions do not match requested derivative order and/or basis set. Recomputing integral derivatives")
            if os.path.exists("eri_derivs.h5"):
                print("Deleting two electron integral derivatives...")
                os.remove("eri_derivs.h5")
            if os.path.exists("oei_derivs.h5"):
                print("Deleting one electron integral derivatives...")
                os.remove("oei_derivs.h5")
            libint_initialize(xyz_path, basis_name, order)
            libint_finalize()
    elif ((os.path.exists("eri_partials.h5") and os.path.exists("oei_partials.h5"))):
        print("Found currently existing partial derivatives in working directory. I hope you know what you are doing!")
    else:
        libint_initialize(xyz_path, basis_name, order)
        libint_finalize()

    # Wrap energy functions with unpacked geometric coordinates as single arguments, so we can differentiate w.r.t. single coords
    if method == 'scf' or method == 'hf' or method == 'rhf':
        def partial_wrapper(*args, **kwargs):
            geom = jnp.asarray(args)
            basis_name = kwargs['basis_name']
            xyz_path = kwargs['xyz_path']
            nuclear_charges = kwargs['nuclear_charges']
            charge = kwargs['charge']
            E_scf = restricted_hartree_fock(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False)
            return E_scf
    elif method =='mp2':
        def partial_wrapper(*args, **kwargs):
            geom = jnp.asarray(args)
            basis_name = kwargs['basis_name']
            xyz_path = kwargs['xyz_path']
            nuclear_charges = kwargs['nuclear_charges']
            charge = kwargs['charge']
            E_mp2 = restricted_mp2(geom, basis_name, xyz_path, nuclear_charges, charge)
            return E_mp2
    elif method =='ccsd':
        def partial_wrapper(*args, **kwargs):
            geom = jnp.asarray(args)
            basis_name = kwargs['basis_name']
            xyz_path = kwargs['xyz_path']
            nuclear_charges = kwargs['nuclear_charges']
            charge = kwargs['charge']
            E_ccsd = rccsd(geom, basis_name, xyz_path, nuclear_charges, charge)
            return E_ccsd
    elif method =='ccsd(t)':
        def partial_wrapper(*args, **kwargs):
            geom = jnp.asarray(args)
            basis_name = kwargs['basis_name']
            xyz_path = kwargs['xyz_path']
            nuclear_charges = kwargs['nuclear_charges']
            charge = kwargs['charge']
            E_ccsd_t = rccsd_t(geom, basis_name, xyz_path, nuclear_charges, charge)
            return E_ccsd_t
    else:
        raise Exception("Error: Method {} not supported.".format(method))

    if order == 1:
        i = address[0]
        partial_deriv = jacfwd(partial_wrapper, i)(*geom_list, **kwargs)
    elif order == 2:
        i,j = address[0], address[1]
        partial_deriv = jacfwd(jacfwd(partial_wrapper, i), j)(*geom_list, **kwargs)
    elif order == 3:
        i,j,k = address[0], address[1], address[2]
        partial_deriv = jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k)(*geom_list, **kwargs)
    elif order == 4:
        i,j,k,l = address[0], address[1], address[2], address[3]
        partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l)(*geom_list, **kwargs)
    elif order == 5:
        i,j,k,l,m = address[0], address[1], address[2], address[3], address[4]
        partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l), m)(*geom_list, **kwargs)
    elif order == 6:
        i,j,k,l,m,n = address[0], address[1], address[2], address[3], address[4], address[5]
        partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l), m), n)(*geom_list, **kwargs)
    else:
        print("Error: Order {} partial derivatives are not exposed to the API.".format(order))

    return partial_deriv

def write_integrals(molecule, basis_name, order, address):
    """
    Writes all required (TODO only for diagonal) partial of one and two electron derivatives to disk
    using PsiJax integrals.
    
    Temporary function to write all needed integrals to disk.
    Goal: Benchmark partial derivatives with address = (2,2,...,2)
    up to quartic. only need derivative vectors
    [0,0,1,0,0,0,...]
    [0,0,2,0,0,0,...]
    [0,0,3,0,0,0,...]
    [0,0,4,0,0,0,...]
    Eventually maybe do sextic
    """

    geom = jnp.asarray(np.asarray(molecule.geometry()))
    geom_list = np.asarray(molecule.geometry()).reshape(-1).tolist()
    mult = molecule.multiplicity()
    charge = molecule.molecular_charge()
    nuclear_charges = jnp.asarray([molecule.charge(i) for i in range(geom.shape[0])])

    basis_dict = build_basis_set(molecule,basis_name)
    kwargs = {"basis_dict":basis_dict,"nuclear_charges":nuclear_charges}

    def oei_wrapper(*args, **kwargs):
        geom = jnp.asarray(args)
        basis_dict = kwargs['basis_dict']
        nuclear_charges = kwargs['nuclear_charges']
        S, T, V = oei.oei_arrays(geom.reshape(-1,3),basis_dict,nuclear_charges)
        return S, T, V

    def tei_wrapper(*args, **kwargs):
        geom = jnp.asarray(args)
        basis_dict = kwargs['basis_dict']
        nuclear_charges = kwargs['nuclear_charges']
        G = tei.tei_array(geom.reshape(-1,3),basis_dict)
        return G
    
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
    else:
        print("Error: Order {} partial derivatives are not exposed to the API.".format(order))

    # Convert address tuple to (NCART,) derivative vector
    deriv_vec = [0] * len(geom_list)
    for i in address:
        deriv_vec[i] += 1
    deriv_vec = np.asarray(deriv_vec)
    # Get flattened upper triangle index for this derivative vector
    flat_idx = get_deriv_vec_idx(deriv_vec)

    # Write to HDF5
    # Open the h5py file without deleting all contents ('a' instead of 'a')
    # and create a dataset
    # Write this set of partial derivatives of integrals to disk.
    f = h5py.File("oei_partials.h5","a")
    f.create_dataset("overlap_deriv"+str(order)+"_"+str(flat_idx), data=dS)
    f.create_dataset("kinetic_deriv"+str(order)+"_"+str(flat_idx), data=dT)
    f.create_dataset("potential_deriv"+str(order)+"_"+str(flat_idx), data=dV)
    f.close()

    f = h5py.File("eri_partials.h5","a")
    f.create_dataset("eri_deriv"+str(order)+"_"+str(flat_idx), data=dG)
    f.close()

    return 0

    


