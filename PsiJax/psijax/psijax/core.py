import jax 
from jax import jacfwd
from jax.config import config
config.update("jax_enable_x64", True)
config.enable_omnistaging()
import jax.numpy as jnp
import psi4
import numpy as np
import os

from .external_integrals import libint_initialize, libint_finalize

from .integrals.basis_utils import build_basis_set
from .methods.energy_utils import nuclear_repulsion, cholesky_orthogonalization
from .methods.hartree_fock import restricted_hartree_fock
from .methods.mp2 import restricted_mp2
from .methods.ccsd import rccsd
from .methods.ccsd_t import rccsd_t

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
    #basis_dict = build_basis_set(molecule, basis_name)
    # TODO when integrals are exported, switch to mints args, flatten geometry
    # also adjust arguments of energy, gradient, partial gradient functions
    #basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
    #mints = psi4.core.MintsHelper(basis_set)

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
    #TODO TODO TODO: support internal coordinate wrapper function.
    # This will take in internal coordinates, transform them into cartesians, and then compute integrals, energy
    # JAX will then collect the internal coordinate derivative tensor instead. 

    # Initialize libint here and precompute ERI derivatives, then finalize
    libint_initialize(xyz_path, basis_name, order)
    libint_finalize()

    if method == 'scf' or method == 'hf' or method == 'rhf':
        if order == 1:
            grad = jacfwd(restricted_hartree_fock, 0)(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False)
            #return jnp.round(grad, 10)
            deriv = jnp.round(grad, 10)
        elif order == 2:
            hess = jacfwd(jacfwd(restricted_hartree_fock, 0))(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False)
            #return jnp.round(hess.reshape(dim,dim), 10)
            deriv = jnp.round(hess.reshape(dim,dim), 10)
        elif order == 3:
            cubic = jacfwd(jacfwd(jacfwd(restricted_hartree_fock, 0)))(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False)
            #return jnp.round(cubic.reshape(dim,dim,dim), 10)
            deriv = jnp.round(cubic.reshape(dim,dim,dim), 10)
        elif order == 4:
            quartic = jacfwd(jacfwd(jacfwd(jacfwd(restricted_hartree_fock, 0))))(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False)
            #return jnp.round(quartic.reshape(dim,dim,dim,dim), 10)
            deriv = jnp.round(quartic.reshape(dim,dim,dim,dim), 10)

    elif method =='mp2':
        if order == 1:
            grad = jacfwd(restricted_mp2, 0)(geom, basis_name, xyz_path, nuclear_charges, charge)
            #return jnp.round(grad, 10)
            deriv = jnp.round(grad, 10)
        elif order == 2:
            hess = jacfwd(jacfwd(restricted_mp2, 0))(geom, basis_name, xyz_path, nuclear_charges, charge)
            #return jnp.round(hess.reshape(dim,dim), 10)
            deriv = jnp.round(hess.reshape(dim,dim), 10)
        elif order == 3:
            cubic = jacfwd(jacfwd(jacfwd(restricted_mp2, 0)))(geom, basis_name, xyz_path, nuclear_charges, charge)
            #return jnp.round(cubic.reshape(dim,dim,dim), 10)
            deriv = jnp.round(cubic.reshape(dim,dim,dim), 10)
        elif order == 4:
            quartic = jacfwd(jacfwd(jacfwd(jacfwd(restricted_mp2, 0))))(geom, basis_name, xyz_path, nuclear_charges, charge)
            #return jnp.round(quartic.reshape(dim,dim,dim,dim), 10)
            deriv =jnp.round(quartic.reshape(dim,dim,dim,dim), 10)

    elif method =='ccsd':
        if order == 1:
            grad = jacfwd(rccsd, 0)(geom, basis_name, xyz_path, nuclear_charges, charge)
            #return jnp.round(grad, 10)
            deriv = jnp.round(grad, 10)
        elif order == 2:
            hess = jacfwd(jacfwd(rccsd, 0))(geom, basis_name, xyz_path, nuclear_charges, charge)
            #return jnp.round(hess.reshape(dim,dim), 10)
            deriv = jnp.round(hess.reshape(dim,dim), 10)
        elif order == 3:
            cubic = jacfwd(jacfwd(jacfwd(rccsd, 0)))(geom, basis_name, xyz_path, nuclear_charges, charge)
            #return jnp.round(cubic.reshape(dim,dim,dim), 10)
            deriv = jnp.round(cubic.reshape(dim,dim,dim), 10)
        elif order == 4:
            quartic = jacfwd(jacfwd(jacfwd(jacfwd(rccsd, 0))))(geom, basis_name, xyz_path, nuclear_charges, charge)
            #return jnp.round(quartic.reshape(dim,dim,dim,dim), 10)
            deriv = jnp.round(quartic.reshape(dim,dim,dim,dim), 10)

    elif method =='ccsd(t)':
        if order == 1:
            grad = jacfwd(rccsd_t, 0)(geom, basis_name, xyz_path, nuclear_charges, charge)
            #return jnp.round(grad, 10)
            deriv = jnp.round(grad, 10)
        elif order == 2:
            hess = jacfwd(jacfwd(rccsd_t, 0))(geom, basis_name, xyz_path, nuclear_charges, charge)
            #return jnp.round(hess.reshape(dim,dim), 10)
            deriv = jnp.round(hess.reshape(dim,dim), 10)
        elif order == 3:
            cubic = jacfwd(jacfwd(jacfwd(rccsd_t, 0)))(geom, basis_name, xyz_path, nuclear_charges, charge)
            #return jnp.round(cubic.reshape(dim,dim,dim), 10)
            deriv = jnp.round(cubic.reshape(dim,dim,dim), 10)
        elif order == 4:
            quartic = jacfwd(jacfwd(jacfwd(jacfwd(rccsd_t, 0))))(geom, basis_name, xyz_path, nuclear_charges, charge)
            #return jnp.round(quartic.reshape(dim,dim,dim,dim), 10)
            deriv = jnp.round(quartic.reshape(dim,dim,dim,dim), 10)
    else:
        print("Desired electronic structure method not understood. Use 'scf' 'hf' 'mp2' 'ccsd' or 'ccsd(t)' ")

    if os.path.exists("eri_derivs.h5"):
        print("Deleting two electron integral derivatives...")
        os.remove("eri_derivs.h5")
    if os.path.exists("oei_derivs.h5"):
        print("Deleting one electron integral derivatives...")
        os.remove("oei_derivs.h5")
    return deriv 

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

    # Save xyz file, get path
    xyz_file_name = "geom.xyz"
    molecule.save_xyz_file(xyz_file_name, True)
    xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name

    #basis_dict = build_basis_set(molecule, basis_name
    kwargs = {"basis_name":basis_name,"xyz_path":xyz_path, "nuclear_charges":nuclear_charges, "charge":charge}

    #TODO TODO TODO: support internal coordinate wrapper function.
    # This will take in internal coordinates, transform them into cartesians, and then compute integrals, energy
    # JAX will then collect the internal coordinate partial derivative instead. 

    # Initialize libint here and precompute ERI derivatives, then finalize
    # TODO this is temporary, eventually would like to only write to disk the needed partial derivs.
    #libint_initialize(xyz_path, basis_name, order)
    #libint_finalize()

    # TODO use HDF5 to check the shape of the written integrals, if found.
    if ((os.path.exists("eri_derivs.h5") and os.path.exists("oei_derivs.h5"))):
        print("Found currently existing integrals in your working directory. Trying to use them.")
    if not ((os.path.exists("eri_derivs.h5") and os.path.exists("oei_derivs.h5"))):
        libint_initialize(xyz_path, basis_name, order)
        libint_finalize()

    if method == 'scf' or method == 'hf' or method == 'rhf':
        # Unpack the geometry as a list of single coordinates so we can differentiate w.r.t. single coords
        def scf_partial_wrapper(*args, **kwargs):
            geom = jnp.asarray(args)
            #basis_dict = kwargs['basis_dict']
            basis_name = kwargs['basis_name']
            xyz_path = kwargs['xyz_path']
            nuclear_charges = kwargs['nuclear_charges']
            charge = kwargs['charge']
            E_scf = restricted_hartree_fock(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False)
            return E_scf

        if order == 1:
            i = address[0]
            partial_deriv = jacfwd(scf_partial_wrapper, i)(*geom_list, **kwargs)
        elif order == 2:
            i,j = address[0], address[1]
            partial_deriv = jacfwd(jacfwd(scf_partial_wrapper, i), j)(*geom_list, **kwargs)
        elif order == 3:
            i,j,k = address[0], address[1], address[2]
            partial_deriv = jacfwd(jacfwd(jacfwd(scf_partial_wrapper, i), j), k)(*geom_list, **kwargs)
        elif order == 4:
            i,j,k,l = address[0], address[1], address[2], address[3]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(scf_partial_wrapper, i), j), k), l)(*geom_list, **kwargs)
        elif order == 5:
            i,j,k,l,m = address[0], address[1], address[2], address[3], address[4]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(scf_partial_wrapper, i), j), k), l), m)(*geom_list, **kwargs)
        elif order == 6:
            i,j,k,l,m,n = address[0], address[1], address[2], address[3], address[4], address[5]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(scf_partial_wrapper, i), j), k), l), m), n)(*geom_list, **kwargs)
        else:
            print("Error: Order {} partial derivatives are not implemented nor recommended for Hartree-Fock.".format(order))
        
    elif method =='mp2':
        # Unpack the geometry as a list of single coordinates so we can differentiate w.r.t. single coords
        def mp2_partial_wrapper(*args, **kwargs):
            geom = jnp.asarray(args)
            #basis_dict = kwargs['basis_dict']
            basis_name = kwargs['basis_name']
            xyz_path = kwargs['xyz_path']
            nuclear_charges = kwargs['nuclear_charges']
            charge = kwargs['charge']
            E_mp2 = restricted_mp2(geom, basis_name, xyz_path, nuclear_charges, charge)
            return E_mp2
        if order == 1:
            i = address[0]
            partial_deriv = jacfwd(mp2_partial_wrapper, i)(*geom_list, **kwargs)
        elif order == 2:
            i,j = address[0], address[1]
            partial_deriv = jacfwd(jacfwd(mp2_partial_wrapper, i), j)(*geom_list, **kwargs)
        elif order == 3:
            i,j,k = address[0], address[1], address[2]
            partial_deriv = jacfwd(jacfwd(jacfwd(mp2_partial_wrapper, i), j), k)(*geom_list, **kwargs)
        elif order == 4:
            i,j,k,l = address[0], address[1], address[2], address[3]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(mp2_partial_wrapper, i), j), k), l)(*geom_list, **kwargs)
        elif order == 5:
            i,j,k,l,m = address[0], address[1], address[2], address[3],address[4]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(mp2_partial_wrapper, i), j), k), l),m)(*geom_list, **kwargs)
        elif order == 6:
            i,j,k,l,m,n = address[0], address[1], address[2], address[3], address[4], address[5]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(mp2_partial_wrapper, i), j), k), l), m), n)(*geom_list, **kwargs)
        else:
            print("Error: Order {} partial derivatives are not implemented nor recommended for MP2.".format(order))

    elif method =='ccsd':
        def ccsd_partial_wrapper(*args, **kwargs):
            geom = jnp.asarray(args)
            #basis_dict = kwargs['basis_dict']
            basis_name = kwargs['basis_name']
            xyz_path = kwargs['xyz_path']
            nuclear_charges = kwargs['nuclear_charges']
            charge = kwargs['charge']
            E_ccsd = rccsd(geom, basis_name, xyz_path, nuclear_charges, charge)
            return E_ccsd
        if order == 1:
            i = address[0]
            partial_deriv = jacfwd(ccsd_partial_wrapper, i)(*geom_list, **kwargs)
        elif order == 2:
            i,j = address[0], address[1]
            partial_deriv = jacfwd(jacfwd(ccsd_partial_wrapper, i), j)(*geom_list, **kwargs)
        elif order == 3:
            i,j,k = address[0], address[1], address[2]
            partial_deriv = jacfwd(jacfwd(jacfwd(ccsd_partial_wrapper, i), j), k)(*geom_list, **kwargs)
        elif order == 4:
            i,j,k,l = address[0], address[1], address[2], address[3]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(ccsd_partial_wrapper, i), j), k), l)(*geom_list, **kwargs)
        elif order == 5:
            i,j,k,l,m = address[0], address[1], address[2], address[3], address[4]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(ccsd_partial_wrapper, i), j), k), l), m)(*geom_list, **kwargs)
        elif order == 6:
            i,j,k,l,m,n = address[0], address[1], address[2], address[3], address[4], address[5]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(ccsd_partial_wrapper, i), j), k), l), m), n)(*geom_list, **kwargs)
        else:
            print("Error: Order {} partial derivatives are not implemented nor recommended for CCSD.".format(order))

    elif method =='ccsd(t)':
        def ccsd_t_partial_wrapper(*args, **kwargs):
            geom = jnp.asarray(args)
            basis_name = kwargs['basis_name']
            xyz_path = kwargs['xyz_path']
            nuclear_charges = kwargs['nuclear_charges']
            charge = kwargs['charge']
            E_ccsd_t = rccsd_t(geom, basis_name, xyz_path, nuclear_charges, charge)
            return E_ccsd_t
        if order == 1:
            i = address[0]
            partial_deriv = jacfwd(ccsd_t_partial_wrapper, i)(*geom_list, **kwargs)
        elif order == 2:
            i,j = address[0], address[1]
            partial_deriv  = jacfwd(jacfwd(ccsd_t_partial_wrapper, i), j)(*geom_list, **kwargs)
        elif order == 3:
            i,j,k = address[0], address[1], address[2]
            partial_deriv = jacfwd(jacfwd(jacfwd(ccsd_t_partial_wrapper, i), j), k)(*geom_list, **kwargs)
        elif order == 4:
            i,j,k,l = address[0], address[1], address[2], address[3]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(ccsd_t_partial_wrapper, i), j), k), l)(*geom_list, **kwargs)
        elif order == 5:
            i,j,k,l,m = address[0], address[1], address[2], address[3], address[4]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(ccsd_t_partial_wrapper, i), j), k), l), m)(*geom_list, **kwargs)
        elif order == 6:
            i,j,k,l,m,n = address[0], address[1], address[2], address[3], address[4], address[5]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(ccsd_t_partial_wrapper, i), j), k), l), m), n)(*geom_list, **kwargs) 
        else:
            print("Error: Order {} partial derivatives are not implemented nor recommended for CCSD(T).".format(order))

    else:
        print("Error: Method {} not supported.".format(method))

    #if os.path.exists("eri_derivs.h5"):
    #    print("Deleting two electron integral derivatives...")
    #    os.remove("eri_derivs.h5")
    #if os.path.exists("oei_derivs.h5"):
    #    print("Deleting one electron integral derivatives...")
    #    os.remove("oei_derivs.h5")
    return partial_deriv


