import jax 
from jax import jacfwd
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import psi4
import numpy as np
import os

from .methods.basis_utils import build_RIBS
from .methods.hartree_fock import restricted_hartree_fock
from .methods.mp2 import restricted_mp2
from .methods.mp2f12 import restricted_mp2_f12
from .methods.ccsd import rccsd
from .methods.ccsd_t import rccsd_t
from .utils import n_frozen_core

psi4.core.be_quiet()

def check_options(options):
    """
    Checks user-supplied keyword options and assigns them

    Parameters
    ----------
    options : dict
        Dictionary of options controlling electronic structure code parameters

    Returns
    -------
    keyword_options : dict
        Dictionary of options controlling electronic structure code parameters
    """
    # Add all additional keywords to here
    keyword_options = {'maxit': 100,
                       'damping': False,
                       'damp_factor': 0.5,
                       'spectral_shift': True,
                       'integral_algo': 'libint_core',
                       'ints_tolerance': 1.0e-14,
                       'freeze_core': False,
                       'beta': 1.0,
                       'electric_field': False
                      }

    for key in options.keys():
        if key in keyword_options.keys(): 
            if type(options[key]) == type(keyword_options[key]):
                # Override default and assign, else print warning
                keyword_options[key] = options[key]
            else:
                print("Value '{}' for keyword option '{}' not recognized. Ignoring.".format(options[key],key))
        else:
            print("{} keyword option not recognized.".format(key))
    return keyword_options

def compute_standard(method, method_args, deriv_order=0, partial=None, options=None):
    """
    General function for computing energies, derivatives, and partial derivatives with respect to one input variable.
    """
    # Energy and full derivative tensor evaluations
    if not partial:
        # Create energy evaluation function
        if method == 'scf' or method == 'hf' or method == 'rhf':
            def electronic_energy(*args, options=options, deriv_order=deriv_order):
                return restricted_hartree_fock(*args, options=options, deriv_order=deriv_order)
        elif method =='mp2':
            def electronic_energy(*args, options=options, deriv_order=deriv_order):
                return restricted_mp2(*args, options=options, deriv_order=deriv_order)
        elif method =='mp2-f12':
            def electronic_energy(*args, options=options, deriv_order=deriv_order):
                return restricted_mp2_f12(*args, options=options, deriv_order=deriv_order)
        elif method =='ccsd':
            def electronic_energy(*args, options=options, deriv_order=deriv_order):
                return rccsd(*args, options=options, deriv_order=deriv_order)
        elif method =='ccsd(t)':
            def electronic_energy(*args, options=options, deriv_order=deriv_order):
                return rccsd_t(*args, options=options, deriv_order=deriv_order)
        else:
            raise Exception("Error: Method {} not supported.".format(method))

        # Evaluate energy or derivative 
        if deriv_order == 0:
            energy = electronic_energy(*method_args)
            return energy
        elif deriv_order == 1:
            grad = jacfwd(electronic_energy, 0)(*method_args)
            deriv = jnp.round(grad, 10)
        elif deriv_order == 2:
            hess = jacfwd(jacfwd(electronic_energy, 0))(*method_args)
            deriv = jnp.round(hess, 10)
        elif deriv_order == 3:
            cubic = jacfwd(jacfwd(jacfwd(electronic_energy, 0)))(*method_args)
            deriv = jnp.round(cubic, 10)
        elif deriv_order == 4:
            quartic = jacfwd(jacfwd(jacfwd(jacfwd(electronic_energy, 0))))(*method_args)
            deriv = jnp.round(quartic, 10)
        else:
            raise Exception("Error: Order {} derivatives are not exposed to the API.".format(deriv_order))
            deriv = 0
        return np.asarray(deriv)
    
    # Partial derivatives
    else:
        if len(partial) != deriv_order:
            raise Exception("The length of the index coordinates given by 'partial' argument should be the same as the order of differentiation")

        # For partial derivatives, need to unpack each geometric or electric field coordinate into separate arguments
        # to differentiate wrt specific coordinates using JAX AD utilities.
        param_list = method_args[0]

        #TODO support internal coordinate wrapper function.
        # This will take in internal coordinates, transform them into cartesians, and then compute integrals, energy
        # JAX will then collect the internal coordinate partial derivative instead. 
        if method == 'scf' or method == 'hf' or method == 'rhf':
            def partial_wrapper(*args):
                param = jnp.asarray(args)
                args = (param,) + method_args[1:]
                E_scf = restricted_hartree_fock(*args, options=options, deriv_order=deriv_order, return_aux_data=False)
                return E_scf
        elif method =='mp2':
            def partial_wrapper(*args):
                param = jnp.asarray(args)
                args = (param,) + method_args[1:]
                E_mp2 = restricted_mp2(*args, options=options, deriv_order=deriv_order)
                return E_mp2
        elif method =='mp2-f12':
            def partial_wrapper(*args):
                param = jnp.asarray(args)
                args = (param,) + method_args[1:]
                E_mp2f12 = restricted_mp2_f12(*args, options=options, deriv_order=deriv_order)
                return E_mp2f12
        elif method =='ccsd':
            def partial_wrapper(*args):
                param = jnp.asarray(args)
                args = (param,) + method_args[1:]
                E_ccsd = rccsd(*args, options=options, deriv_order=deriv_order)
                return E_ccsd
        elif method =='ccsd(t)':
            def partial_wrapper(*args):
                param = jnp.asarray(args)
                args = (param,) + method_args[1:]
                E_ccsd_t = rccsd_t(*args, options=options, deriv_order=deriv_order)
                return E_ccsd_t
        else:
            raise Exception("Error: Method {} not supported.".format(method))

        if deriv_order == 1:
            i = partial[0]
            partial_deriv = jacfwd(partial_wrapper, i)(*param_list)
        elif deriv_order == 2:
            i,j = partial[0], partial[1]
            partial_deriv = jacfwd(jacfwd(partial_wrapper, i), j)(*param_list)
        elif deriv_order == 3:
            i,j,k = partial[0], partial[1], partial[2]
            partial_deriv = jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k)(*param_list)
        elif deriv_order == 4:
            i,j,k,l = partial[0], partial[1], partial[2], partial[3]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l)(*param_list)
        elif deriv_order == 5:
            i,j,k,l,m = partial[0], partial[1], partial[2], partial[3], partial[4]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l), m)(*param_list)
        elif deriv_order == 6:
            i,j,k,l,m,n = partial[0], partial[1], partial[2], partial[3], partial[4], partial[5]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l), m), n)(*param_list)
        else:
            raise Exception("Error: Order {} partial derivatives are not exposed to the API.".format(deriv_order))
            partial_deriv = 0
        return jnp.round(partial_deriv, 14)
    
def compute_mixed(method, method_args, deriv_order_F=1, deriv_order_R=1, partial_F=None, partial_R=None, options=None):
    """
    General function for computing energies, derivatives, and partial derivatives with respect to two input variables.
    """
    # Number of differentiation calls depends on the total
    total_deriv_order = deriv_order_F + deriv_order_R
    
    # Energy and full derivative tensor evaluations
    if not partial_F or not partial_R:
        # Creates indices list to decide electric_field or coordinate differentiation
        FR_list = np.append(np.zeros(deriv_order_F, int), np.ones(deriv_order_R, int))

        # Create energy evaluation function
        if method == 'scf' or method == 'hf' or method == 'rhf':
            def electronic_energy(*args, options=options, deriv_order=deriv_order_R):
                return restricted_hartree_fock(*args, options=options, deriv_order=deriv_order)
        elif method =='mp2':
            def electronic_energy(*args, options=options, deriv_order=deriv_order_R):
                return restricted_mp2(*args, options=options, deriv_order=deriv_order)
        elif method =='mp2-f12':
            def electronic_energy(*args, options=options, deriv_order=deriv_order_R):
                return restricted_mp2_f12(*args, options=options, deriv_order=deriv_order)
        elif method =='ccsd':
            def electronic_energy(*args, options=options, deriv_order=deriv_order_R):
                return rccsd(*args, options=options, deriv_order=deriv_order)
        elif method =='ccsd(t)':
            def electronic_energy(*args, options=options, deriv_order=deriv_order_R):
                return rccsd_t(*args, options=options, deriv_order=deriv_order)
        else:
            print("Desired electronic structure method not understood. Use 'scf' 'hf' 'mp2' 'ccsd' or 'ccsd(t)' ")

        if total_deriv_order == 2:
            i,j = FR_list[0], FR_list[1]
            deriv = jacfwd(jacfwd(electronic_energy, i), j)(*method_args)
        elif total_deriv_order == 3:
            i,j,k = FR_list[0], FR_list[1], FR_list[2]
            deriv = jacfwd(jacfwd(jacfwd(electronic_energy, i), j), k)(*method_args)
        elif total_deriv_order == 4:
            i,j,k,l = FR_list[0], FR_list[1], FR_list[2], FR_list[3]
            deriv = jacfwd(jacfwd(jacfwd(jacfwd(electronic_energy, i), j), k), l)(*method_args)
        elif total_deriv_order == 5:
            i,j,k,l,m = FR_list[0], FR_list[1], FR_list[2], FR_list[3], FR_list[4]
            deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(electronic_energy, i), j), k), l), m)(*method_args)
        elif total_deriv_order == 6:
            i,j,k,l,m,n = FR_list[0], FR_list[1], FR_list[2], FR_list[3], FR_list[4], FR_list[5]
            deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(electronic_energy, i), j), k), l), m), n)(*method_args)
        else:
            print("Error: Order {},{} mixed derivatives are not exposed to the API.".format(deriv_order_F, deriv_order_R))
            deriv = 0
        return np.asarray(deriv)
    
    # Partial derivatives
    else:
        if len(partial_F) != deriv_order_F or len(partial_R) != deriv_order_R:
            raise Exception("The length of the index coordinates given by 'partial' argument should be the same as the order of differentiation")

        # For partial derivatives, need to unpack each geometric or electric field coordinate into separate arguments
        # to differentiate wrt specific coordinates using JAX AD utilities.
        param_list = (*method_args[0],) + (*method_args[1],)

        #TODO support internal coordinate wrapper function.
        # This will take in internal coordinates, transform them into cartesians, and then compute integrals, energy
        # JAX will then collect the internal coordinate partial derivative instead. 
        if method == 'scf' or method == 'hf' or method == 'rhf':
            def partial_wrapper(*args):
                param1 = jnp.asarray(args[0:3])
                param2 = jnp.asarray(args[3:])
                args = (param1, param2) + method_args[2:]
                E_scf = restricted_hartree_fock(*args, options=options, deriv_order=deriv_order_R, return_aux_data=False)
                return E_scf
        elif method =='mp2':
            def partial_wrapper(*args):
                param1 = jnp.asarray(args[0:3])
                param2 = jnp.asarray(args[3:])
                args = (param1, param2) + method_args[2:]
                E_mp2 = restricted_mp2(*args, options=options, deriv_order=deriv_order_R)
                return E_mp2
        elif method =='mp2-f12':
            def partial_wrapper(*args):
                param1 = jnp.asarray(args[0:3])
                param2 = jnp.asarray(args[3:])
                args = (param1, param2) + method_args[2:]
                E_mp2f12 = restricted_mp2_f12(*args, options=options, deriv_order=deriv_order_R)
                return E_mp2f12
        elif method =='ccsd':
            def partial_wrapper(*args):
                param1 = jnp.asarray(args[0:3])
                param2 = jnp.asarray(args[3:])
                args = (param1, param2) + method_args[2:]
                E_ccsd = rccsd(*args, options=options, deriv_order=deriv_order_R)
                return E_ccsd
        elif method =='ccsd(t)':
            def partial_wrapper(*args):
                param1 = jnp.asarray(args[0:3])
                param2 = jnp.asarray(args[3:])
                args = (param1, param2) + method_args[2:]
                E_ccsd_t = rccsd_t(*args, options=options, deriv_order=deriv_order_R)
                return E_ccsd_t
        else:
            raise Exception("Error: Method {} not supported.".format(method))
        
        # Combine partial tuples into one array
        partial = np.append(np.array(partial_F), np.array(partial_R) + 3)

        if total_deriv_order == 2:
            i,j = partial[0], partial[1]
            partial_deriv = jacfwd(jacfwd(partial_wrapper, i), j)(*param_list)
        elif total_deriv_order == 3:
            i,j,k = partial[0], partial[1], partial[2]
            partial_deriv = jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k)(*param_list)
        elif total_deriv_order == 4:
            i,j,k,l = partial[0], partial[1], partial[2], partial[3]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l)(*param_list)
        elif total_deriv_order == 5:
            i,j,k,l,m = partial[0], partial[1], partial[2], partial[3], partial[4]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l), m)(*param_list)
        elif total_deriv_order == 6:
            i,j,k,l,m,n = partial[0], partial[1], partial[2], partial[3], partial[4], partial[5]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l), m), n)(*param_list)
        elif total_deriv_order == 7:
            i,j,k,l,m,n,p = partial[0], partial[1], partial[2], partial[3], partial[4], partial[5], partial[6]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l), m), n), p)(*param_list)
        elif total_deriv_order == 8:
            i,j,k,l,m,n,p,q = partial[0], partial[1], partial[2], partial[3], partial[4], partial[5], partial[6], partial[7]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l), m), n), p), q)(*param_list)
        else:
            print("Error: Order {},{} mixed derivatives are not exposed to the API.".format(deriv_order_F, deriv_order_R))
            partial_deriv = 0
        return jnp.round(partial_deriv, 14)

def energy(molecule, basis_name, method, options=None):
    """
    """
    # Set keyword options
    if options:
        options = check_options(options)
    else:
        options = check_options({'integral_algo': 'libint_core'})
    print("Using integral method: {}".format(options['integral_algo']))
    print("Number of OMP Threads: {}".format(psi4.core.get_num_threads()))

    # Load molecule data
    geom2d = np.asarray(molecule.geometry())
    geom_list = geom2d.reshape(-1).tolist()
    geom = jnp.asarray(geom2d.flatten())
    xyz_file_name = "geom.xyz"
    molecule.save_xyz_file(xyz_file_name, True)
    xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name
    mult = molecule.multiplicity()
    charge = molecule.molecular_charge()
    nuclear_charges = jnp.asarray([molecule.charge(i) for i in range(geom2d.shape[0])])
    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    nfrzn = n_frozen_core(molecule, charge) if options['freeze_core'] else 0

    basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
    nbf = basis_set.nbf()
    print("Basis name: ", basis_set.name())
    print("Number of basis functions: ", nbf)

    if method == 'scf' or method == 'hf' or method == 'rhf':
        args = (geom, basis_set, nelectrons, nuclear_charges, xyz_path)
    elif method =='mp2':
        args = (geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    elif method =='mp2-f12':
        cabs_set = build_RIBS(molecule, basis_set, basis_name + '-cabs')
        args = (geom, basis_set, cabs_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    elif method =='ccsd':
        args = (geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    elif method =='ccsd(t)':
        args = (geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    else:
        print("Desired electronic structure method not understood. Use 'scf' 'hf' 'mp2' 'ccsd' or 'ccsd(t)' ")

    return compute_standard(method, args, deriv_order=0, partial=None, options=options)

def geom_deriv(molecule, basis_name, method, deriv_order=1, partial=None, options=None):
    """
    """
    # Set keyword options
    if options:
        options = check_options(options)
        if deriv_order == 0:
            options['integral_algo'] = 'libint_core'
    else:
        options = check_options({})
    print("Using integral method: {}".format(options['integral_algo']))
    print("Number of OMP Threads: {}".format(psi4.core.get_num_threads()))

    # Load molecule data
    geom2d = np.asarray(molecule.geometry())
    geom_list = geom2d.reshape(-1).tolist()
    geom = jnp.asarray(geom2d.flatten())
    xyz_file_name = "geom.xyz"
    molecule.save_xyz_file(xyz_file_name, True)
    xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name
    mult = molecule.multiplicity()
    charge = molecule.molecular_charge()
    nuclear_charges = jnp.asarray([molecule.charge(i) for i in range(geom2d.shape[0])])
    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    nfrzn = n_frozen_core(molecule, charge) if options['freeze_core'] else 0

    basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
    nbf = basis_set.nbf()
    print("Basis name: ", basis_set.name())
    print("Number of basis functions: ", nbf)

    if method == 'scf' or method == 'hf' or method == 'rhf':
        args = (geom, basis_set, nelectrons, nuclear_charges, xyz_path)
    elif method =='mp2':
        args = (geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    elif method =='mp2-f12':
        cabs_set = build_RIBS(molecule, basis_set, basis_name + '-cabs')
        args = (geom, basis_set, cabs_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    elif method =='ccsd':
        args = (geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    elif method =='ccsd(t)':
        args = (geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    else:
        print("Desired electronic structure method not understood. Use 'scf' 'hf' 'mp2' 'ccsd' or 'ccsd(t)' ")

    return compute_standard(method, args, deriv_order=deriv_order, partial=partial, options=options)

def efield_deriv(molecule, basis_name, method, electric_field=None, deriv_order=1, partial=None, options=None):
    """
    """
    if type(electric_field) == type(None):
        raise Exception("Electric field must be given for dipole computation.")
    
    try:
        options['electric_field']
    except:
        options['electric_field'] = True
    
    # Set keyword options
    if options:
        options = check_options(options)
        if deriv_order == 0:
            options['integral_algo'] = 'libint_core'

    print("Using integral method: {}".format(options['integral_algo']))
    print("Number of OMP Threads: {}".format(psi4.core.get_num_threads()))

    # Load molecule data
    geom2d = np.asarray(molecule.geometry())
    geom_list = geom2d.reshape(-1).tolist()
    geom = jnp.asarray(geom2d.flatten())
    xyz_file_name = "geom.xyz"
    molecule.save_xyz_file(xyz_file_name, True)
    xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name
    mult = molecule.multiplicity()
    charge = molecule.molecular_charge()
    nuclear_charges = jnp.asarray([molecule.charge(i) for i in range(geom2d.shape[0])])
    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    nfrzn = n_frozen_core(molecule, charge) if options['freeze_core'] else 0

    basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
    nbf = basis_set.nbf()
    print("Basis name: ", basis_set.name())
    print("Number of basis functions: ", nbf)

    if method == 'scf' or method == 'hf' or method == 'rhf':
        args = (electric_field, geom, basis_set, nelectrons, nuclear_charges, xyz_path)
    elif method =='mp2':
        args = (electric_field, geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    elif method =='mp2-f12':
        cabs_set = build_RIBS(molecule, basis_set, basis_name + '-cabs')
        args = (electric_field, geom, basis_set, cabs_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    elif method =='ccsd':
        args = (electric_field, geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    elif method =='ccsd(t)':
        args = (electric_field, geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    else:
        print("Desired electronic structure method not understood. Use 'scf' 'hf' 'mp2' 'ccsd' or 'ccsd(t)' ")

    return compute_standard(method, args, deriv_order=deriv_order, partial=partial, options=options)

def mixed_deriv(molecule, basis_name, method, electric_field=None,
                deriv_order_F=1, deriv_order_R=1, partial_F=None, partial_R=None, options=None):
    """
    """
    if deriv_order_F == 0 or deriv_order_R == 0:
        raise Exception("Error: Order of differentiation cannot equal zero. Use energy or geometry_deriv or electric_field instead.")

    if type(electric_field) == type(None):
        raise Exception("Electric field must be given for dipole computation.")
    
    try:
        options['electric_field']
    except:
        options['electric_field'] = True
    
    # Set keyword options
    if options:
        options = check_options(options)
        if deriv_order_F == 0 and deriv_order_R == 0:
            options['integral_algo'] = 'libint_core'

    print("Using integral method: {}".format(options['integral_algo']))
    print("Number of OMP Threads: {}".format(psi4.core.get_num_threads()))

    # Load molecule data
    geom2d = np.asarray(molecule.geometry())
    geom_list = geom2d.reshape(-1).tolist()
    geom = jnp.asarray(geom2d.flatten())
    xyz_file_name = "geom.xyz"
    molecule.save_xyz_file(xyz_file_name, True)
    xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name
    mult = molecule.multiplicity()
    charge = molecule.molecular_charge()
    nuclear_charges = jnp.asarray([molecule.charge(i) for i in range(geom2d.shape[0])])
    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    nfrzn = n_frozen_core(molecule, charge) if options['freeze_core'] else 0

    basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
    nbf = basis_set.nbf()
    print("Basis name: ", basis_set.name())
    print("Number of basis functions: ", nbf)

    if method == 'scf' or method == 'hf' or method == 'rhf':
        args = (electric_field, geom, basis_set, nelectrons, nuclear_charges, xyz_path)
    elif method =='mp2':
        args = (electric_field, geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    elif method =='mp2-f12':
        cabs_set = build_RIBS(molecule, basis_set, basis_name + '-cabs')
        args = (electric_field, geom, basis_set, cabs_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    elif method =='ccsd':
        args = (electric_field, geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    elif method =='ccsd(t)':
        args = (electric_field, geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
    else:
        print("Desired electronic structure method not understood. Use 'scf' 'hf' 'mp2' 'ccsd' or 'ccsd(t)' ")

    return compute_mixed(method, args, deriv_order_F=deriv_order_F, deriv_order_R=deriv_order_R, 
                         partial_F=partial_F, partial_R=partial_R, options=options)