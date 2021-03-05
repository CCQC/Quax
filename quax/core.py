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

from .integrals.basis_utils import build_basis_set
from .methods.energy_utils import nuclear_repulsion, cholesky_orthogonalization
from .methods.hartree_fock import restricted_hartree_fock
from .methods.mp2 import restricted_mp2
from .methods.ccsd import rccsd
from .methods.ccsd_t import rccsd_t

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

def compute(molecule, basis_name, method, options=None, deriv_order=0, partial=None):
    """
    General function for computing energies, derivatives, and partial derivatives.
    """
    # Set keyword options
    if options:
        options = check_options(options)
    else:
        options = check_options({})

    # Load molecule data
    geom2d = np.asarray(molecule.geometry())
    geom_list = geom2d.reshape(-1).tolist() 
    geom = jnp.asarray(geom2d.flatten())
    dim = geom.reshape(-1).shape[0]
    xyz_file_name = "geom.xyz"
    molecule.save_xyz_file(xyz_file_name, True)
    xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name
    mult = molecule.multiplicity()
    charge = molecule.molecular_charge()
    nuclear_charges = jnp.asarray([molecule.charge(i) for i in range(geom2d.shape[0])])
    args = (geom, basis_name, xyz_path, nuclear_charges, charge, options)

    basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
    nbf = basis_set.nbf()
    print("Number of basis functions: ", nbf)

    # Energy and full derivative tensor evaluations
    if not partial:
        # Create energy evaluation function
        if method == 'scf' or method == 'hf' or method == 'rhf':
            def electronic_energy(*args, deriv_order=deriv_order):
                return restricted_hartree_fock(*args, deriv_order=deriv_order)
        elif method =='mp2':
            def electronic_energy(*args, deriv_order=deriv_order):
                return restricted_mp2(*args, deriv_order=deriv_order)
        elif method =='ccsd':
            def electronic_energy(*args, deriv_order=deriv_order):
                return rccsd(*args, deriv_order=deriv_order)
        elif method =='ccsd(t)':
            def electronic_energy(*args, deriv_order=deriv_order):
                return rccsd_t(*args, deriv_order=deriv_order)
        else:
            print("Desired electronic structure method not understood. Use 'scf' 'hf' 'mp2' 'ccsd' or 'ccsd(t)' ")

        # Evaluate energy or derivative 
        if deriv_order == 0:
            energy = electronic_energy(*args)
            return energy
        elif deriv_order == 1:
            grad = jacfwd(electronic_energy, 0)(*args)
            deriv = jnp.round(grad, 10)
        elif deriv_order == 2:
            hess = jacfwd(jacfwd(electronic_energy, 0))(*args)
            deriv = jnp.round(hess.reshape(dim,dim), 10)
        elif deriv_order == 3:
            cubic = jacfwd(jacfwd(jacfwd(electronic_energy, 0)))(*args)
            deriv = jnp.round(cubic.reshape(dim,dim,dim), 10)
        elif deriv_order == 4:
            quartic = jacfwd(jacfwd(jacfwd(jacfwd(electronic_energy, 0))))(*args)
            deriv = jnp.round(quartic.reshape(dim,dim,dim,dim), 10)
        else:
            print("Error: Order {} derivatives are not exposed to the API.".format(deriv_order))
            deriv = 0
        return np.asarray(deriv)

    # Partial derivatives
    else:
        if len(partial) != deriv_order:
            raise Exception("The length of the index coordinates given by 'partial' argument should be the same as the order of differentiation")
        # For partial derivatives, need to unpack each geometric coordinate into separate arguments
        # to differentiate wrt specific coordinates using JAX AD utilities. 

        #TODO TODO TODO: support internal coordinate wrapper function.
        # This will take in internal coordinates, transform them into cartesians, and then compute integrals, energy
        # JAX will then collect the internal coordinate partial derivative instead. 
        if method == 'scf' or method == 'hf' or method == 'rhf':
            def partial_wrapper(*args):
                geom = jnp.asarray(args)
                E_scf = restricted_hartree_fock(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order, return_aux_data=False)
                return E_scf
        elif method =='mp2':
            def partial_wrapper(*args):
                geom = jnp.asarray(args)
                E_mp2 = restricted_mp2(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order)
                return E_mp2
        elif method =='ccsd':
            def partial_wrapper(*args):
                geom = jnp.asarray(args)
                E_ccsd = rccsd(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order)
                return E_ccsd
        elif method =='ccsd(t)':
            def partial_wrapper(*args):
                geom = jnp.asarray(args)
                E_ccsd_t = rccsd_t(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order)
                return E_ccsd_t
        else:
            raise Exception("Error: Method {} not supported.".format(method))

        if deriv_order == 1:
            i = partial[0]
            partial_deriv = jacfwd(partial_wrapper, i)(*geom_list)
        elif deriv_order == 2:
            i,j = partial[0], partial[1]
            partial_deriv = jacfwd(jacfwd(partial_wrapper, i), j)(*geom_list)
        elif deriv_order == 3:
            i,j,k = partial[0], partial[1], partial[2]
            partial_deriv = jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k)(*geom_list)
        elif deriv_order == 4:
            i,j,k,l = partial[0], partial[1], partial[2], partial[3]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l)(*geom_list)
        elif deriv_order == 5:
            i,j,k,l,m = partial[0], partial[1], partial[2], partial[3], partial[4]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l), m)(*geom_list)
        elif deriv_order == 6:
            i,j,k,l,m,n = partial[0], partial[1], partial[2], partial[3], partial[4], partial[5]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l), m), n)(*geom_list)
        else:
            print("Error: Order {} partial derivatives are not exposed to the API.".format(deriv_order))
            partial_deriv = 0
        return jnp.round(partial_deriv,10)

def energy(molecule, basis_name, method, options=None):
    """
    Call an energy method on a molecule and basis set.

    Parameters
    ----------
    molecule : psi4.Molecule
        A Psi4 Molecule object containing geometry, charge, multiplicity, and optionally units in a multiline string. 
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
        A string representing a Gaussian basis set available in Psi4's basis set library (also needs to be in Libint's basis set library if using Libint interface).

    method : str
        A string representing a quantum chemistry method supported in Quax
        method = 'scf', method = 'mp2', method = 'ccsd(t)'

    options : dict
        Dictionary of user-supplied keyword options.

    Returns
    -------
    The electronic energy in a.u. (Hartrees)
    """
    E = compute(molecule, basis_name, method, options)
    return E

def derivative(molecule, basis_name, method, deriv_order, options=None):
    """
    Compute the full Cartesian derivative tensor for a particular energy method, molecule, and basis set. 

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
        A string representing a Gaussian basis set available in Psi4's basis set library (also needs to be in Libint's basis set library if using Libint interface).

    method : str
        A string representing a quantum chemistry method supported in Quax
        method = 'scf', method = 'mp2', method = 'ccsd(t)'

    deriv_order : int
        The order of the derivative. order = 1 -> first derivative ; order = 2 --> second derivative ...

    options : dict
        Dictionary of user-supplied keyword options.

    Returns
    -------
    deriv : float
        The requested derivative tensor, elements have units of Hartree/bohr^(n)
    """
    deriv = compute(molecule, basis_name, method, options, deriv_order)
    return deriv

def partial_derivative(molecule, basis_name, method, deriv_order, partial, options=None):
    """
    Computes one particular nth-order partial derivative of the energy of an electronic structure method
    w.r.t. a set of cartesian coordinates. If you have N cartesian coordinates in your molecule, the nuclear derivative tensor
    is N x N x N ... however many orders of differentiation. This function computes one element of that tensor, depending
    on the address of the derivative you supply.
    If you have 9 cartesian coordinates x1,y1,z1,x2,y2,z2,x3,y3,z3 and you want the quartic derivative d^4E/dx1dy2(dz3)^2
    the partial derivative address in the quartic derivative tensor would be (0, 4, 8, 8).
    Note that this is the same derivative as, say, (4, 8, 0, 8), or any other permutation of that tuple.
    Also note this is dependent upon the order in which you supply the cartesian coordinates in the molecule object,
    because that will determine the indices of the coordinates.

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
        A string representing a Gaussian basis set available in Psi4's basis set library (also needs to be in Libint's basis set library if using Libint interface).

    method : str
        A string representing a quantum chemistry method supported in Quax e.g. 'scf', 'mp2' 'ccsd(t)'

    deriv_order : int
        The order of the derivative. order = 1 -> first derivative ; order = 2 --> second derivative ...

    partial : tuple of ints
       A tuple of indices at which the desired derivative appears in the derivative tensor. 
       Coordinates are indexed according to their location in the row-wise flattened Cartesian coordinate array:
       atom  x   y   z
        A    0   1   2
        B    3   4   5 
        C    6   7   8 
       E.g. The second derivative w.r.t the first atoms x-components would have partial=(0,0)
       The mixed partial derivative w.r.t. y-components on first and third atoms would be partial=(1,7)

    options : dict
        Dictionary of user-supplied keyword options.

    Returns
    -------
    partial_deriv : float
        The requested partial derivative of the energy in units of Hartree/bohr^(n)
    """
    partial_deriv = compute(molecule, basis_name, method, options, deriv_order, partial)
    return partial_deriv


#def write_integrals(molecule, basis_name, order, address=None):
#    """
#    Writes all required (TODO only for diagonal) partial of one and two electron derivatives to disk
#    using Quax integrals code.
#    
#    Temporary function to write all needed integrals to disk.
#    Goal: Benchmark partial derivatives with address = (2,2,...,2)
#    up to quartic. only need derivative vectors
#    [0,0,1,0,0,0,...]
#    [0,0,2,0,0,0,...]
#    [0,0,3,0,0,0,...]
#    [0,0,4,0,0,0,...]
#    """
#    geom = jnp.asarray(np.asarray(molecule.geometry()))
#    geom_list = np.asarray(molecule.geometry()).reshape(-1).tolist()
#    mult = molecule.multiplicity()
#    charge = molecule.molecular_charge()
#    nuclear_charges = jnp.asarray([molecule.charge(i) for i in range(geom.shape[0])])
#
#    basis_dict = build_basis_set(molecule,basis_name)
#    kwargs = {"basis_dict":basis_dict,"nuclear_charges":nuclear_charges}
#
#    def oei_wrapper(*args, **kwargs):
#        geom = jnp.asarray(args)
#        basis_dict = kwargs['basis_dict']
#        nuclear_charges = kwargs['nuclear_charges']
#        S, T, V = oei.oei_arrays(geom.reshape(-1,3),basis_dict,nuclear_charges)
#        return S, T, V
#
#    def tei_wrapper(*args, **kwargs):
#        geom = jnp.asarray(args)
#        basis_dict = kwargs['basis_dict']
#        G = tei.tei_array(geom.reshape(-1,3),basis_dict)
#        return G
#    
#    # TODO can these be passed list of lists to generate all partials at a given order?
#    if order == 1:
#        i = address[0]
#        dS, dT, dV = jacfwd(oei_wrapper, i)(*geom_list, **kwargs)
#        dG = jacfwd(tei_wrapper, i)(*geom_list, **kwargs)
#    elif order == 2:
#        i,j = address[0], address[1]
#        dS, dT, dV = jacfwd(jacfwd(oei_wrapper, i), j)(*geom_list, **kwargs)
#        dG = jacfwd(jacfwd(tei_wrapper, i), j)(*geom_list, **kwargs)
#    elif order == 3:
#        i,j,k = address[0], address[1], address[2]
#        dS, dT, dV = jacfwd(jacfwd(jacfwd(oei_wrapper, i), j), k)(*geom_list, **kwargs)
#        dG = jacfwd(jacfwd(jacfwd(tei_wrapper, i), j), k)(*geom_list, **kwargs)
#    elif order == 4:
#        i,j,k,l = address[0], address[1], address[2], address[3]
#        dS, dT, dV= jacfwd(jacfwd(jacfwd(jacfwd(oei_wrapper, i), j), k), l)(*geom_list, **kwargs)
#        dG = jacfwd(jacfwd(jacfwd(jacfwd(tei_wrapper, i), j), k), l)(*geom_list, **kwargs)
#    elif order == 5:
#        i,j,k,l,m = address[0], address[1], address[2], address[3], address[4]
#        dS, dT, dV= jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(oei_wrapper, i), j), k), l), m)(*geom_list, **kwargs)
#        dG = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(tei_wrapper, i), j), k), l), m)(*geom_list, **kwargs)
#    elif order == 6:
#        i,j,k,l,m,n = address[0], address[1], address[2], address[3], address[4], address[5]
#        dS, dT, dV= jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(oei_wrapper, i), j), k), l), m), n)(*geom_list, **kwargs)
#        dG = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(tei_wrapper, i), j), k), l), m), n)(*geom_list, **kwargs)
#    else:
#        print("Error: Order {} partial derivatives are not exposed to the API.".format(order))
#
#    # Convert address tuple to (NCART,) derivative vector
#    deriv_vec = [0] * len(geom_list)
#    for i in address:
#        deriv_vec[i] += 1
#    deriv_vec = np.asarray(deriv_vec)
#    # Get flattened upper triangle index for this derivative vector
#    flat_idx = get_deriv_vec_idx(deriv_vec)
#
#    # Write to HDF5
#    # Open the h5py file without deleting all contents ('a' instead of 'a')
#    # and create a dataset
#    # Write this set of partial derivatives of integrals to disk.
#    f = h5py.File("oei_partials.h5","a")
#    f.create_dataset("overlap_deriv"+str(order)+"_"+str(flat_idx), data=dS)
#    f.create_dataset("kinetic_deriv"+str(order)+"_"+str(flat_idx), data=dT)
#    f.create_dataset("potential_deriv"+str(order)+"_"+str(flat_idx), data=dV)
#    f.close()
#
#    f = h5py.File("eri_partials.h5","a")
#    f.create_dataset("eri_deriv"+str(order)+"_"+str(flat_idx), data=dG)
#    f.close()
#
#    return 0

    


