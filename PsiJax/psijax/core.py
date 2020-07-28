import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
import psi4
import numpy as onp
from jax import jacfwd
from tei import tei_array 
from oei import oei_arrays
from basis_utils import build_basis_set
from energy_utils import nuclear_repulsion, symmetric_orthogonalization, cholesky_orthogonalization

from hartree_fock import restricted_hartree_fock
from mp2 import restricted_mp2, restricted_mp2_lowmem
from ccsd import rccsd

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
    geom = np.asarray(onp.asarray(molecule.geometry()))
    mult = molecule.multiplicity()
    charge = molecule.molecular_charge()
    nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])
    basis_dict = build_basis_set(molecule, basis_name)

    if method == 'scf' or method == 'hf' or method == 'rhf':
        E_scf = restricted_hartree_fock(geom, basis_dict, nuclear_charges, charge, return_aux_data=False)  
        return E_scf

    if method == 'mp2':
        E_mp2 = restricted_mp2(geom, basis_dict, nuclear_charges, charge)
        return E_mp2

    if method == 'ccsd':
        E_ccsd = rccsd(geom, basis_dict, nuclear_charges, charge) 
        return E_ccsd

def derivative(molecule, basis_name, method, order=1):
    """
    Convenience function for computing the full nuclear derivative tensor at some order
    for a particular energy method, molecule, and basis set.
    May be memory-intensive.
    For gradients, choose order=1, hessian order=2, cubic derivative tensor order=3, quartic order = 4.
    Anything higher order derivatives should use the partial derivative utility.
    """
    geom = np.asarray(onp.asarray(molecule.geometry()))
    mult = molecule.multiplicity()
    charge = molecule.molecular_charge()
    nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])
    basis_dict = build_basis_set(molecule, basis_name)
    dim = geom.reshape(-1).shape[0]
    if method == 'scf' or method == 'hf' or method == 'rhf':
        if order == 1:
            grad = jacfwd(restricted_hartree_fock, 0)(geom, basis_dict, nuclear_charges, charge, return_aux_data=False)
            return np.round(grad, 10)
        if order == 2:
            hess = jacfwd(jacfwd(restricted_hartree_fock, 0))(geom, basis_dict, nuclear_charges, charge, return_aux_data=False)
            return np.round(hess.reshape(dim,dim), 10)
        if order == 3:
            cubic = jacfwd(jacfwd(jacfwd(restricted_hartree_fock, 0)))(geom, basis_dict, nuclear_charges, charge, return_aux_data=False)
            return np.round(cubic.reshape(dim,dim,dim), 10)
        if order == 4:
            quartic = jacfwd(jacfwd(jacfwd(jacfwd(restricted_hartree_fock, 0))))(geom, basis_dict, nuclear_charges, charge, return_aux_data=False)
            return np.round(quartic.reshape(dim,dim,dim,dim), 10)

    if method =='mp2':
        if order == 1:
            grad = jacfwd(restricted_mp2, 0)(geom, basis_dict, nuclear_charges, charge)
            return np.round(grad, 10)
        if order == 2:
            hess = jacfwd(jacfwd(restricted_mp2, 0))(geom, basis_dict, nuclear_charges, charge)
            return np.round(hess.reshape(dim,dim), 10)
        if order == 3:
            cubic = jacfwd(jacfwd(jacfwd(restricted_mp2, 0)))(geom, basis_dict, nuclear_charges, charge, return_aux_data=False)
            return np.round(cubic.reshape(dim,dim,dim), 10)
        if order == 4:
            quartic = jacfwd(jacfwd(jacfwd(jacfwd(restricted_mp2, 0))))(geom, basis_dict, nuclear_charges, charge, return_aux_data=False)
            return np.round(quartic.reshape(dim,dim,dim,dim), 10)

    if method =='ccsd':
        if order == 1:
            grad = jacfwd(rccsd, 0)(geom, basis_dict, nuclear_charges, charge)
            return np.round(grad, 10)
        if order == 2:
            hess = jacfwd(jacfwd(rccsd, 0))(geom, basis_dict, nuclear_charges, charge)
            return np.round(hess.reshape(dim,dim), 10)
        if order == 3:
            cubic = jacfwd(jacfwd(jacfwd(rccsd, 0)))(geom, basis_dict, nuclear_charges, charge, return_aux_data=False)
            return np.round(cubic.reshape(dim,dim,dim), 10)
        if order == 4:
            quartic = jacfwd(jacfwd(jacfwd(jacfwd(rccsd, 0))))(geom, basis_dict, nuclear_charges, charge, return_aux_data=False)
            return np.round(quartic.reshape(dim,dim,dim,dim), 10)
    return 0


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
    geom = np.asarray(onp.asarray(molecule.geometry()))
    geom_list = onp.asarray(molecule.geometry()).reshape(-1).tolist()
    mult = molecule.multiplicity()
    charge = molecule.molecular_charge()
    nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])
    basis_dict = build_basis_set(molecule, basis_name)

    if method == 'scf' or method == 'hf' or method == 'rhf':
        # Unpack the geometry as a list of single coordinates so we can differentiate w.r.t. single coords
        def scf_partial_wrapper(*args, **kwargs):
            geom = np.asarray(args).reshape(-1,3)
            basis_dict = kwargs['basis_dict']
            nuclear_charges = kwargs['nuclear_charges']
            charge = kwargs['charge']
            E_scf = restricted_hartree_fock(geom, basis_dict, nuclear_charges, charge, return_aux_data=False)
            return E_scf

        if order == 1:
            i = address[0]
            partial_grad = jacfwd(scf_partial_wrapper, i)(*geom_list, basis_dict=basis_dict, nuclear_charges=nuclear_charges, charge=charge)
            return partial_grad
        if order == 2:
            i,j = address[0], address[1]
            partial_hess = jacfwd(jacfwd(scf_partial_wrapper, i), j)(*geom_list, basis_dict=basis_dict, nuclear_charges=nuclear_charges, charge=charge)
            return partial_hess
        if order == 3:
            i,j,k = address[0], address[1], address[2]
            partial_cubic = jacfwd(jacfwd(jacfwd(scf_partial_wrapper, i), j), k)(*geom_list, basis_dict=basis_dict, nuclear_charges=nuclear_charges, charge=charge)
            return partial_cubic
        if order == 4:
            i,j,k,l = address[0], address[1], address[2], address[3]
            partial_quartic = jacfwd(jacfwd(jacfwd(jacfwd(scf_partial_wrapper, i), j), k), l)(*geom_list, basis_dict=basis_dict, nuclear_charges=nuclear_charges, charge=charge)
            return partial_quartic
        if order == 5:
            i,j,k,l,m = address[0], address[1], address[2], address[3], address[4]
            partial_quintic = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(scf_partial_wrapper, i), j), k), l), m)(*geom_list, basis_dict=basis_dict, nuclear_charges=nuclear_charges, charge=charge)
            return partial_quintic
        if order == 6:
            i,j,k,l,m,n = address[0], address[1], address[2], address[3], address[4], address[5]
            partial_sextic = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(scf_partial_wrapper, i), j), k), l), m), n)(*geom_list, basis_dict=basis_dict, nuclear_charges=nuclear_charges, charge=charge)
            return partial_sextic
        else:
            return 0

    if method =='mp2':
        # Unpack the geometry as a list of single coordinates so we can differentiate w.r.t. single coords
        def mp2_partial_wrapper(*args, **kwargs):
            geom = np.asarray(args).reshape(-1,3)
            basis_dict = kwargs['basis_dict']
            nuclear_charges = kwargs['nuclear_charges']
            charge = kwargs['charge']
            E_mp2 = restricted_mp2(geom, basis_dict, nuclear_charges, charge)
            return E_mp2
        if order == 1:
            i = address[0]
            partial_grad = jacfwd(mp2_partial_wrapper, i)(*geom_list, basis_dict=basis_dict, nuclear_charges=nuclear_charges, charge=charge)
            return partial_grad
        if order == 2:
            i,j = address[0], address[1]
            partial_hess = jacfwd(jacfwd(mp2_partial_wrapper, i), j)(*geom_list, basis_dict=basis_dict, nuclear_charges=nuclear_charges, charge=charge)
            return partial_hess
        if order == 3:
            i,j,k = address[0], address[1], address[2]
            partial_cubic = jacfwd(jacfwd(jacfwd(mp2_partial_wrapper, i), j), k)(*geom_list, basis_dict=basis_dict, nuclear_charges=nuclear_charges, charge=charge)
            return partial_cubic
        if order == 4:
            i,j,k,l = address[0], address[1], address[2], address[3]
            partial_quartic = jacfwd(jacfwd(jacfwd(jacfwd(mp2_partial_wrapper, i), j), k), l)(*geom_list, basis_dict=basis_dict, nuclear_charges=nuclear_charges, charge=charge)
            return partial_quartic
        else:
            print("Error: {}'th order derivative tensor is not implemented nor recommended for MP2.".format(order))

    if method =='ccsd':
        def ccsd_partial_wrapper(*args, **kwargs):
            geom = np.asarray(args).reshape(-1,3)
            basis_dict = kwargs['basis_dict']
            nuclear_charges = kwargs['nuclear_charges']
            charge = kwargs['charge']
            E_ccsd = rccsd(geom, basis_dict, nuclear_charges, charge)
            return E_ccsd
        if order == 1:
            i = address[0]
            partial_grad = jacfwd(ccsd_partial_wrapper, i)(*geom_list, basis_dict=basis_dict, nuclear_charges=nuclear_charges, charge=charge)
            return partial_grad
        if order == 2:
            i,j = address[0], address[1]
            partial_hess = jacfwd(jacfwd(ccsd_partial_wrapper, i), j)(*geom_list, basis_dict=basis_dict, nuclear_charges=nuclear_charges, charge=charge)
            return partial_hess
        if order == 3:
            i,j,k = address[0], address[1], address[2]
            partial_cubic = jacfwd(jacfwd(jacfwd(ccsd_partial_wrapper, i), j), k)(*geom_list, basis_dict=basis_dict, nuclear_charges=nuclear_charges, charge=charge)
            return partial_cubic
        if order == 4:
            i,j,k,l = address[0], address[1], address[2], address[3]
            partial_quartic = jacfwd(jacfwd(jacfwd(jacfwd(ccsd_partial_wrapper, i), j), k), l)(*geom_list, basis_dict=basis_dict, nuclear_charges=nuclear_charges, charge=charge)
            return partial_quartic
        else:
            print("Error: {}'th order derivative tensor is not implemented nor recommended for CCSD.".format(order))
    else:
        print("Error: Method {} not supported.".format(method))


# Examples: Compute an energy, full derivative, or partial derivative
molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.80000000000
                         H 0.0 0.0  0.80000000000
                         units bohr
                         """)
basis_name = 'sto-3g'
#E_scf = energy(molecule, basis_name, 'scf')
#E_mp2 = energy(molecule, basis_name, 'mp2')
#E_ccsd = energy(molecule, basis_name, 'ccsd')

#grad = derivative(molecule, basis_name, 'scf', order=1)
#hess = derivative(molecule, basis_name,  'scf', order=2)
#cube = derivative(molecule, basis_name,  'scf', order=3)
#quar = derivative(molecule, basis_name,  'scf', order=4)

#partial_grad = partial_derivative(molecule, basis_name, 'scf', order=1, address=(2,)) 
#partial_hess = partial_derivative(molecule, basis_name, 'scf', order=2, address=(0,1)) 
#partial_cube = partial_derivative(molecule, basis_name, 'scf', order=3, address=(1,0,2)) 
#partial_quar = partial_derivative(molecule, basis_name, 'scf', order=4, address=(1,0,2,3)) 
#partial_quintic = partial_derivative(molecule, basis_name, 'scf', order=5, address=(5,5,5,5,5))
#partial_sextic = partial_derivative(molecule, basis_name, 'scf', order=6, address=(5,5,5,5,5,5))

psi4.core.be_quiet()
psi4.set_options({'basis': basis_name, 'scf_type': 'pk', 'mp2_type':'conv', 'e_convergence': 1e-10, 'diis': False, 'puream': 0})
print('PSI4 results')
psi_method = 'scf'
print(psi4.energy(psi_method + '/' +basis_name))
print(onp.asarray(psi4.gradient(psi_method+'/'+basis_name)))
print(onp.asarray(psi4.hessian(psi_method+'/'+basis_name)))


