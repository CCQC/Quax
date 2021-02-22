import psi4 
import jax.numpy as np
import numpy as onp

def build_basis_set(molecule, basis):
    # Avoids printing from psi4
    psi4.core.be_quiet()
    # Create empty dictionary to hold basis information
    basis_dict = {}
    # Build basis in Psi4
    basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis, puream=0)
    # Get total number of shells for the molecule
    nshell = basis_set.nshell()
    # Loop over each shell
    for i in range(nshell):
        # Create subdictionary for shell i that contains angular momentum
        # and coefficient/exponent information for each primitive
        basis_dict[i] = {}
        basis_dict[i]['am'] = basis_set.shell(i).am
        basis_dict[i]['atom'] = basis_set.shell_to_center(i)
        basis_dict[i]['exp'] = []
        basis_dict[i]['coef'] = []
        basis_dict[i]['idx'] = basis_set.shell(i).function_index
        basis_dict[i]['idx_stride'] = int(0.5 * (basis_set.shell(i).am + 1) * ((basis_set.shell(i).am + 1) + 1))
        # Get total number of primitives for shell i
        nprim = basis_set.shell(i).nprimitive
        # Loop over each primitive in shell i
        for j in range(nprim):
            # Save the exponent and normalized coefficient of each primitive
            basis_dict[i]['exp'].append(basis_set.shell(i).exp(j))
            basis_dict[i]['coef'].append(basis_set.shell(i).coef(j))
    return basis_dict

def get_nbf(basis):
    nshells = len(basis)
    nbf = 0
    for i in range(nshells):
        nbf += basis[i]['idx_stride']
    return nbf

def flatten_basis_data(basis):
    """
    Takes in a dictionary of basis set info and flattens 
    all primitive data into vectors.
    """
    nshells = len(basis)
    coeffs = []
    exps = []
    atoms = []
    ams = []
    indices = []
    dims = []
    # Smush primitive data together into vectors
    nbf = 0
    for i in range(nshells):
        tmp_coeffs = basis[i]['coef']  
        tmp_exps = basis[i]['exp']  
        nbf += basis[i]['idx_stride']
        for j in tmp_coeffs:
            coeffs.append(j)
            atoms.append(basis[i]['atom'])
            ams.append(basis[i]['am'])
            indices.append(basis[i]['idx'])
            dims.append(basis[i]['idx_stride'])
        for j in tmp_exps:
            exps.append(j)
    coeffs = np.array(onp.asarray(coeffs))
    exps = np.array(onp.asarray(exps))
    atoms = np.array(onp.asarray(atoms))
    ams = np.array(onp.asarray(ams))
    indices = np.array(onp.asarray(indices))
    dims = np.array(onp.asarray(dims))
    return coeffs, exps, atoms, ams, indices, dims



