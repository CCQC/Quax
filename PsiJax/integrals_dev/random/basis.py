import pprint
import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)

molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         units bohr
                         """)

# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))

#basis = 'sto-3g'
#basis = '6-31g'
basis = 'cc-pvtz'

basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis, puream=0)

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
        basis_dict[i]['am'] = basis_set.shell(i).amchar
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
        basis_dict[i]['exp'] = np.asarray(basis_dict[i]['exp'])
        basis_dict[i]['coef'] = np.asarray(basis_dict[i]['coef'])
    return basis_dict
        
basis_dict = build_basis_set(molecule,basis)
#pprint.pprint(basis_dict)
