
import pprint
import psi4
geom = """
        0 1
        O
        H 1 R
        H 1 R 2 A
        R = 1.0
        A = 104.5
        """
#basis = 'sto-3g'
basis = 'cc-pvdz'
def build_basis_set(geom, basis):
    # Avoids printing from psi4
    psi4.core.be_quiet()
    # Create empty dictionary to hold basis information
    basis_dict = {}
    # Make Psi4 geometry
    molecule = psi4.geometry(geom)                   
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
        basis_dict[i]['exp'] = []
        basis_dict[i]['coef'] = []
        # Get total number of primitives for shell i
        nprim = basis_set.shell(i).nprimitive
        # Loop over each primitive in shell i
        for j in range(nprim):
            # Save the exponent and coefficient of each primitive
            basis_dict[i]['exp'].append(basis_set.shell(i).exp(j))
            basis_dict[i]['coef'].append(basis_set.shell(i).original_coef(j))
           # basis_dict[i]['coef'].append(basis_set.shell(i).coef(j))
    return basis_dict
        
basis_dict = build_basis_set(geom,basis)
basis_dict1 = build_basis_set(geom,basis)
pprint.pprint(basis_dict)
