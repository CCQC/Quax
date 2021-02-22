
import time
import itertools
import pprint
import psi4
import numpy as np

def old_cartesian_product(*arrays):
    '''Generalized cartesian product of any number of arrays'''
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

#cart_geom = [[ 0.0   ,0.0  ,0.0],
#        [ 0.757 ,0.586,0.0],
#        [ -0.757,0.586,0.0]]

cart_geom = [[ 0.0  ,0.0  ,0.0],  
                   [ 0.757,0.586,0.0],
                   [-0.757,0.586,0.0],
                   [ 2.757,0.586,0.0],
                   [ 3.757,0.586,0.0],
                   [ 4.757,0.586,0.0],
                   [ 5.757,0.586,0.0],
                   [ 6.757,0.586,0.0]]
geom = """
        0 1
        
        H 0.0 0.0 0.0
        H 0.757 0.586 0.0
        H -0.757 0.586 0.0
        H  2.757 0.586 0.0
        H  3.757 0.586 0.0
        H  4.757 0.586 0.0
        H  5.757 0.586 0.0
        H  6.757 0.586 0.0
        """

#basis = 'sto-3g'
basis = 'cc-pvtz'

        
def preprocess(geom, basis_dict, nshells):
    segment_id = 0
    data = []
    segment = []
    for i in range(nshells):
        for j in range(nshells):
            print(i, j)
            # Load data for this contracted integral
            c1 =    np.asarray(basis_dict[i]['coef'])
            c2 =    np.asarray(basis_dict[j]['coef'])
            exp1 =  np.asarray(basis_dict[i]['exp'])
            exp2 =  np.asarray(basis_dict[j]['exp'])
            atom1 = basis_dict[i]['atom']
            atom2 = basis_dict[j]['atom']
            Ax,Ay,Az = geom[atom1]
            Bx,By,Bz = geom[atom2]
            bra = basis_dict[i]['am']
            ket = basis_dict[j]['am']

            exp_combos = old_cartesian_product(exp1,exp2)
            coeff_combos = old_cartesian_product(c1,c2)
            for k in range(exp_combos.shape[0]):
                data.append(np.array([Ax,Ay,Az,Bx,By,Bz,exp_combos[k,0],exp_combos[k,1],coeff_combos[k,0], coeff_combos[k,1],int(bra),int(ket)]))
                segment.append(segment_id)
            segment_id += 1
    return np.asarray(data), np.asarray(segment)



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
        basis_dict[i]['am'] = basis_set.shell(i).am
        basis_dict[i]['atom'] = basis_set.shell_to_center(i)
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

    return basis_dict, nshell
        
basis_dict, nshell = build_basis_set(geom,basis)


t3 = time.time()
preprocess(cart_geom, basis_dict, nshell)
t4 = time.time()
print(t4-t3)

