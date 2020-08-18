# Code writer for benchmark files
h2 = """ 
molecule = psi4.geometry('''
                              0 1
                              H 0.0 0.0 -0.80000000000
                              H 0.0 0.0  0.80000000000
                              symmetry c1
                              units bohr
                              ''')
     """

n2 = """
molecule = psi4.geometry('''
                              0 1
                              N 0.0 0.0 -0.80000000000
                              N 0.0 0.0  0.80000000000
                              symmetry c1
                              units bohr
                              ''')
     """

h2o = """
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
"""

molecules = [h2, n2, h2o]
molecule_names = ['h2', 'n2', 'h2o']
methods = ['scf', 'mp2', 'ccsd(t)']
method_names = ['scf', 'mp2', 'ccsd_t']
basis_sets = ['cc-pvdz', 'cc-pvtz']
basis_names = ['dz', 'tz']

# Write energy files
for i in range(len(molecules)):
    for j in range(len(methods)):
        for k in range(len(basis_sets)):
            for l in range(1,5):
                if l == 1:
                    filename = '1_' + molecule_names[i] + '_' + method_names[j] + '_' + basis_names[k] + '.py'
                    print(filename)
                    with open(filename, 'w') as f: 
                        f.write('import psijax\nimport psi4\n\n')
                        f.write(molecules[i])
                        f.write('\n\n')
                        f.write("print('{}')".format(molecule_names[i] + ' ' + methods[j] + '/' + basis_sets[k] + ' partial gradient'))
                        f.write("\na = psijax.core.partial_derivative(molecule, '{}', '{}', order=1,address=(5,))".format(basis_sets[k], methods[j]))
                        f.write("\nprint(a)")
                if l == 2:
                    filename = '2_' + molecule_names[i] + '_' + method_names[j] + '_' + basis_names[k] + '.py'
                    print(filename)
                    with open(filename, 'w') as f: 
                        f.write('import psijax\nimport psi4\n\n')
                        f.write(molecules[i])
                        f.write('\n\n')
                        f.write("print('{}')".format(molecule_names[i] + ' ' + methods[j] + '/' + basis_sets[k] + ' partial hessian'))
                        f.write("\na = psijax.core.partial_derivative(molecule, '{}', '{}', order=2,address=(5,5))".format(basis_sets[k], methods[j]))
                        f.write("\nprint(a)")
                if l == 3:
                    filename = '3_' + molecule_names[i] + '_' + method_names[j] + '_' + basis_names[k] + '.py'
                    print(filename)
                    with open(filename, 'w') as f: 
                        f.write('import psijax\nimport psi4\n\n')
                        f.write(molecules[i])
                        f.write('\n\n')
                        f.write("print('{}')".format(molecule_names[i] + ' ' + methods[j] + '/' + basis_sets[k] + ' partial cubic'))
                        f.write("\na = psijax.core.partial_derivative(molecule, '{}', '{}', order=3,address=(5,5,5))".format(basis_sets[k], methods[j]))
                        f.write("\nprint(a)")

                if l == 4:
                    filename = '4_' + molecule_names[i] + '_' + method_names[j] + '_' + basis_names[k] + '.py'
                    print(filename)
                    with open(filename, 'w') as f: 
                        f.write('import psijax\nimport psi4\n\n')
                        f.write(molecules[i])
                        f.write('\n\n')
                        f.write("print('{}')".format(molecule_names[i] + ' ' + methods[j] + '/' + basis_sets[k] + ' partial quartic'))
                        f.write("\na = psijax.core.partial_derivative(molecule, '{}', '{}', order=4,address=(5,5,5,5))".format(basis_sets[k], methods[j]))
                        f.write("\nprint(a)")

