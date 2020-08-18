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
methods = ['scf', 'mp2', 'ccsd', 'ccsd(t)']
method_names = ['scf', 'mp2', 'ccsd', 'ccsd_t']
basis_sets = ['cc-pvdz', 'cc-pvtz']
basis_names = ['dz', 'tz']

# Write energy files
for i in range(len(molecules)):
    for j in range(len(methods)):
        for k in range(len(basis_sets)):
            filename = '0_' + molecule_names[i] + '_' + method_names[j] + '_' + basis_names[k] + '.py'
            print(filename)
            with open(filename, 'w') as f: 
                f.write('import psijax\nimport psi4\n\n')
                f.write(molecules[i])
                f.write('\n\n')
                f.write("print('{}')".format(molecule_names[i] + ' ' + methods[j] + '/' + basis_sets[k] + ' energy'))
                f.write("\npsijax.core.energy(molecule, '{}', '{}')".format(basis_sets[k], methods[j]))

# Write gradient files
for i in range(len(molecules)):
    for j in range(len(methods)):
        for k in range(len(basis_sets)):
            filename = '1_' + molecule_names[i] + '_' + method_names[j] + '_' + basis_names[k] + '.py'
            print(filename)
            with open(filename, 'w') as f: 
                f.write('import psijax\nimport psi4\n\n')
                f.write(molecules[i])
                f.write('\n\n')
                f.write("print('{}')".format(molecule_names[i] + ' ' + methods[j] + '/' + basis_sets[k] + ' gradient'))
                f.write("\npsijax.core.derivative(molecule, '{}', '{}', order=1)".format(basis_sets[k], methods[j]))

# Write hessian files
for i in range(len(molecules)):
    for j in range(len(methods)):
        for k in range(len(basis_sets)):
            filename = '2_' + molecule_names[i] + '_' + method_names[j] + '_' + basis_names[k] + '.py'
            print(filename)
            with open(filename, 'w') as f: 
                f.write('import psijax\nimport psi4\n\n')
                f.write(molecules[i])
                f.write('\n\n')
                f.write("print('{}')".format(molecule_names[i] + ' ' + methods[j] + '/' + basis_sets[k] + ' hessian'))
                f.write("\npsijax.core.derivative(molecule, '{}', '{}', order=2)".format(basis_sets[k], methods[j]))


