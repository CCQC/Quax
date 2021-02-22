import psijax
import numpy as np
import jax.numpy as jnp
np.set_printoptions(linewidth=800, precision=10)

import psi4
import time
psi4.core.be_quiet()

molecule = psi4.geometry("""
                         0 1
                         N  0.000000000000     0.000000000000    -1.040129860737
                         N  0.000000000000     0.000000000000     1.040129860737
                         symmetry c1
                         units bohr
                         """)

basis_name = 'cc-pvtz'
psi4.set_memory(int(5e9))
psi4.set_options({'basis': basis_name, 'scf_type': 'pk', 'mp2_type':'conv', 'e_convergence': 1e-10, 'diis': True, 'd_convergence':1e-10, 'puream': 0, 'points':5, 'fd_project':False})

# Upper triangle CFOUR analytic Hessians
cfour_scf = np.array([[ 0.0543989239,  0.          ,  0.          , -0.0543989239,  0.          ,  0.          ],
                      [ 0.          ,  0.0543989239,  0.          ,  0.          , -0.0543989239,  0.          ],
                      [ 0.          ,  0.          ,  1.5672520204,  0.          ,  0.          , -1.5672520204],
                      [-0.0543989239,  0.          ,  0.          ,  0.0543989239,  0.          ,  0.          ],
                      [ 0.          , -0.0543989239,  0.          ,  0.          ,  0.0543989239,  0.          ],
                      [ 0.          ,  0.          , -1.5672520204,  0.          ,  0.          ,  1.5672520204]])


cfour_mp2 = np.array([[-0.0097469114,  0.          ,  0.          ,  0.0097469114,  0.          ,  0.          ],
                      [ 0.          , -0.0097469114,  0.          ,  0.          ,  0.0097469114,  0.          ],
                      [ 0.          ,  0.          ,  1.3980379518,  0.          ,  0.          , -1.3980379518],
                      [ 0.0097469114,  0.          ,  0.          , -0.0097469114,  0.          ,  0.          ],
                      [ 0.          ,  0.0097469114,  0.          ,  0.          , -0.0097469114,  0.          ],
                      [ 0.          ,  0.          , -1.3980379518,  0.          ,  0.          ,  1.3980379518]])

cfour_ccsdt = np.array([[ 0.0021704783,  0.          ,  0.          , -0.0021704783,  0.          ,  0.          ],
                        [ 0.          ,  0.0021704783,  0.          ,  0.          , -0.0021704783,  0.          ],
                        [ 0.          ,  0.          ,  1.473543491 ,  0.          ,  0.          , -1.473543491 ],
                        [-0.0021704783,  0.          ,  0.          ,  0.0021704783,  0.          ,  0.          ],
                        [ 0.          , -0.0021704783,  0.          ,  0.          ,  0.0021704783,  0.          ],
                        [ 0.          ,  0.          , -1.473543491 ,  0.          ,  0.          ,  1.473543491 ]])


print("Testing Energies")
for method in ['scf', 'mp2', 'ccsd(t)']:
    psi_e = psi4.energy(method + '/' + basis_name)
    psijax_e = psijax.core.energy(molecule, basis_name, method)
    print(psijax_e)
    print("\n{} energies match: ".format(method), np.allclose(psi_e, psijax_e, rtol=0.0, atol=1e-6), '\n')
    print('Error:', np.abs(psijax_e - psi_e))
print("\n")
print("\n")

print("Testing Hessians")
for method in ['scf', 'mp2', 'ccsd(t)']:
    if method == 'scf': cfour_deriv = cfour_scf
    if method == 'mp2': cfour_deriv = cfour_mp2
    if method == 'ccsd(t)': cfour_deriv = cfour_ccsdt
    psijax_deriv = np.asarray(psijax.core.derivative(molecule, basis_name, method, order=2))
    print("\n{} hessians match: ".format(method),np.allclose(psijax_deriv, cfour_deriv,rtol=0.0,atol=1e-4), '\n')
    print('Error:', np.abs(psijax_deriv - cfour_deriv))
print("\n")
print("\n")


