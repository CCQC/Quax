import psijax
import numpy as onp
import jax.numpy as np
onp.set_printoptions(linewidth=800, precision=10)

import psi4
import time
psi4.core.be_quiet()

print("Running Accuracy Test: Testing Against Psi4 energies, gradients and Hessians")
print("Test system: N2 cc-pvdz")
molecule = psi4.geometry("""
                         0 1
                         N 0.0 0.0 -0.80000000000
                         N 0.0 0.0  0.80000000000
                         symmetry c1
                         units bohr
                         """)
basis_name = 'cc-pvdz'
psi4.set_options({'basis': basis_name, 'scf_type': 'pk', 'mp2_type':'conv', 'e_convergence': 1e-10, 'diis': False, 'puream': 0, 'points':5})

print("Testing Energies")
for method in ['scf', 'mp2', 'ccsd(t)']:
    psi_e = psi4.energy(method + '/' + basis_name)
    psijax_e = psijax.core.energy(molecule, basis_name, method)
    print("{} energies match: ".format(method), onp.allclose(psi_e, psijax_e), psi_e, psijax_e)
print("\n")

print("Testing Gradients")
for method in ['scf', 'mp2', 'ccsd(t)']:
    psi_deriv = onp.round(onp.asarray(psi4.gradient(method + '/' + basis_name)), 10)
    psijax_deriv = onp.asarray(psijax.core.derivative(molecule, basis_name, method, order=1)).reshape(-1,3)
    print("{} gradients match: ".format(method),onp.allclose(psijax_deriv, psi_deriv))
print("\n")

print("Testing Hessians")
for method in ['scf', 'mp2', 'ccsd(t)']:
    psi_deriv = onp.round(onp.asarray(psi4.hessian(method + '/' + basis_name, dertype='gradient')), 10)
    psijax_deriv = onp.asarray(psijax.core.derivative(molecule, basis_name, method, order=2))
    print("{} hessians match: ".format(method),onp.allclose(psijax_deriv, psi_deriv,rtol=1e-4,atol=1.e-4))
print("\n")

method = 'ccsd(t)'
print("Testing partial Hessians at CCSD(T)")
psijax_partial00 = psijax.core.partial_derivative(molecule, basis_name, method, order=2, address=(0,0))
psijax_partial01 = psijax.core.partial_derivative(molecule, basis_name, method, order=2, address=(0,1))
psijax_partial02 = psijax.core.partial_derivative(molecule, basis_name, method, order=2, address=(0,2))
psijax_partial03 = psijax.core.partial_derivative(molecule, basis_name, method, order=2, address=(0,3))
psijax_partial04 = psijax.core.partial_derivative(molecule, basis_name, method, order=2, address=(0,4))
psijax_partial05 = psijax.core.partial_derivative(molecule, basis_name, method, order=2, address=(0,5))

psijax_partial33 = psijax.core.partial_derivative(molecule, basis_name, method, order=2, address=(3,3))
psijax_partial34 = psijax.core.partial_derivative(molecule, basis_name, method, order=2, address=(3,4))
psijax_partial35 = psijax.core.partial_derivative(molecule, basis_name, method, order=2, address=(3,5))
psijax_partial55 = psijax.core.partial_derivative(molecule, basis_name, method, order=2, address=(5,5))

print("Checking first row of hessian")
print(onp.allclose(psi_deriv[0,0], psijax_partial00,rtol=1e-4,atol=1.e-4),psi_deriv[0,0], psijax_partial00)
print(onp.allclose(psi_deriv[0,1], psijax_partial01,rtol=1e-4,atol=1.e-4),psi_deriv[0,1], psijax_partial01)
print(onp.allclose(psi_deriv[0,2], psijax_partial02,rtol=1e-4,atol=1.e-4),psi_deriv[0,2], psijax_partial02)
print(onp.allclose(psi_deriv[0,3], psijax_partial03,rtol=1e-4,atol=1.e-4),psi_deriv[0,3], psijax_partial03)
print(onp.allclose(psi_deriv[0,4], psijax_partial04,rtol=1e-4,atol=1.e-4),psi_deriv[0,4], psijax_partial04)
print(onp.allclose(psi_deriv[0,5], psijax_partial05,rtol=1e-4,atol=1.e-4),psi_deriv[0,5], psijax_partial05)
print("Checking third row of hessian")
print(onp.allclose(psi_deriv[3,3], psijax_partial33,rtol=1e-4,atol=1.e-4),psi_deriv[3,3], psijax_partial33)
print(onp.allclose(psi_deriv[3,4], psijax_partial34,rtol=1e-4,atol=1.e-4),psi_deriv[3,4], psijax_partial34)
print(onp.allclose(psi_deriv[3,5], psijax_partial35,rtol=1e-4,atol=1.e-4),psi_deriv[3,5], psijax_partial35)
print("Checking last element of hessian")
print(onp.allclose(psi_deriv[5,5], psijax_partial55,rtol=1e-4,atol=1.e-4),psi_deriv[5,5], psijax_partial55)


