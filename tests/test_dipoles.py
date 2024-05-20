"""
Test gradient computations
"""
import quax
import psi4
import pytest
import numpy as np

molecule = psi4.geometry("""
0 1
O   -0.000007070942     0.125146536460     0.000000000000
H   -1.424097055410    -0.993053750648     0.000000000000
H    1.424209276385    -0.993112599269     0.000000000000
units bohr
""")
basis_name = 'sto-3g'
psi4.set_options({
                  'basis': basis_name,
                  'scf_type': 'pk',
                  'mp2_type':'conv',
                  'e_convergence': 1e-10,
                  'd_convergence':1e-10,
                  'puream': 0
                })

options = {'damping':True, 'spectral_shift':False}
efield = np.zeros((3))

def test_hartree_fock_gradient(method='hf'):
    psi4.properties(method, properties=['dipole'])
    psi_deriv = psi4.variable("SCF DIPOLE")
    quax_deriv = quax.core.efield_deriv(molecule, basis_name, method, efield=efield, deriv_order=1, options=options).reshape(-1,3)
    quax_partial0 = quax.core.efield_deriv(molecule, basis_name, method, efield=efield, deriv_order=1, partial=(0,), options=options)
    assert np.allclose(psi_deriv, quax_deriv)
    assert np.allclose(psi_deriv[0], quax_partial0)

def test_ccsd_gradient(method='ccsd'):
    psi4.properties(method, properties=['dipole'])
    psi_deriv = psi4.variable("CC DIPOLE")
    quax_deriv = quax.core.efield_deriv(molecule, basis_name, method, efield=efield, deriv_order=1, options=options).reshape(-1,3)
    quax_partial0 = quax.core.efield_deriv(molecule, basis_name, method, efield=efield, deriv_order=1, partial=(0,), options=options)
    assert np.allclose(psi_deriv, quax_deriv, rtol=1e-4, atol=1e-4)
    assert np.allclose(psi_deriv[0], quax_partial0)


