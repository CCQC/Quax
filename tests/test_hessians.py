"""
Test hessian computations
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
psi4.set_options({'basis': basis_name,
                  'scf_type': 'pk',
                  'mp2_type':'conv',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10,
                  'puream': 0,
                  'points': 5,
                  'fd_project': False})

options = {'damping': True, 'spectral_shift': False}

def test_hartree_fock_hessian(method='hf'):
    psi_deriv = np.round(np.asarray(psi4.hessian(method + '/' + basis_name)), 10)
    n = psi_deriv.shape[0]
    quax_deriv = quax.core.geom_deriv(molecule, basis_name, method, deriv_order=2, options=options).reshape(n,n)
    quax_partial00 = quax.core.geom_deriv(molecule, basis_name, method, deriv_order=2, partial=(0,0), options=options)
    assert np.allclose(psi_deriv, quax_deriv, rtol=5e-5)
    assert np.allclose(psi_deriv[0,0], quax_partial00)

def test_mp2_hessian(method='mp2'):
    psi_deriv = np.round(np.asarray(psi4.hessian(method + '/' + basis_name, dertype='gradient')), 10)
    n = psi_deriv.shape[0]
    quax_deriv = quax.core.geom_deriv(molecule, basis_name, method, deriv_order=2, options=options).reshape(n,n)
    quax_partial00 = quax.core.geom_deriv(molecule, basis_name, method, deriv_order=2, partial=(0,0), options=options)
    assert np.allclose(psi_deriv, quax_deriv, rtol=5e-5)
    assert np.allclose(psi_deriv[0,0], quax_partial00)

def test_ccsd_t_hessian(method='ccsd(t)'):
    psi_deriv = np.round(np.asarray(psi4.hessian(method + '/' + basis_name, dertype='energy')), 10)
    n = psi_deriv.shape[0]
    quax_deriv = quax.core.geom_deriv(molecule, basis_name, method, deriv_order=2, options=options).reshape(n,n)
    quax_partial00 = quax.core.geom_deriv(molecule, basis_name, method, deriv_order=2, partial=(0,0), options=options)
    assert np.allclose(psi_deriv, quax_deriv, rtol=7e-5)
    assert np.allclose(psi_deriv[0,0], quax_partial00)

