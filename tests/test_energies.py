"""
Test energy computations
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
                  'd_convergence':1e-10,
                  'puream': 0})

def test_hartree_fock(method='hf'):
    psi_e = psi4.energy(method + '/' + basis_name)
    quax_e = quax.core.energy(molecule, basis_name, method)
    assert np.allclose(psi_e, quax_e)

def test_mp2(method='mp2'):
    psi_e = psi4.energy(method + '/' + basis_name)
    quax_e = quax.core.energy(molecule, basis_name, method)
    assert np.allclose(psi_e, quax_e)

def test_ccsd(method='ccsd'):
    psi_e = psi4.energy(method + '/' + basis_name)
    quax_e = quax.core.energy(molecule, basis_name, method)
    assert np.allclose(psi_e, quax_e)

def test_ccsd_t(method='ccsd(t)'):
    psi_e = psi4.energy(method + '/' + basis_name)
    quax_e = quax.core.energy(molecule, basis_name, method)
    assert np.allclose(psi_e, quax_e)
