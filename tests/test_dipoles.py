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
psi4.set_options({'basis': basis_name,
                  'scf_type': 'pk',
                  'mp2_type':'conv',
                  'e_convergence': 1e-10,
                  'd_convergence':1e-10,
                  'puream': 0})

options = {'damping': True, 'spectral_shift': False}
efield = np.zeros((3))

def findif_dipole(method, pert):
    lambdas = [pert, -pert, 2.0*pert, -2.0*pert]
    dip_vec = np.zeros((3))

    for i in range(3):
        pert_vec = [0, 0, 0]
        energies = []
        for l in lambdas:
            pert_vec[i] = l
            psi4.set_options({'perturb_h': True,
                              'perturb_with': 'dipole',
                              'perturb_dipole': pert_vec})
            energies.append(psi4.energy(method))
        val = (8.0*energies[0] - 8.0*energies[1] - energies[2] + energies[3]) / (12.0*pert)
        dip_vec[i] = val
    return dip_vec

def test_hartree_fock_dipole(method='hf'):
    psi_deriv = findif_dipole(method, 0.0005)
    quax_deriv = quax.core.efield_deriv(molecule, basis_name, method, efield=efield, deriv_order=1, options=options).reshape(-1,3)
    quax_partial0 = quax.core.efield_deriv(molecule, basis_name, method, efield=efield, deriv_order=1, partial=(0,), options=options)
    assert np.allclose(psi_deriv, quax_deriv)
    assert np.allclose(psi_deriv[0], quax_partial0)

def test_mp2_dipole(method='mp2'):
    psi_deriv = findif_dipole(method, 0.0005)
    quax_deriv = quax.core.efield_deriv(molecule, basis_name, method, efield=efield, deriv_order=1, options=options).reshape(-1,3)
    quax_partial0 = quax.core.efield_deriv(molecule, basis_name, method, efield=efield, deriv_order=1, partial=(0,), options=options)
    assert np.allclose(psi_deriv, quax_deriv)
    assert np.allclose(psi_deriv[0], quax_partial0)

def test_ccsd_t_dipole(method='ccsd(t)'):
    psi_deriv = findif_dipole(method, 0.0005)
    quax_deriv = quax.core.efield_deriv(molecule, basis_name, method, efield=efield, deriv_order=1, options=options).reshape(-1,3)
    quax_partial0 = quax.core.efield_deriv(molecule, basis_name, method, efield=efield, deriv_order=1, partial=(0,), options=options)
    assert np.allclose(psi_deriv, quax_deriv, atol=1e-7)
    assert np.allclose(psi_deriv[0], quax_partial0)
