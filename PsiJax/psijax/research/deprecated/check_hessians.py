import psijax
import psi4
import jax
from jax.config import config; config.update("jax_enable_x64", True)
from psijax.integrals.basis_utils import build_basis_set
from psijax.integrals.tei import tei_array
from psijax.integrals.oei import oei_arrays
from psijax.methods.hartree_fock import restricted_hartree_fock
import jax.numpy as np
import numpy as onp
import os
np.set_printoptions(linewidth=800)

molecule = psi4.geometry("""
                         0 1
                         O  0.0  0.0  0.0
                         H  0.0  1.0  0.0
                         H  0.0  0.0  1.0
                         units ang 
                         """)
# NOTE flattened geometry
geom = onp.asarray(molecule.geometry())
geomflat = np.asarray(geom.flatten())
basis_name = 'sto-3g'
xyz_file_name = "geom.xyz"
# Save xyz file, get path
molecule.save_xyz_file(xyz_file_name, True)
xyz_path = os.path.abspath(os.getcwd()) + "/" + xyz_file_name

basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
mints = psi4.core.MintsHelper(basis_set)
basis_dict = build_basis_set(molecule, basis_name)
charge = molecule.molecular_charge()
nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])

# My libint interface integrals
eri1 = psijax.external_integrals.libint_interface.eri(xyz_path, basis_name)
dim = int(onp.sqrt(onp.sqrt(eri1.shape[0])))
eri1 = eri1.reshape(dim,dim,dim,dim)
eri2 = onp.asarray(mints.ao_eri())
print("Two electron integrals match:", onp.allclose(eri1,eri2))

## My Libint interface integral derivative
#deriv_vec = np.array([0,0,0,0,0,1])
#grad = psijax.external_integrals.libint_interface.eri_deriv(xyz_path, basis_name, deriv_vec)
#grad = grad.reshape(dim,dim,dim,dim)
#
def wrap(geomflat):
    geom = geomflat.reshape(-1,3)
    return tei_array(geom, basis_dict) 

#tei_grad = jax.jacfwd(wrap)(geomflat)
#print("gradients match?",onp.allclose(tei_grad[:,:,:,:,5], grad))

deriv_vec = np.array([0,0,0,0,0,2])
hess1 = psijax.external_integrals.libint_interface.eri_deriv(xyz_path, basis_name, deriv_vec)
hess1 = hess1.reshape(dim,dim,dim,dim)

#deriv_vec = np.array([0,0,0,0,1,1])
#hess2 = psijax.external_integrals.libint_interface.eri_deriv(xyz_path, basis_name, deriv_vec)
#hess2 = hess2.reshape(dim,dim,dim,dim)
#
#deriv_vec = np.array([0,0,1,0,0,1])
#hess3 = psijax.external_integrals.libint_interface.eri_deriv(xyz_path, basis_name, deriv_vec)
#hess3 = hess3.reshape(dim,dim,dim,dim)

tei_hess = jax.jacfwd(jax.jacfwd(wrap))(geomflat)
print("TEI diag hessian match?",onp.allclose(tei_hess[:,:,:,:,5,5], hess1))
print("TEI offdiag same atom hessian match?",onp.allclose(tei_hess[:,:,:,:,4,5], hess2))
print("TEI offdiag diff atom hessian match?",onp.allclose(tei_hess[:,:,:,:,2,5], hess3))

# My libint interface integrals
S = psijax.external_integrals.libint_interface.overlap(xyz_path, basis_name)
T = psijax.external_integrals.libint_interface.kinetic(xyz_path, basis_name)
V = psijax.external_integrals.libint_interface.potential(xyz_path, basis_name)
dim = int(onp.sqrt(S.shape[0]))
S = S.reshape(dim,dim)
T = T.reshape(dim,dim)
V = V.reshape(dim,dim)
S_psi = onp.asarray(mints.ao_overlap())
T_psi = onp.asarray(mints.ao_kinetic())
V_psi = onp.asarray(mints.ao_potential())

print("Overlap integrals match:", onp.allclose(S,S_psi))
print("Kinetic integrals match:", onp.allclose(T,T_psi))
print("Potential integrals match:", onp.allclose(V,V_psi))

def wrap_oeis(geomflat):
    geom = geomflat.reshape(-1,3)
    S, T, V = oei_arrays(geom,basis_dict,nuclear_charges)
    return S, T, V

overlap_hess, kinetic_hess, potential_hess = jax.jacfwd(jax.jacfwd(wrap_oeis))(geomflat)

deriv_vec = np.array([0,0,0,0,0,2])
overlap_hess1 = psijax.external_integrals.libint_interface.overlap_deriv(xyz_path, basis_name, deriv_vec)
overlap_hess1 = overlap_hess1.reshape(dim,dim)
print("overlap diag hessian match?",onp.allclose(overlap_hess[:,:,5,5], overlap_hess1))

kinetic_hess1 = psijax.external_integrals.libint_interface.kinetic_deriv(xyz_path, basis_name, deriv_vec)
kinetic_hess1 = kinetic_hess1.reshape(dim,dim)
print("kinetic diag hessian match?",onp.allclose(kinetic_hess[:,:,5,5], kinetic_hess1))

potential_hess1 = psijax.external_integrals.libint_interface.potential_deriv(xyz_path, basis_name, deriv_vec)
potential_hess1 = potential_hess1.reshape(dim,dim)
print("potential diag hessian match?",onp.allclose(potential_hess[:,:,5,5], potential_hess1))

deriv_vec = np.array([0,0,0,0,1,1])
overlap_hess1 = psijax.external_integrals.libint_interface.overlap_deriv(xyz_path, basis_name, deriv_vec)
overlap_hess1 = overlap_hess1.reshape(dim,dim)
print("overlap off diag same atom hessian match?",onp.allclose(overlap_hess[:,:,4,5], overlap_hess1))

kinetic_hess1 = psijax.external_integrals.libint_interface.kinetic_deriv(xyz_path, basis_name, deriv_vec)
kinetic_hess1 = kinetic_hess1.reshape(dim,dim)
print("kinetic off diag same atom  hessian match?",onp.allclose(kinetic_hess[:,:,4,5], kinetic_hess1))

potential_hess1 = psijax.external_integrals.libint_interface.potential_deriv(xyz_path, basis_name, deriv_vec)
potential_hess1 = potential_hess1.reshape(dim,dim)
print("potential off diag same atom hessian match?",onp.allclose(potential_hess[:,:,4,5], potential_hess1))

deriv_vec = np.array([0,0,1,0,0,1])
overlap_hess1 = psijax.external_integrals.libint_interface.overlap_deriv(xyz_path, basis_name, deriv_vec)
overlap_hess1 = overlap_hess1.reshape(dim,dim)
print("overlap off diag diff atom hessian match?",onp.allclose(overlap_hess[:,:,2,5], overlap_hess1))

kinetic_hess1 = psijax.external_integrals.libint_interface.kinetic_deriv(xyz_path, basis_name, deriv_vec)
kinetic_hess1 = kinetic_hess1.reshape(dim,dim)
print("kinetic off diag diff atom  hessian match?",onp.allclose(kinetic_hess[:,:,2,5], kinetic_hess1))

#potential_hess1 = psijax.external_integrals.libint_interface.potential_deriv(xyz_path, basis_name, deriv_vec)
#potential_hess1 = potential_hess1.reshape(dim,dim)
#print("potential off diag diff atom hessian match?",onp.allclose(potential_hess[:,:,2,5], potential_hess1))





