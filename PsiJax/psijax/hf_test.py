import jax 
import jax.numpy as np
import psi4
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
#config.update("jax_debug_nans", True)

from tei import tei_array 
from oei import oei_arrays
from basis_utils import build_basis_set
from energy_utils import nuclear_repulsion, symmetric_orthogonalization, cholesky_orthogonalization

molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.80000000000
                         H 0.0 0.0  0.80000000000
                         units bohr
                         """)

geom = np.asarray(onp.asarray(molecule.geometry()))
charge = molecule.molecular_charge()
mult = molecule.multiplicity()
nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])

basis_name = 'cc-pvtz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
nprim = basis_set.nprimitive()
nbf = basis_set.nbf()
print("Number of basis functions: ", nbf)
basis_dict = build_basis_set(molecule, basis_name)

def hartree_fock(geom, basis, nuclear_charges, charge):
    nelectrons = int(np.sum(nuclear_charges)) + charge
    ndocc = nelectrons // 2

    S, T, V = oei_arrays(geom,basis,nuclear_charges)
    #A = symmetric_orthogonalization(S)
    A = cholesky_orthogonalization(S)
    G = tei_array(geom,basis)
    
    H = T + V
    Enuc = nuclear_repulsion(geom,nuclear_charges)
    D = np.zeros_like(H)

    @jax.jit
    def hf_iter(D,H,G, A, Enuc):
        J = np.einsum('pqrs,rs->pq', G, D)
        K = np.einsum('prqs,rs->pq', G, D)
        F = H + J * 2 - K
        E_scf = np.einsum('pq,pq->', F + H, D) + Enuc
        Fp = np.linalg.multi_dot((A.T, F, A))
        # Slightly shift eigenspectrum of Fp for degenerate eigenvalues 
        # (JAX cannot differentiate degenerate eigenvalue eigh) 
        # TODO are there consequences to doing this for correlated methods?
        seed = jax.random.PRNGKey(0)
        fudge = jax.random.uniform(seed, (Fp.shape[0],)) / 1e10
        Fp = Fp + fudge
        eps, C2 = np.linalg.eigh(Fp)
        C = np.dot(A,C2)
        Cocc = C[:, :ndocc]
        D = np.einsum('pi,qi->pq', Cocc, Cocc)
        return D, E_scf  
    
    SCF_MAX_ITER = 15
    iteration = 0
    E_scf = 1.0
    E_old = 0.0
    while abs(E_scf - E_old) > 1e-9:
        E_old = E_scf * 1
        D, E_scf = hf_iter(D, H, G, A, Enuc)
        iteration += 1
        if iteration == SCF_MAX_ITER:
            break
    return E_scf

E_scf = hartree_fock(geom,basis_dict, nuclear_charges, charge)
print(E_scf)

#grad = jax.jacfwd(hartree_fock, 0)(geom,basis_dict, nuclear_charges, charge)
#print(grad)

hess = jax.jacfwd(jax.jacfwd(hartree_fock, 0))(geom,basis_dict, nuclear_charges, charge)
print(hess)

#cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(hartree_fock, 0)))(geom,basis_dict, nuclear_charges, charge)
#print(cube)

#quar = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(hartree_fock, 0))))(geom,basis_dict, nuclear_charges, charge)
#print(quar)

psi4.set_options({'basis': basis_name, 'scf_type': 'pk', 'e_convergence': 1e-8, 'diis': False, 'puream': 0})
print('PSI4 results')
print(psi4.energy('scf/'+basis_name))
#print(onp.asarray(psi4.gradient('scf/'+basis_name)))
print(onp.asarray(psi4.hessian('scf/'+basis_name)))

# Check integrals
#S, T, V = oei_arrays(geom,basis_dict,nuclear_charges)
#G = tei_array(geom,basis_dict)
#A = orthogonalizer(S)
#mints = psi4.core.MintsHelper(basis_set)
#psi_S = np.asarray(onp.asarray(mints.ao_overlap()))
#psi_T = np.asarray(onp.asarray(mints.ao_kinetic()))
#psi_V = np.asarray(onp.asarray(mints.ao_potential()))
#psi_G = np.asarray(onp.asarray(mints.ao_eri()))
#psi_A = mints.ao_overlap()
#psi_A.power(-0.5, 1.e-16) #diagonalize S matrix and take to negative one half power 
#psi_A = np.asarray(onp.asarray(psi_A))
#print("Overlap matches Psi4: ", np.allclose(S, psi_S))
#print("Kinetic matches Psi4: ", np.allclose(T, psi_T))
#print("Potential matches Psi4: ", np.allclose(V, psi_V))
#print("ERI matches Psi4: ", np.allclose(G, psi_G))
#print("Orthogonalizer matches Psi4: ", np.allclose(A, psi_A))



