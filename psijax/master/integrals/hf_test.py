import jax 
import jax.numpy as np
import psi4
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)

from tei import tei_array 
from oei import oei_arrays
from basis_utils import build_basis_set

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

basis_name = 'cc-pvdz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
nprim = basis_set.nprimitive()
nbf = basis_set.nbf()
print("Number of basis functions: ", nbf)
print("Number of primitives: ", nprim)
basis_dict = build_basis_set(molecule, basis_name)

def nuclear_repulsion(geom, nuclear_charges):
    natom = nuclear_charges.shape[0]
    nuc = 0
    for i in range(natom):
        for j in range(i):
            nuc += nuclear_charges[i] * nuclear_charges[j] / np.linalg.norm(geom[i] - geom[j])
    return nuc

def orthogonalizer(S):
    '''Compute overlap to the negative 1/2 power'''
    # STABLE FOR SMALL EIGENVALUES (not stable enough for H2 TZ grad apparently)
    # OG code
    eigval, eigvec = np.linalg.eigh(S)
    cutoff = 1.0e-12
    above_cutoff = (abs(eigval) > cutoff * np.max(abs(eigval)))
    val = 1 / np.sqrt(eigval[above_cutoff])
    vec = eigvec[:, above_cutoff]
    A = vec.dot(np.diag(val)).dot(vec.T)
    return A
    #return eigvec

    #A = vec.dot(np.diag(val))#.dot(vec.T)
    #A = np.dot(vec, np.diag(val))
    #eigval, eigvec = np.linalg.eigh(S)
    ##val = 1 / np.sqrt(eigval)
    #val = jax.lax.rsqrt(eigval)
    ##print(val)
    #A = eigvec.dot(np.diag(val)).dot(eigvec.T)
    #return A

#mints = psi4.core.MintsHelper(basis_set)
#psi_A = mints.ao_overlap()
#psi_A = np.asarray(onp.asarray(psi_A))

#res = jax.jacfwd(orthogonalizer, 0)(psi_A)
#print(res)
#res = jax.jacrev(orthogonalizer, 0)(psi_A)
#print(res)


#psi_A.power(-0.5, 1.e-16) #diagonalize S matrix and take to negative one half power 
#blah = orthogonalizer(psi_A)

#eigval, eigvec = np.linalg.eigh(psi_A)
#print(eigval)
#val = 1 / np.sqrt(eigval)
#print(eigvec.dot(np.diag(val)).dot(eigvec.T))


def hartree_fock(geom, basis, nuclear_charges, charge):
    nelectrons = int(np.sum(nuclear_charges)) + charge
    ndocc = nelectrons // 2

    S, T, V = oei_arrays(geom,basis,nuclear_charges)
    A = orthogonalizer(S)
    # Cholesky orthgonalization
    #A = np.linalg.inv(np.linalg.cholesky(S)).T
    G = tei_array(geom,basis)
    
    H = T + V
    Enuc = nuclear_repulsion(geom,nuclear_charges)
    D = np.zeros_like(H)
    
    for i in range(6):
        J = np.einsum('pqrs,rs->pq', G, D)                    
        K = np.einsum('prqs,rs->pq', G, D)                    
        F = H + J * 2 - K                                        
        E_scf = np.einsum('pq,pq->', F + H, D) + Enuc         
        print(E_scf)
        #Fp = A.dot(F).dot(A)
        Fp = A.T.dot(F).dot(A) # need transpose for cholesky orthogonalization
        eps, C2 = np.linalg.eigh(Fp)
        C = A.dot(C2)                                  
        Cocc = C[:, :ndocc]                                      
        D = np.einsum('pi,qi->pq', Cocc, Cocc)                
    return E_scf

#E_scf = hartree_fock(geom,basis_dict, nuclear_charges, charge)
#print(E_scf)

grad = jax.jacfwd(hartree_fock, 0)(geom,basis_dict, nuclear_charges, charge)
print(grad)

#hess = jax.jacfwd(jax.jacfwd(hartree_fock, 0))(geom,basis_dict, nuclear_charges, charge)
#print(hess)

#cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(hartree_fock, 0)))(geom,basis_dict, nuclear_charges, charge)
#print(cube)

#quar = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(hartree_fock, 0))))(geom,basis_dict, nuclear_charges, charge)
#print(quar)

# DEBUG ORTHOGONALIZER ERROR
#S, T, V = oei_arrays(geom,basis_dict,nuclear_charges)
#res = jax.jacfwd(orthogonalizer, 0)(S)
#print(res)

#def canorth(S):
#    "Canonical orthogonalization U/sqrt(lambda)"
#    E,U = np.linalg.eigh(S)
#    for i in range(len(E)):
#        U[:,i] = U[:,i] / np.sqrt(E[i])
#    return U
#
#res = jax.jacfwd(canorth, 0)(S)
#print(res)

#def cholorth(S):
#    "Cholesky orthogonalization"
#    return np.linalg.inv(np.linalg.cholesky(S)).T

#res = jax.jacfwd(cholorth, 0)(S)
#print(res)


psi4.set_options({'basis': basis_name, 'scf_type': 'pk', 'e_convergence': 1e-8, 'diis': False, 'puream': 0})
#print('PSI4 results')
print(psi4.energy('scf/'+basis_name))
print(onp.asarray(psi4.gradient('scf/'+basis_name)))
#print(onp.asarray(psi4.hessian('scf/'+basis_name)))

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



