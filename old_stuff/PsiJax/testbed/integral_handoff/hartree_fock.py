import jax 
from jax.config import config; config.update("jax_enable_x64", True)
config.enable_omnistaging()
import jax.numpy as np
import psi4
import numpy as onp

import tei
import oei 
from energy_utils import nuclear_repulsion, cholesky_orthogonalization
from basis_utils import build_basis_set


def integrals(geom,basis,nuclear_charges,charge):
    STV = oei.oei_arrays(geom,basis,nuclear_charges)
    G = tei.tei_array(geom,basis)

    S = STV[0]
    T = STV[1]
    V = STV[2]
    nbf = STV.shape[1]
    # Create superarray with signature G=arr[0], S = arr[1,:,:,0,0], T=arr[1,:,:,0,1], V=arr[1,:,:,1,0]
    superarray = np.zeros((2,nbf,nbf,nbf,nbf))
    superarray = jax.ops.index_update(superarray, jax.ops.index[0,:,:,:,:], G)
    superarray = jax.ops.index_update(superarray, jax.ops.index[1,:,:,0,0], S)
    superarray = jax.ops.index_update(superarray, jax.ops.index[1,:,:,0,1], T)
    superarray = jax.ops.index_update(superarray, jax.ops.index[1,:,:,1,0], V)
    return superarray

def hf(STVG): 
    S = STVG[1,:,:,0,0]
    T = STVG[1,:,:,0,1]
    V = STVG[1,:,:,1,0]
    G = STVG[0]

    # Hardcoded
    ndocc = 1
    SCF_MAX_ITER=10
    # For slightly shifting eigenspectrum of Fp for degenerate eigenvalues 
    # (JAX cannot differentiate degenerate eigenvalue eigh) 
    seed = jax.random.PRNGKey(0)
    epsilon = 1e-9
    fudge = jax.random.uniform(seed, (S.shape[0],), minval=0.1, maxval=1.0) * epsilon
    fudge_factor = np.diag(fudge)


    A = cholesky_orthogonalization(S)
    H = T + V
    D = np.zeros_like(H)

    @jax.jit
    def rhf_iter(H,A,G,D):
        J = np.einsum('pqrs,rs->pq', G, D)
        K = np.einsum('prqs,rs->pq', G, D)
        F = H + J * 2 - K
        E_scf = np.einsum('pq,pq->', F + H, D) 
        Fp = np.linalg.multi_dot((A.T, F, A))
        Fp = Fp + fudge_factor
        
        eps, C2 = np.linalg.eigh(Fp)
        C = np.dot(A,C2)
        Cocc = C[:, :ndocc]
        D = np.einsum('pi,qi->pq', Cocc, Cocc)
        return E_scf, D, C, eps 

    iteration = 0
    E_scf = 1.0
    E_old = 0.0
    while abs(E_scf - E_old) > 1e-12:
        E_old = E_scf * 1
        E_scf, D, C, eps = rhf_iter(H,A,G,D)
        iteration += 1
        if iteration == SCF_MAX_ITER:
            break
    return E_scf


molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.8
                         H 0.0 0.0  0.8
                         units bohr
                         """)

basis_name = 'sto-3g'
basis_dict = build_basis_set(molecule, basis_name)
geom = np.asarray(onp.asarray(molecule.geometry()))
mult = molecule.multiplicity()
charge = molecule.molecular_charge()
nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])

# compute integrals and nuclear repulsion energy
#STVG = integrals(geom,basis_dict,nuclear_charges,charge)
#Enuc = nuclear_repulsion(geom,nuclear_charges)

## Gradients
#dInts_dgeom = jax.jacfwd(integrals, 0)(geom,basis_dict,nuclear_charges,charge) # (2,nbf,nbf,nbf,nbf,natom,3)
#dEnuc_dgeom = jax.jacfwd(nuclear_repulsion, 0)(geom,nuclear_charges) # (natom,3,natom,3) 
#dE_dInts = jax.jacrev(hf, 0)(STVG) # (2,nbf,nbf,nbf,nbf)
#dE_dgeom = np.einsum('ijklmno,ijklm->no', dInts_dgeom, dE_dInts)
#E_grad = dE_dgeom + dEnuc_dgeom
#print(E_grad)
## Gradients

# Hessians WAIT cant you just make a function that does what the above gradients thing does and just diffy it?
#dInts_dgeom = jax.jacfwd(jax.jacfwd(integrals, 0),0)(geom,basis_dict,nuclear_charges,charge) # (2,nbf,nbf,nbf,nbf,natom,3,natom,3)
#dEnuc_dgeom = jax.jacfwd(jax.jacfwd(nuclear_repulsion, 0),0)(geom,nuclear_charges) # (natom,3,natom,3)
# This blows up since you are not doing jacobian-vector products like JAX does internally.
#dE_dInts = jax.jacfwd(jax.jacrev(hf, 0),0)(STVG) # (2,nbf,nbf,nbf,nbf,nbf,nbf,nbf,nbf) (YIKES)
#print(dE_dInts.shape)
#print(dE_dInts)
#print(np.allclose(
#dE_dgeom = np.einsum('rijklabcd,ijklm->no', dInts_dgeom, dE_dInts)
#E_grad = dE_dgeom + dEnuc_dgeom
#print(E_grad)


def save_nuclear_grad():
    STVG = integrals(geom,basis_dict,nuclear_charges,charge)
    int_nuc_grad = onp.asarray(jax.jacfwd(integrals, 0)(geom,basis_dict,nuclear_charges,charge))
    int_nuc_hess = onp.asarray(jax.jacfwd(jax.jacfwd(integrals, 0),0)(geom,basis_dict,nuclear_charges,charge))
    with open('h2sto3g_int_nuc_hess.npy', 'wb') as f:
        onp.save(f,int_nuc_hess)
    with open('h2sto3g_int_nuc_grad.npy', 'wb') as f:
        onp.save(f,int_nuc_grad)
    with open('h2sto3g_ints.npy', 'wb') as f:
        onp.save(f,STVG)
    return None
#save_nuclear_grad()

def test(geom):
    # Compute integrals
    STVG = integrals(geom,basis_dict,nuclear_charges,charge)
    #Enuc = nuclear_repulsion(geom,nuclear_charges)

    # Compute nuclear derivative integrals (ideally these would just be given)
    dInts_dgeom = jax.jacfwd(integrals, 0)(geom,basis_dict,nuclear_charges,charge) # (2,nbf,nbf,nbf,nbf,natom,3)
    dEnuc_dgeom = jax.jacfwd(nuclear_repulsion, 0)(geom,nuclear_charges) # (natom,3,natom,3) 
    dE_dInts = jax.jacrev(hf, 0)(STVG) # (2,nbf,nbf,nbf,nbf)
    dE_dgeom = np.einsum('ijklmno,ijklm->no', dInts_dgeom, dE_dInts)
    E_grad = dE_dgeom + dEnuc_dgeom
    return E_grad

#print(test(geom))
#hess = jax.jacfwd(test, 0)(geom)
#print(hess)

def loaded_gradient(geom):
    with open('h2sto3g_int_nuc_grad.npy', 'rb') as f:
        dInts_dgeom = np.load(f)
    with open('h2sto3g_ints.npy', 'rb') as f:
        STVG = np.load(f)
    # We load in integrals and integral 1st derivatives, so JAX only needs to differentiate the energy and nuclear repulsion
    dEnuc_dgeom = jax.jacfwd(nuclear_repulsion, 0)(geom,nuclear_charges) # (natom,3,natom,3) 
    dE_dInts = jax.jacrev(hf, 0)(STVG) # (2,nbf,nbf,nbf,nbf)
    dE_dgeom = np.einsum('ijklmno,ijklm->no', dInts_dgeom, dE_dInts)
    E_grad = dE_dgeom + dEnuc_dgeom
    return E_grad

def loaded_hessian(geom):
    """
    Use second order Faa Di Bruno formula (tensor form)
    Question is: How does JAX do it to avoid storing the monster that 
    is dE2? The second derivative of energy w.r.t. integrals?
    Clearly JAX doesn't store this, so JAX must use Jacobian vector products or something.
    """
    with open('h2sto3g_ints.npy', 'rb') as f:
        STVG = np.load(f)
    with open('h2sto3g_int_nuc_grad.npy', 'rb') as f: # FIRST NUCLEAR INTEGRAL DERIVATIVE
        dI1 = np.load(f)
    with open('h2sto3g_int_nuc_hess.npy', 'rb') as f: # SECOND NUCLEAR INTEGRAL DERIVATIVE
        dI2 = np.load(f) # (2,nbf,nbf,nbf,nbf,natom,3,natom,3)

    dE1 = jax.jacrev(hf, 0)(STVG)
    dE2 = jax.jacfwd(jax.jacrev(hf, 0),0)(STVG) 
    term1 = np.einsum('AijklBabcd,AijklPQ,BabcdRS->PQRS', dE2, dI1, dI1) 
    term2 = np.einsum('Aijkl,AijklPQRS->PQRS', dE1, dI2) 
    dE_dgeom = term1 + term2

    # We load in integrals and integral 1st derivatives, so JAX only needs to differentiate the energy and nuclear repulsion
    dEnuc_dgeom = jax.jacfwd(jax.jacfwd(nuclear_repulsion, 0),0)(geom,nuclear_charges) # (natom,3,natom,3) 
    #dE_dgeom = np.einsum('Aijklabcd,ijklm->no', dInts_dgeom, dE_dInts)
    E_hess = dE_dgeom + dEnuc_dgeom
    return E_hess.reshape(6,6)


print(loaded_gradient(geom))
print(loaded_hessian(geom))

# Taking the derivative of test2 should not work, because we have no information about second derivative of ints
#print(jax.jacfwd(test2,0)(geom))

    



# Hessians


#
## GRADIENTS
## Compute dInts/dGeom
#dSTV_dgeom, dG_dgeom = jax.jacfwd(integrals, (0))(geom,basis_dict,nuclear_charges,charge)
## Compute dEnuc/dGeom
#dEnuc_dgeom = jax.jacfwd(nuclear_repulsion, 0)(geom,nuclear_charges)
## Compute dE/dInts
#dE_dSTV = jax.jacfwd(hf, 0)(STV,G)
##dE_dG = jax.jacfwd(hf, 1)(STV,G,Enuc) # This incurs massive memory for some reason?
#dE_dG = jax.jacrev(hf, 1)(STV,G) 
#
##print("dSTV_dgeom  ",dSTV_dgeom.shape)      #(3, 2, 2, 2, 3)         
##print("dE_dSTV     ",dE_dSTV.shape)         #(3, 2, 2)
##print("dG_dgeom    ",dG_dgeom.shape)        #(2, 2, 2, 2, 2, 3)
##print("dE_dG       ",dE_dG.shape)           #(2, 2, 2, 2)
##print("dEnuc_dgeom ",dEnuc_dgeom.shape)     #(2, 3)
##print("dE_dEnuc    ",dE_dEnuc.shape)        #()                 
#dOne =  np.einsum('ijklm,ijk->lm', dSTV_dgeom, dE_dSTV)
#dTwo = np.einsum('ijklmn,ijkl->mn', dG_dgeom, dE_dG)
#print("Nuclear gradient from decomposition")
#print(dOne + dTwo + dEnuc_dgeom)
# GRADIENTS

## Compute second derivatives
#d2STV_dgeom2, d2G_dgeom2, d2Enuc_dgeom2 = jax.jacfwd(jax.jacfwd(integrals, 0), 0)(geom,basis_dict,nuclear_charges,charge)
#result = jax.jacfwd(jax.jacrev(hf, (0,1,2)), (0,1,2))(STV,G,Enuc)
#print(len(result)) # lenght is 3, tuple of tuples
#print(result)
#
#for i in result:
#    print(i.shape)
##dE_dSTV = jax.jacfwd(hf, 0)(STV,G,Enuc)
##dE_dG = jax.jacfwd(jax.jacrev(hf, 1)(STV,G,Enuc) # This incurs massive memory for some reason?
##dE_dEnuc = jax.jacfwd(hf, 2)(STV,G,Enuc)


