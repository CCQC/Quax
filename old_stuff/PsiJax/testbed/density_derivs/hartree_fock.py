import jax 
from jax.config import config; config.update("jax_enable_x64", True)
config.enable_omnistaging()
import jax.numpy as np
import psi4
import numpy as onp
np.set_printoptions(linewidth=500)

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
    SCF_MAX_ITER=20


    A = cholesky_orthogonalization(S)
    H = T + V

    @jax.jit
    def rhf_iter(H,A,G,D):
        J = np.einsum('pqrs,rs->pq', G, D)
        K = np.einsum('prqs,rs->pq', G, D)
        F = H + J * 2 - K
        E_scf = np.einsum('pq,pq->', F + H, D) 
        Fp = np.linalg.multi_dot((A.T, F, A))
        eps, C2 = np.linalg.eigh(Fp)
        C = np.dot(A,C2)
        Cocc = C[:, :ndocc]
        D = np.einsum('pi,qi->pq', Cocc, Cocc)
        return E_scf, D, C, eps 

    #def rhf_energy(H,A,G,D):
    #    J = np.einsum('pqrs,rs->pq', G, D)
    #    K = np.einsum('prqs,rs->pq', G, D)
    #    F = H + J * 2 - K
    #    E_scf = np.einsum('pq,pq->', F + H, D) 
    #    return E_scf

    #def delta_E(D_old, D_new):
    #    '''Derivatives of this function give us the derivative of the change in energy through an iteration wrt density '''
    #    J_old = np.einsum('pqrs,rs->pq', G, D_old)
    #    K_old = np.einsum('prqs,rs->pq', G, D_old)
    #    F_old = H + J_old * 2 - K_old
    #    E_old = np.einsum('pq,pq->', F_old + H, D_old) 

    #    J_new = np.einsum('pqrs,rs->pq', G, D_new)
    #    K_new = np.einsum('prqs,rs->pq', G, D_new)
    #    F_new = H + J_new * 2 - K_new
    #    E_new = np.einsum('pq,pq->', F_new + H, D_new) 

    #    delta_E = E_new - E_old
    #    return delta_E

    def delta_E(D_delta):
        '''Derivatives of this function give us the derivative of the change in energy through an iteration wrt density '''
        J = np.einsum('pqrs,rs->pq', G, D_delta)
        K = np.einsum('prqs,rs->pq', G, D_delta)
        F = H + J * 2 - K
        E = np.einsum('pq,pq->', F + H, D_delta) 

        return E

    #dE_dD = jax.jacrev(rhf_energy,3)
    #d2E_dD2 = jax.jacfwd(jax.jacrev(rhf_energy,3))

    # Should i do gradient wrt old density or new density?
    delta_E_grad0 = jax.jacrev(delta_E,0)
    #delta_E_grad1 = jax.jacrev(delta_E,1)

    iteration = 0
    E_scf = 1.0
    D = np.zeros_like(H)
    E_old = 0.0
    while abs(E_scf - E_old) > 1e-12:
        E_old = E_scf * 1
        D_old = D * 1
        E_scf, D, C, eps = rhf_iter(H,A,G,D_old)

        res = delta_E_grad0(D - D_old)
        print(np.round(res,5))
        
        #res0 = delta_E_grad0(D_old, D)
        #res1 = delta_E_grad1(D_old, D)
        #print(np.round(res0-res1,5) + np.round(res1-res0,5))
        

        #D_old = D_new * 1

        ## Derivative wrt density
        #deriv = dE_dD(H,A,G,D)
        #print(np.round(deriv,5))
        #deriv = d2E_dD2(H,A,G,D)
        #print(np.round(deriv,5))

        iteration += 1
        if iteration == SCF_MAX_ITER:
            break


    return E_scf


molecule = psi4.geometry("""
                         0 1
                         O
                         H 1 0.9
                         H 1 0.9 2 104.5
                         units bohr
                         """)

basis_name = 'sto-3g'
basis_dict = build_basis_set(molecule, basis_name)
geom = np.asarray(onp.asarray(molecule.geometry()))
mult = molecule.multiplicity()
charge = molecule.molecular_charge()
nuclear_charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])
nelectrons = int(np.sum(nuclear_charges)) - charge
ndocc = nelectrons // 2

#with open('h2sto3g_ints.npy', 'rb') as f:
#    STVG = np.load(f)

STVG = integrals(geom,basis_dict,nuclear_charges,charge)
hf(STVG)

#print(hf(STVG))

