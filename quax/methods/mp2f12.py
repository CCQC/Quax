import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import fori_loop
import psi4
import sys
jnp.set_printoptions(threshold=sys.maxsize, linewidth=100)

from .ints import compute_f12_oeints, compute_f12_teints
from .mp2 import restricted_mp2

def restricted_mp2_f12(geom, basis_set, xyz_path, nuclear_charges, charge, options, cabs_space, deriv_order=0):
    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2
    E_mp2, C_obs, eps = restricted_mp2(geom, basis_set, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order, return_aux_data=True)

    print("Running MP2-F12 Computation...")
    cabs_set = cabs_space.basisset()
    C_cabs = jnp.array(cabs_space.C().to_array())
    nobs = C_obs.shape[0]
    nri = C_cabs.shape[0]

    f, fk = form_Fock(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)

    V = form_V(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, xyz_path, deriv_order, options)

    X = form_X(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, xyz_path, deriv_order, options)

    C = form_C(geom, basis_set, cabs_set, C_obs, C_cabs, f, ndocc, nobs, xyz_path, deriv_order, options)
    jax.debug.breakpoint()

    return 0

def form_h(geom, basis_set, cabs_set, C_obs, C_cabs, nobs, nri, xyz_path, deriv_order, options):
    h = jnp.empty((nri, nri))

    h_tmp = compute_f12_oeints(geom, basis_set, basis_set, xyz_path, deriv_order, options)
    h_tmp = jnp.einsum('pP,qQ,pq->PQ', C_obs, C_obs, h_tmp, optimize='optimal')
    h = h.at[:nobs, :nobs].set(h_tmp) # <O|O>
    del h_tmp

    h_tmp = compute_f12_oeints(geom, basis_set, cabs_set, xyz_path, deriv_order, options)
    h_tmp = jnp.einsum('pP,qQ,pq->PQ', C_obs, C_cabs, h_tmp, optimize='optimal')
    h = h.at[:nobs, nobs:nri].set(h_tmp) # <O|C>
    h = h.at[nobs:nri, :nobs].set(jnp.transpose(h_tmp)) # <C|O>
    del h_tmp

    h_tmp = compute_f12_oeints(geom, cabs_set, cabs_set, xyz_path, deriv_order, options)
    h_tmp = jnp.einsum('pP,qQ,pq->PQ', C_cabs, C_cabs, h_tmp, optimize='optimal')
    h = h.at[nobs:nri, nobs:nri].set(h_tmp) # <C|C>
    del h_tmp

    return h

def form_Fock(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options):
    # OEINTS
    f = form_h(geom, basis_set, cabs_set, C_obs, C_cabs, nobs, nri, xyz_path, deriv_order, options)

    # TEINTS
    G = jnp.empty((nri, nobs, nri, nri))

    G_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "eri", xyz_path, deriv_order, options)
    G_tmp = jnp.einsum('pP,rR,qQ,sS,prqs->PQRS', C_obs, C_obs, C_obs[:, :ndocc], C_obs, G_tmp, optimize='optimal')
    G = G.at[:nobs, :ndocc, :nobs, :nobs].set(G_tmp) # <Oo|OO>
    del G_tmp

    G_tmp = compute_f12_teints(geom, cabs_set, basis_set, basis_set, basis_set, "eri", xyz_path, deriv_order, options)
    G_tmp = jnp.einsum('pP,rR,qQ,sS,prqs->PQRS', C_cabs, C_obs, C_obs, C_obs, G_tmp, optimize='optimal')
    G = G.at[nobs:nri, :nobs, :nobs, :nobs].set(G_tmp) # <CO|OO>
    G = G.at[:nobs, :nobs, nobs:nri, :nobs].set(jnp.transpose(G_tmp, (2,1,0,3))) # <OO|CO>
    G = G.at[:nobs, :nobs, :nobs, nobs:nri].set(jnp.transpose(G_tmp, (3,2,1,0))) # <OO|OC>
    del G_tmp

    G_tmp = compute_f12_teints(geom, cabs_set, basis_set, basis_set, cabs_set, "eri", xyz_path, deriv_order, options)
    G_tmp = jnp.einsum('pP,rR,qQ,sS,prqs->PQRS', C_cabs, C_obs, C_obs[:, :ndocc], C_cabs, G_tmp, optimize='optimal')
    G = G.at[nobs:nri, :ndocc, :nobs, nobs:nri].set(G_tmp) # <Co|OC>
    del G_tmp

    G_tmp = compute_f12_teints(geom, cabs_set, cabs_set, basis_set, basis_set, "eri", xyz_path, deriv_order, options)
    G_tmp = jnp.einsum('pP,rR,qQ,sS,prqs->PQRS', C_cabs, C_cabs, C_obs[:, :ndocc], C_obs, G_tmp, optimize='optimal')
    G = G.at[nobs:nri, :ndocc, nobs:nri, :nobs].set(G_tmp) # <Co|CO>
    del G_tmp

    # Fill Fock Matrix
    f = f.at[:, :].add(2.0 * jnp.einsum('PIQI->PQ', G[:, :ndocc, :, :ndocc], optimize='optimal'))
    fk = f # Fock Matrix without Exchange
    f = f.at[:, :].add(-1.0 * jnp.einsum('PIIQ->PQ', G[:, :ndocc, :ndocc, :], optimize='optimal'))

    return f, fk

def form_V(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, xyz_path, deriv_order, options):

    V = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12g12", xyz_path, deriv_order, options)
    V = jnp.einsum('iI,kK,jJ,lL,ikjl->IJKL', C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], V, optimize='optimal')

    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, cabs_set, "f12", xyz_path, deriv_order, options)
    F_tmp = jnp.einsum('iI,mM,jJ,yY,imjy->IJMY', C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_cabs, F_tmp, optimize='optimal')
    G_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, cabs_set, "eri", xyz_path, deriv_order, options)
    G_tmp = jnp.einsum('kK,mM,lL,yY,kmly->KLMY', C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_cabs, G_tmp, optimize='optimal')
    V_tmp = -1.0 * jnp.einsum('IJMY,KLMY->IJKL', F_tmp, G_tmp, optimize='optimal')
    V = V.at[:, :, :, :].add(V_tmp)
    V = V.at[:, :, :, :].add(jnp.transpose(V_tmp, (1,0,3,2)))
    del V_tmp
    del F_tmp
    del G_tmp

    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12", xyz_path, deriv_order, options)
    F_tmp = jnp.einsum('iI,rR,jJ,sS,irjs->IJRS', C_obs[:, :ndocc], C_obs[:, :nobs], C_obs[:, :ndocc], C_obs[:, :nobs], F_tmp, optimize='optimal')
    G_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "eri", xyz_path, deriv_order, options)
    G_tmp = jnp.einsum('kK,rR,lL,sS,krls->KLRS', C_obs[:, :ndocc], C_obs[:, :nobs], C_obs[:, :ndocc], C_obs[:, :nobs], G_tmp, optimize='optimal')
    V = V.at[:, :, :, :].add(-1.0 * jnp.einsum('IJRS,KLRS->IJKL', F_tmp, G_tmp, optimize='optimal'))
    del F_tmp
    del G_tmp

    return V

def form_X(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, xyz_path, deriv_order, options):

    X = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12_squared", xyz_path, deriv_order, options)
    X = jnp.einsum('iI,kK,jJ,lL,ikjl->IJKL', C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], X, optimize='optimal')

    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, cabs_set, "f12", xyz_path, deriv_order, options)
    F_tmp = jnp.einsum('iI,mM,jJ,yY,imjy->IJMY', C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_cabs, F_tmp, optimize='optimal')
    X_tmp = -1.0 * jnp.einsum('IJMY,KLMY->IJKL', F_tmp, F_tmp, optimize='optimal')
    X = X.at[:, :, :, :].add(X_tmp)
    X = X.at[:, :, :, :].add(jnp.transpose(X_tmp, (1,0,3,2)))
    del X_tmp
    del F_tmp

    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12", xyz_path, deriv_order, options)
    F_tmp = jnp.einsum('iI,rR,jJ,sS,irjs->IJRS', C_obs[:, :ndocc], C_obs[:, :nobs], C_obs[:, :ndocc], C_obs[:, :nobs], F_tmp, optimize='optimal')
    X = X.at[:, :, :, :].add(-1.0 * jnp.einsum('IJRS,KLRS->IJKL', F_tmp, F_tmp, optimize='optimal'))
    del F_tmp

    return X

def form_C(geom, basis_set, cabs_set, C_obs, C_cabs, Fock, ndocc, nobs, xyz_path, deriv_order, options):

    C = jnp.empty((ndocc, ndocc, nobs - ndocc, nobs - ndocc))

    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, cabs_set, "f12", xyz_path, deriv_order, options)
    F_tmp = jnp.einsum('kK,aA,lL,yY,kaly->KLAY', C_obs[:, :ndocc], C_obs[:, ndocc:nobs], C_obs[:, :ndocc], C_cabs, F_tmp, optimize='optimal')
    C_tmp = jnp.einsum('KLAY,BY->KLAB', F_tmp, Fock[ndocc:nobs, nobs:])
    del F_tmp

    C = C.at[:, :, :, :].set(C_tmp)
    C = C.at[:, :, :, :].add(jnp.transpose(C_tmp, (1,0,3,2)))

    return C
