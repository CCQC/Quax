import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import fori_loop
import psi4
import sys
jnp.set_printoptions(threshold=sys.maxsize, linewidth=100)

from .ints import compute_f12_oeints, compute_f12_teints # F12 TEINTS are entered in Chem and returned in Phys
from .energy_utils import partial_tei_transformation, f12_transpose
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

    f, fk, k = form_Fock(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)

    V = form_V(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, xyz_path, deriv_order, options)

    X = form_X(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, xyz_path, deriv_order, options)

    C = form_C(geom, basis_set, cabs_set, C_obs, C_cabs, f, ndocc, nobs, xyz_path, deriv_order, options)

    B = form_B(geom, basis_set, cabs_set, C_obs, C_cabs, f, fk, k, ndocc, nobs, nri, xyz_path, deriv_order, options)
    del fk
    jax.debug.breakpoint()

    return jnp.array([0])

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
    G_tmp = partial_tei_transformation(G_tmp, C_obs, C_obs[:, :ndocc], C_obs, C_obs)
    G = G.at[:nobs, :ndocc, :nobs, :nobs].set(G_tmp) # <Oo|OO>
    del G_tmp

    G_tmp = compute_f12_teints(geom, cabs_set, basis_set, basis_set, basis_set, "eri", xyz_path, deriv_order, options)
    G_tmp = partial_tei_transformation(G_tmp, C_cabs, C_obs, C_obs, C_obs)
    G = G.at[nobs:nri, :nobs, :nobs, :nobs].set(G_tmp) # <CO|OO>
    G = G.at[:nobs, :nobs, nobs:nri, :nobs].set(jnp.transpose(G_tmp, (2,1,0,3))) # <OO|CO>
    G = G.at[:nobs, :nobs, :nobs, nobs:nri].set(jnp.transpose(G_tmp, (3,2,1,0))) # <OO|OC>
    del G_tmp

    G_tmp = compute_f12_teints(geom, cabs_set, basis_set, basis_set, cabs_set, "eri", xyz_path, deriv_order, options)
    G_tmp = partial_tei_transformation(G_tmp, C_cabs, C_obs[:, :ndocc], C_obs, C_cabs)
    G = G.at[nobs:nri, :ndocc, :nobs, nobs:nri].set(G_tmp) # <Co|OC>
    del G_tmp

    G_tmp = compute_f12_teints(geom, cabs_set, cabs_set, basis_set, basis_set, "eri", xyz_path, deriv_order, options)
    G_tmp = partial_tei_transformation(G_tmp, C_cabs, C_obs[:, :ndocc], C_cabs, C_obs)
    G = G.at[nobs:nri, :ndocc, nobs:nri, :nobs].set(G_tmp) # <Co|CO>
    del G_tmp

    # Fill Fock Matrix
    f = f.at[:, :].add(2.0 * jnp.einsum('piqi->pq', G[:, :ndocc, :, :ndocc], optimize='optimal'))
    fk = f # Fock Matrix without Exchange
    k =  jnp.einsum('piiq->pq', G[:, :ndocc, :ndocc, :], optimize='optimal')
    f = f.at[:, :].add(-1.0 * k)
    del G

    return f, fk, k

def form_V(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, xyz_path, deriv_order, options):

    V = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12g12", xyz_path, deriv_order, options)
    V = partial_tei_transformation(V, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc])

    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, cabs_set, "f12", xyz_path, deriv_order, options)
    F_tmp = partial_tei_transformation(F_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_cabs)
    G_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, cabs_set, "eri", xyz_path, deriv_order, options)
    G_tmp = partial_tei_transformation(G_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_cabs)
    V_tmp = -1.0 * jnp.einsum('ijmy,klmy->ijkl', G_tmp, F_tmp, optimize='optimal')
    V = V.at[:, :, :, :].add(V_tmp)
    V = V.at[:, :, :, :].add(f12_transpose(V_tmp))
    del V_tmp
    del F_tmp
    del G_tmp

    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12", xyz_path, deriv_order, options)
    F_tmp = partial_tei_transformation(F_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :nobs], C_obs[:, :nobs])
    G_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "eri", xyz_path, deriv_order, options)
    G_tmp = partial_tei_transformation(G_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :nobs], C_obs[:, :nobs])
    V = V.at[:, :, :, :].add(-1.0 * jnp.einsum('ijrs,klrs->ijkl', G_tmp, F_tmp, optimize='optimal'))
    del F_tmp
    del G_tmp

    return V

def form_X(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, xyz_path, deriv_order, options):

    X = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12_squared", xyz_path, deriv_order, options)
    X = partial_tei_transformation(X, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc])

    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, cabs_set, "f12", xyz_path, deriv_order, options)
    F_tmp = partial_tei_transformation(F_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_cabs)
    X_tmp = -1.0 * jnp.einsum('ijmy,klmy->ijkl', F_tmp, F_tmp, optimize='optimal')
    X = X.at[:, :, :, :].add(X_tmp)
    X = X.at[:, :, :, :].add(f12_transpose(X_tmp))
    del X_tmp
    del F_tmp

    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12", xyz_path, deriv_order, options)
    F_tmp = partial_tei_transformation(F_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :nobs], C_obs[:, :nobs])
    X = X.at[:, :, :, :].add(-1.0 * jnp.einsum('ijrs,klrs->ijkl', F_tmp, F_tmp, optimize='optimal'))
    del F_tmp

    return X

def form_C(geom, basis_set, cabs_set, C_obs, C_cabs, Fock, ndocc, nobs, xyz_path, deriv_order, options):

    C = jnp.empty((ndocc, ndocc, nobs - ndocc, nobs - ndocc))

    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, cabs_set, "f12", xyz_path, deriv_order, options)
    F_tmp = partial_tei_transformation(F_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, ndocc:nobs], C_cabs)
    C_tmp = jnp.einsum('klay,by->klab', F_tmp, Fock[ndocc:nobs, nobs:], optimize='optimal')
    del F_tmp

    C = C.at[:, :, :, :].set(C_tmp)
    C = C.at[:, :, :, :].add(f12_transpose(C_tmp))
    del C_tmp

    return C

def form_B(geom, basis_set, cabs_set, C_obs, C_cabs, Fock, noK, K, ndocc, nobs, nri, xyz_path, deriv_order, options):
    # Term 1
    B = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12_double_commutator", xyz_path, deriv_order, options)
    B = partial_tei_transformation(B, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc])

    # Term 2
    F2 = jnp.empty((ndocc, ndocc, ndocc, nri))

    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12_squared", xyz_path, deriv_order, options)
    F2 = F2.at[:, :, :, :nobs].set(partial_tei_transformation(F_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs)) # <oo|oO>
    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, cabs_set, "f12_squared", xyz_path, deriv_order, options)
    F2 = F2.at[:, :, :, nobs:].set(partial_tei_transformation(F_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_cabs)) # <oo|oC>
    del F_tmp

    tmp = jnp.einsum('lknI,mI->lknm', F2, noK[:ndocc, :])
    del F2
    B = B.at[:, :, :, :].add(tmp)
    B = B.at[:, :, :, :].add(f12_transpose(tmp))
    del tmp

    # F12 Integral
    F_oo11 = jnp.empty((ndocc, ndocc, nri, nri))
    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12", xyz_path, deriv_order, options)
    F_oo11 = F_oo11.at[:, :, :nobs, :nobs].set(partial_tei_transformation(F_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs, C_obs)) # <oo|OO>
    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, cabs_set, "f12", xyz_path, deriv_order, options)
    F_tmp = partial_tei_transformation(F_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs, C_cabs)
    F_oo11 = F_oo11.at[:, :, :nobs, nobs:].set(F_tmp) # <oo|OC>
    F_oo11 = F_oo11.at[:, :, nobs:, :nobs].set(f12_transpose(F_tmp)) # <oo|CO>
    F_tmp = compute_f12_teints(geom, basis_set, cabs_set, basis_set, cabs_set, "f12", xyz_path, deriv_order, options)
    F_oo11 = F_oo11.at[:, :, nobs:, nobs:].set(partial_tei_transformation(F_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_cabs, C_cabs)) # <oo|CC>
    del F_tmp

    # Term 3
    tmp = -1.0 * jnp.einsum('lkPC,CA,nmPA->lknm', F_oo11, K, F_oo11, optimize='optimal')
    B = B.at[:, :, :, :].add(tmp)
    B = B.at[:, :, :, :].add(f12_transpose(tmp))

    # Term 4
    tmp = -1.0 * jnp.einsum('lkjC,CA,nmjA->lknm', F_oo11[:, :, :ndocc, :], Fock, F_oo11[:, :, :ndocc, :], optimize='optimal')
    B = B.at[:, :, :, :].add(tmp)
    B = B.at[:, :, :, :].add(f12_transpose(tmp))

    # Term 5
    tmp = jnp.einsum('lkxj,ji,nmxi->lknm', F_oo11[:, :, nobs:, :ndocc], Fock[:ndocc, :ndocc], F_oo11[:, :, nobs:, :ndocc], optimize='optimal')
    B = B.at[:, :, :, :].add(tmp)
    B = B.at[:, :, :, :].add(f12_transpose(tmp))

    # Term 6
    tmp = -1.0 * jnp.einsum('lkbp,pq,nmbq->lknm', F_oo11[:, :, ndocc:nobs, :nobs], Fock[:nobs, :nobs], F_oo11[:, :, ndocc:nobs, :nobs], optimize='optimal')
    B = B.at[:, :, :, :].add(tmp)
    B = B.at[:, :, :, :].add(f12_transpose(tmp))

    # Term 7
    tmp = -2.0 * jnp.einsum('lkxI,jI,nmxj->lknm', F_oo11[:, :, nobs:, :], Fock[:ndocc, :], F_oo11[:, :, nobs:, :ndocc], optimize='optimal')
    B = B.at[:, :, :, :].add(tmp)
    B = B.at[:, :, :, :].add(f12_transpose(tmp))

    # Term 8
    tmp = -2.0 * jnp.einsum('lkbq,qy,nmby->lknm', F_oo11[:, :, ndocc:nobs, :nobs], Fock[:nobs, nobs:], F_oo11[:, :, ndocc:nobs, nobs:], optimize='optimal')
    del F_oo11
    B = B.at[:, :, :, :].add(tmp)
    B = B.at[:, :, :, :].add(f12_transpose(tmp))

    tmp = jnp.transpose(B, (2,3,0,1))
    B = B.at[:, :, :, :].add(tmp)
    del tmp

    return 0.5 * B
