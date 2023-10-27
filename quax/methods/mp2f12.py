import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import fori_loop
import psi4
import sys
jnp.set_printoptions(threshold=sys.maxsize, linewidth=100)

from .ints import compute_f12_oeints, compute_f12_teints
from .energy_utils import partial_tei_transformation, f12_transpose
from .mp2 import restricted_mp2

def restricted_mp2_f12(geom, basis_set, xyz_path, nuclear_charges, charge, options, cabs_space, deriv_order=0):
    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2
    E_mp2, C_obs, eps = restricted_mp2(geom, basis_set, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order, return_aux_data=True)
    eps_occ, eps_vir = eps[:ndocc], eps[ndocc:]

    print("Running MP2-F12 Computation...")
    cabs_set = cabs_space.basisset()
    C_cabs = jnp.asarray(cabs_space.C().to_array())
    nobs = C_obs.shape[0]
    nri = C_cabs.shape[0]

    f, fk, k = form_Fock(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)

    V = form_V(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, xyz_path, deriv_order, options)

    X = form_X(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, xyz_path, deriv_order, options)

    C = form_C(geom, basis_set, cabs_set, C_obs, C_cabs, f, ndocc, nobs, xyz_path, deriv_order, options)

    B = form_B(geom, basis_set, cabs_set, C_obs, C_cabs, f, fk, k, ndocc, nobs, nri, xyz_path, deriv_order, options)

    D = -1.0 * jnp.reciprocal(eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1, 1) - eps_vir.reshape(-1, 1) - eps_vir)

    G = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "eri", xyz_path, deriv_order, options)
    G = partial_tei_transformation(G, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, ndocc:nobs], C_obs[:, ndocc:nobs])
    
    indices = jnp.asarray(jnp.triu_indices(ndocc)).reshape(2,-1).T

    def loop_energy(idx, f12_corr):
        i,j = indices[idx]
        kd = jax.lax.cond(i == j, lambda: 1.0, lambda: 2.0)

        V_ij = V[i, j, :, :]
        V_ij = V_ij.at[:, :].add(-1.0 *jnp.einsum('klab,ab,ab->kl', C, G[i, j, :, :], D[i, j, :, :], optimize='optimal'))

        V_s = 0.5 * (t_(i, j, i, j) + t_(i, j, j, i)) * kd * (V_ij[i, j] + V_ij[j, i])

        V_t = 0.5 * jax.lax.cond(i != j, lambda: (t_(i, j, i, j) - t_(i, j, j, i))
                                               * kd * (V_ij[i, j] - V_ij[j, i]), lambda: 0.0)

        B_ij = B - (X * (f[i, i] + f[j, j]))
        B_ij = B_ij.at[:, :, :, :].add(-1.0 * jnp.einsum('klab,ab,mnab', C, D[i, j, :, :], C, optimize='optimal'))

        B_s = 0.125 * (t_(i, j, i, j) + t_(i, j, j, i)) * kd \
                     * (B_ij[i, j, i, j] + B_ij[j, i, i, j]) \
                     * (t_(i, j, i, j) + t_(i, j, j, i)) * kd

        B_t = 0.125 * jax.lax.cond(i != j, lambda: (t_(i, j, i, j) - t_(i, j, j, i)) * kd
                                                 * (B_ij[i, j, i, j] - B_ij[j, i, i, j])
                                                 * (t_(i, j, i, j) - t_(i, j, j, i)) * kd,
                                                 lambda: 0.0)

        f12_corr += kd * (V_s + B_s)
        f12_corr += 3.0 * kd * (V_t + B_t)

        return f12_corr

    dE_mp2f12 = fori_loop(0, indices.shape[0], loop_energy, 0.0)

    return E_mp2 + dE_mp2f12

# Fixed Amplitude Ansatz
@jax.jit
def t_(p = 0, q = 0, r = 0, s = 0):
    return jnp.select(
        [(p == q) & (p == r) & (p ==s), (p == r) & (q == s), (p == s) & (q == r)],
        [0.5, 0.375, 0.125],
        default = jnp.nan
    )

# One-Electron Integrals

def form_h(geom, basis_set, cabs_set, C_obs, C_cabs, nobs, nri, xyz_path, deriv_order, options):
    h = jnp.empty((nri, nri))

    h_tmp = compute_f12_oeints(geom, basis_set, basis_set, xyz_path, deriv_order, options)
    h_tmp = jnp.einsum('pP,qQ,pq->PQ', C_obs, C_obs, h_tmp, optimize='optimal')
    h = h.at[:nobs, :nobs].set(h_tmp) # <O|O>

    h_tmp = compute_f12_oeints(geom, basis_set, cabs_set, xyz_path, deriv_order, options)
    h_tmp = jnp.einsum('pP,qQ,pq->PQ', C_obs, C_cabs, h_tmp, optimize='optimal')
    h = h.at[:nobs, nobs:nri].set(h_tmp) # <O|C>
    h = h.at[nobs:nri, :nobs].set(jnp.transpose(h_tmp)) # <C|O>

    h_tmp = compute_f12_oeints(geom, cabs_set, cabs_set, xyz_path, deriv_order, options)
    h_tmp = jnp.einsum('pP,qQ,pq->PQ', C_cabs, C_cabs, h_tmp, optimize='optimal')
    h = h.at[nobs:nri, nobs:nri].set(h_tmp) # <C|C>

    return h

def form_Fock(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options):
    # OEINTS
    f = form_h(geom, basis_set, cabs_set, C_obs, C_cabs, nobs, nri, xyz_path, deriv_order, options)

    # TEINTS
    G = jnp.empty((nri, nobs, nri, nri))

    G_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "eri", xyz_path, deriv_order, options)
    G_tmp = partial_tei_transformation(G_tmp, C_obs, C_obs[:, :ndocc], C_obs, C_obs)
    G = G.at[:nobs, :ndocc, :nobs, :nobs].set(G_tmp) # <Oo|OO>

    G_tmp = compute_f12_teints(geom, cabs_set, basis_set, basis_set, basis_set, "eri", xyz_path, deriv_order, options)
    G_tmp = partial_tei_transformation(G_tmp, C_cabs, C_obs, C_obs, C_obs)
    G = G.at[nobs:nri, :nobs, :nobs, :nobs].set(G_tmp) # <CO|OO>
    G = G.at[:nobs, :nobs, nobs:nri, :nobs].set(jnp.transpose(G_tmp, (2,1,0,3))) # <OO|CO>
    G = G.at[:nobs, :nobs, :nobs, nobs:nri].set(jnp.transpose(G_tmp, (3,2,1,0))) # <OO|OC>

    G_tmp = compute_f12_teints(geom, cabs_set, basis_set, basis_set, cabs_set, "eri", xyz_path, deriv_order, options)
    G_tmp = partial_tei_transformation(G_tmp, C_cabs, C_obs[:, :ndocc], C_obs, C_cabs)
    G = G.at[nobs:nri, :ndocc, :nobs, nobs:nri].set(G_tmp) # <Co|OC>

    G_tmp = compute_f12_teints(geom, cabs_set, cabs_set, basis_set, basis_set, "eri", xyz_path, deriv_order, options)
    G_tmp = partial_tei_transformation(G_tmp, C_cabs, C_obs[:, :ndocc], C_cabs, C_obs)
    G = G.at[nobs:nri, :ndocc, nobs:nri, :nobs].set(G_tmp) # <Co|CO>

    # Fill Fock Matrix
    f = f.at[:, :].add(2.0 * jnp.einsum('piqi->pq', G[:, :ndocc, :, :ndocc], optimize='optimal'))
    fk = f # Fock Matrix without Exchange
    k =  jnp.einsum('piiq->pq', G[:, :ndocc, :ndocc, :], optimize='optimal')
    f = f.at[:, :].add(-1.0 * k)

    return f, fk, k

# F12 Intermediates
# F12 TEINTS are entered in Chem and returned in Phys

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

    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12", xyz_path, deriv_order, options)
    F_tmp = partial_tei_transformation(F_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :nobs], C_obs[:, :nobs])
    G_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "eri", xyz_path, deriv_order, options)
    G_tmp = partial_tei_transformation(G_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :nobs], C_obs[:, :nobs])
    V = V.at[:, :, :, :].add(-1.0 * jnp.einsum('ijrs,klrs->ijkl', G_tmp, F_tmp, optimize='optimal'))

    return V

def form_X(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, xyz_path, deriv_order, options):

    X = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12_squared", xyz_path, deriv_order, options)
    X = partial_tei_transformation(X, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc])

    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, cabs_set, "f12", xyz_path, deriv_order, options)
    F_tmp = partial_tei_transformation(F_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_cabs)
    X_tmp = -1.0 * jnp.einsum('ijmy,klmy->ijkl', F_tmp, F_tmp, optimize='optimal')
    X = X.at[:, :, :, :].add(X_tmp)
    X = X.at[:, :, :, :].add(f12_transpose(X_tmp))

    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12", xyz_path, deriv_order, options)
    F_tmp = partial_tei_transformation(F_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :nobs], C_obs[:, :nobs])
    X = X.at[:, :, :, :].add(-1.0 * jnp.einsum('ijrs,klrs->ijkl', F_tmp, F_tmp, optimize='optimal'))

    return X

def form_C(geom, basis_set, cabs_set, C_obs, C_cabs, Fock, ndocc, nobs, xyz_path, deriv_order, options):

    C = jnp.empty((ndocc, ndocc, nobs - ndocc, nobs - ndocc))

    F_tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, cabs_set, "f12", xyz_path, deriv_order, options)
    F_tmp = partial_tei_transformation(F_tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, ndocc:nobs], C_cabs)
    C_tmp = jnp.einsum('klay,by->klab', F_tmp, Fock[ndocc:nobs, nobs:], optimize='optimal')

    C = C.at[:, :, :, :].set(C_tmp)
    C = C.at[:, :, :, :].add(f12_transpose(C_tmp))

    return C

def form_B(geom, basis_set, cabs_set, C_obs, C_cabs, Fock, noK, K, ndocc, nobs, nri, xyz_path, deriv_order, options):
    # Term 1
    B = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12_double_commutator", xyz_path, deriv_order, options)
    B = partial_tei_transformation(B, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc])

    # Term 2
    F2 = jnp.empty((ndocc, ndocc, ndocc, nri))
    tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12_squared", xyz_path, deriv_order, options)
    F2 = F2.at[:, :, :, :nobs].set(partial_tei_transformation(tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs)) # <oo|oO>
    tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, cabs_set, "f12_squared", xyz_path, deriv_order, options)
    F2 = F2.at[:, :, :, nobs:].set(partial_tei_transformation(tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs[:, :ndocc], C_cabs)) # <oo|oC>

    tmp = jnp.einsum('lknI,mI->lknm', F2, noK[:ndocc, :])
    B = B.at[:, :, :, :].add(tmp)
    B = B.at[:, :, :, :].add(f12_transpose(tmp))

    # F12 Integral
    F_oo11 = jnp.empty((ndocc, ndocc, nri, nri))
    tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, basis_set, "f12", xyz_path, deriv_order, options)
    F_oo11 = F_oo11.at[:, :, :nobs, :nobs].set(partial_tei_transformation(tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs, C_obs)) # <oo|OO>
    tmp = compute_f12_teints(geom, basis_set, basis_set, basis_set, cabs_set, "f12", xyz_path, deriv_order, options)
    tmp = partial_tei_transformation(tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_obs, C_cabs)
    F_oo11 = F_oo11.at[:, :, :nobs, nobs:].set(tmp) # <oo|OC>
    F_oo11 = F_oo11.at[:, :, nobs:, :nobs].set(f12_transpose(tmp)) # <oo|CO>
    tmp = compute_f12_teints(geom, basis_set, cabs_set, basis_set, cabs_set, "f12", xyz_path, deriv_order, options)
    F_oo11 = F_oo11.at[:, :, nobs:, nobs:].set(partial_tei_transformation(tmp, C_obs[:, :ndocc], C_obs[:, :ndocc], C_cabs, C_cabs)) # <oo|CC>

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
    B = B.at[:, :, :, :].add(tmp)
    B = B.at[:, :, :, :].add(f12_transpose(tmp))

    tmp = jnp.transpose(B, (2,3,0,1))
    B = B.at[:, :, :, :].add(tmp)

    return 0.5 * B
