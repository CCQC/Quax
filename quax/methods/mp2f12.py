import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import fori_loop
import psi4
import sys
jnp.set_printoptions(threshold=sys.maxsize, linewidth=100)

from ..integrals.basis_utils import build_CABS
from .ints import compute_f12_oeints, compute_f12_teints
from .energy_utils import partial_tei_transformation
from .mp2 import restricted_mp2

def restricted_mp2_f12(geom, basis_set, xyz_path, nuclear_charges, charge, options, cabs_set, deriv_order=0):
    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2
    E_mp2, C_obs, eps = restricted_mp2(geom, basis_set, xyz_path, nuclear_charges, charge, options, deriv_order, return_aux_data=True)
    eps_occ, eps_vir = eps[:ndocc], eps[ndocc:]

    print("Running MP2-F12 Computation...")
    C_cabs = build_CABS(geom, basis_set, cabs_set, xyz_path, deriv_order, options)

    nobs = C_obs.shape[0]
    nri = C_obs.shape[0] + C_cabs.shape[1]

    # Fock
    f, fk, k = form_Fock(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)

    # V Intermediate
    V = form_V(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)

    # X Intermediate
    X = form_X(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)

    # C Intermediate
    C = form_C(geom, basis_set, cabs_set, f[ndocc:nobs, nobs:], C_obs, C_cabs, ndocc, nobs, xyz_path, deriv_order, options)

    # B Intermediate
    B = form_B(geom, basis_set, cabs_set, f, k, fk[:ndocc, :], C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)

    D = -1.0 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1, 1) - eps_vir.reshape(-1, 1) - eps_vir)

    G = two_body_mo_computer(geom, "eri", basis_set, basis_set, basis_set, basis_set,\
                             C_obs, C_obs, C_obs, C_obs, xyz_path, deriv_order, options)
    
    # indices = jnp.asarray(jnp.triu_indices(ndocc)).reshape(2,-1).T

    # def loop_energy(idx, f12_corr):
        # i, j = indices[idx]
    
    dE_mp2f12 = 0.0
    for i in range(ndocc):
        for j in range(i, ndocc):
            kd = jax.lax.cond(i == j, lambda: 1.0, lambda: 2.0)

            D_ij = D[i, j, :, :]

            GD_ij = G[i, j, ndocc:, ndocc:] * D_ij
            V_ij = V[i, j, :, :] - jnp.tensordot(C, GD_ij, [(2, 3), (0, 1)])

            V_s = 0.25 * (t_(i, j, i, j) + t_(i, j, j, i)) * kd * (V_ij[i, j] + V_ij[j, i])

            V_t = 0.25 * jax.lax.cond(i != j, lambda: (t_(i, j, i, j) - t_(i, j, j, i))
                                                   * kd * (V_ij[i, j] - V_ij[j, i]), lambda: 0.0)

            CD_ij = jnp.einsum('mnab,ab->mnab', C, D_ij, optimize='optimal')
            B_ij = B - (X * (f[i, i] + f[j, j])) - jnp.tensordot(C, CD_ij, [(2, 3), (2, 3)])

            B_s = 0.125 * (t_(i, j, i, j) + t_(i, j, j, i)) * kd \
                         * (B_ij[i, j, i, j] + B_ij[j, i, i, j]) \
                         * (t_(i, j, i, j) + t_(i, j, j, i)) * kd

            B_t = 0.125 * jax.lax.cond(i != j, lambda: (t_(i, j, i, j) - t_(i, j, j, i)) * kd
                                                     * (B_ij[i, j, i, j] - B_ij[j, i, i, j])
                                                     * (t_(i, j, i, j) - t_(i, j, j, i)) * kd,
                                                     lambda: 0.0)

            E_s = kd * (2.0 * V_s + B_s)         # Singlet Pair Energy
            E_t = 3.0 * kd * (2.0 * V_t + B_t)   # Triplet Pair Energy

            # print(E_s)
            # print(E_t)

            dE_mp2f12 += E_s + E_t

    #     return f12_corr

    # dE_mp2f12 = fori_loop(0, indices.shape[0], loop_energy, 0.0)

    jax.debug.print("OG: {e}", e=dE_mp2f12)

    return dE_mp2f12

# Fixed Amplitude Ansatz
@jax.jit
def t_(p, q, r, s):
    return jnp.select(
        [(p == q) & (p == r) & (p == s), (p == r) & (q == s), (p == s) & (q == r)],
        [0.5, 0.375, 0.125],
        default = jnp.nan
    )

# One-Electron Integrals
def one_body_mo_computer(geom, bs1, bs2, C1, C2, xyz_path, deriv_order, options):
    """
    General one-body MO computer
    that computes the AOs and 
    transforms to MOs
    """
    T, V = compute_f12_oeints(geom, bs1, bs2, xyz_path, deriv_order, options, False)
    AO = T + V
    MO = C1.T @ AO @ C2
    return MO

def form_h(geom, basis_set, cabs_set, C_obs, C_cabs, nobs, nri, xyz_path, deriv_order, options):
    tv = jnp.zeros((nri, nri))

    mo1 = one_body_mo_computer(geom, basis_set, basis_set, C_obs, C_obs, xyz_path, deriv_order, options)
    tv = tv.at[:nobs, :nobs].set(mo1) # <O|O>

    mo2 = one_body_mo_computer(geom, basis_set, cabs_set, C_obs, C_cabs, xyz_path, deriv_order, options)
    tv = tv.at[:nobs, nobs:nri].set(mo2) # <O|C>
    tv = tv.at[nobs:nri, :nobs].set(mo2.T) # <C|O>

    mo3 = one_body_mo_computer(geom, cabs_set, cabs_set, C_cabs, C_cabs, xyz_path, deriv_order, options)
    tv = tv.at[nobs:nri, nobs:nri].set(mo3) # <C|C>

    return tv

# Two-Electron Integrals
def two_body_mo_computer(geom, int_type, bs1, bs2, bs3, bs4, C1, C2, C3, C4, xyz_path, deriv_order, options):
    """
    General two-body MO computer
    that computes the AOs in chem notation,
    returns them in phys notation,
    and then transforms to MOs
    """
    AO = compute_f12_teints(geom, bs1, bs3, bs2, bs4, int_type, xyz_path, deriv_order, options)
    MO = partial_tei_transformation(AO, C1, C3, C2, C4)
    MO = jnp.swapaxes(MO, 1, 2)
    return MO

def form_J(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options):
    eri = jnp.zeros((nri, ndocc, nri, ndocc))
    C_occ = C_obs.at[:, :ndocc].get()

    mo1 = two_body_mo_computer(geom, "eri", basis_set, basis_set, basis_set, basis_set,\
                               C_obs, C_occ, C_obs, C_occ, xyz_path, deriv_order, options)
    eri = eri.at[:nobs, :, :nobs, :].set(mo1) # <Oo|Oo>

    mo2 = two_body_mo_computer(geom, "eri", cabs_set, basis_set, basis_set, basis_set,\
                              C_cabs, C_occ, C_obs, C_occ, xyz_path, deriv_order, options)
    eri = eri.at[nobs:nri, :, :nobs, :].set(mo2) # <Co|Oo>
    eri = eri.at[:nobs, :, nobs:nri, :].set(jnp.transpose(mo2, (2,3,0,1))) # <Oo|Co>

    mo3 = two_body_mo_computer(geom, "eri", cabs_set, basis_set, cabs_set, basis_set,\
                              C_cabs, C_occ, C_cabs, C_occ, xyz_path, deriv_order, options)
    eri = eri.at[nobs:nri, :, nobs:nri, :].set(mo3) # <Co|Co>

    return eri

def form_K(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options):
    eri = jnp.empty((nri, ndocc, ndocc, nri))
    C_occ = C_obs.at[:, :ndocc].get()

    mo1 = two_body_mo_computer(geom, "eri", basis_set, basis_set, basis_set, basis_set,\
                              C_obs, C_occ, C_occ, C_obs, xyz_path, deriv_order, options)
    eri = eri.at[:nobs, :, :, :nobs].set(mo1) # <Oo|oO>

    mo2 = two_body_mo_computer(geom, "eri", cabs_set, basis_set, basis_set, basis_set,\
                              C_cabs, C_occ, C_occ, C_obs, xyz_path, deriv_order, options)
    eri = eri.at[nobs:nri, :, :, :nobs].set(mo2) # <Co|oO>
    eri = eri.at[:nobs, :, :, nobs:nri].set(jnp.transpose(mo2, (3,2,1,0))) # <Oo|oC>

    mo3 = two_body_mo_computer(geom, "eri", cabs_set, basis_set, basis_set, cabs_set,\
                              C_cabs, C_occ, C_occ, C_cabs, xyz_path, deriv_order, options)
    eri = eri.at[nobs:nri, :, :, nobs:nri].set(mo3) # <Co|oC>

    return eri

def form_ooO1(geom, int_type, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options):
    eri = jnp.zeros((ndocc, ndocc, nobs, nri))
    C_occ = C_obs.at[:, :ndocc].get()

    mo1 = two_body_mo_computer(geom, int_type, basis_set, basis_set, basis_set, basis_set,\
                              C_occ, C_occ, C_obs, C_obs, xyz_path, deriv_order, options)
    eri = eri.at[:, :, :, :nobs].set(mo1) # <oo|OO>

    mo2 = two_body_mo_computer(geom, int_type, basis_set, basis_set, basis_set, cabs_set,\
                               C_occ, C_occ, C_obs, C_cabs, xyz_path, deriv_order, options)
    eri = eri.at[:, :, :, nobs:].set(mo2) # <oo|OC>

    return eri

def form_F(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options):
    f12 = jnp.zeros((ndocc, ndocc, nri, nri))
    C_occ = C_obs.at[:, :ndocc].get()

    mo1 = two_body_mo_computer(geom, "f12", basis_set, basis_set, basis_set, basis_set,\
                              C_occ, C_occ, C_obs, C_obs, xyz_path, deriv_order, options)
    f12 = f12.at[:, :, :nobs, :nobs].set(mo1) # <oo|OO>

    mo2 = two_body_mo_computer(geom, "f12", basis_set, basis_set, basis_set, cabs_set,\
                              C_occ, C_occ, C_obs, C_cabs, xyz_path, deriv_order, options)
    f12 = f12.at[:, :, :nobs, nobs:].set(mo2) # <oo|OC>
    f12 = f12.at[:, :, nobs:, :nobs].set(jnp.transpose(mo2, (1,0,3,2))) # <oo|CO>

    mo3 = two_body_mo_computer(geom, "f12", basis_set, basis_set, cabs_set, cabs_set,\
                              C_occ, C_occ, C_cabs, C_cabs, xyz_path, deriv_order, options)
    f12 = f12.at[:, :, nobs:, nobs:].set(mo3) # <oo|CC>

    return f12

def form_F2(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options):
    f12_squared = jnp.zeros((ndocc, ndocc, ndocc, nri))
    C_occ = C_obs.at[:, :ndocc].get()

    mo1 = two_body_mo_computer(geom, "f12_squared", basis_set, basis_set, basis_set, basis_set,\
                              C_occ, C_occ, C_occ, C_obs, xyz_path, deriv_order, options)
    f12_squared = f12_squared.at[:, :, :, :nobs].set(mo1) # <oo|oO>

    mo2 = two_body_mo_computer(geom, "f12_squared", basis_set, basis_set, basis_set, cabs_set,\
                              C_occ, C_occ, C_occ, C_cabs, xyz_path, deriv_order, options)
    f12_squared = f12_squared.at[:, :, :, nobs:].set(mo2) # <oo|oC>

    return f12_squared

# Fock
def form_Fock(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options):

    h = form_h(geom, basis_set, cabs_set, C_obs, C_cabs, nobs, nri, xyz_path, deriv_order, options)
    J = form_J(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)
    K = form_K(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)
    
    # Fock Matrix without Exchange
    fk = h + (2.0 * jnp.einsum('piqi->pq', J, optimize='optimal'))

    # Exchange
    k =  jnp.einsum('piiq->pq', K, optimize='optimal')

    f = fk - k

    return f, fk, k

# F12 Intermediates
def form_V(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options):
    C_occ = C_obs.at[:, :ndocc].get()
    
    FG = two_body_mo_computer(geom, "f12g12", basis_set, basis_set, basis_set, basis_set,\
                              C_occ, C_occ, C_occ, C_occ, xyz_path, deriv_order, options)
    G = form_ooO1(geom, "eri", basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)
    F = form_ooO1(geom, "f12", basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)

    ijkl_1 = jnp.tensordot(G[:, :, :ndocc, nobs:], F[:, :, :ndocc, nobs:], [(2, 3), (2, 3)])
    ijkl_2 = jnp.transpose(ijkl_1, (1,0,3,2))
    ijkl_3 = jnp.tensordot(G[:, :, :nobs, :nobs], F[:, :, :nobs, :nobs], [(2, 3), (2, 3)])

    return FG - ijkl_1 - ijkl_2 - ijkl_3

def form_X(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options):
    C_occ = C_obs.at[:, :ndocc].get()
    
    F2 = two_body_mo_computer(geom, "f12_squared", basis_set, basis_set, basis_set, basis_set,\
                              C_occ, C_occ, C_occ, C_occ, xyz_path, deriv_order, options)
    F = form_ooO1(geom, "f12", basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)

    ijkl_1 = jnp.tensordot(F[:, :, :ndocc, nobs:], F[:, :, :ndocc, nobs:], [(2, 3), (2, 3)])
    ijkl_2 = jnp.transpose(ijkl_1, (1,0,3,2))
    ijkl_3 = jnp.tensordot(F[:, :, :nobs, :nobs], F[:, :, :nobs, :nobs], [(2, 3), (2, 3)])

    return F2 - ijkl_1 - ijkl_2 - ijkl_3

def form_C(geom, basis_set, cabs_set, f_vc, C_obs, C_cabs, ndocc, nobs, xyz_path, deriv_order, options):
    C_occ = C_obs.at[:, :ndocc].get()

    F = two_body_mo_computer(geom, "f12", basis_set, basis_set, basis_set, cabs_set,\
                              C_occ, C_occ, C_obs, C_cabs, xyz_path, deriv_order, options)

    klab = jnp.tensordot(F[:, :, ndocc:nobs, :], f_vc, [(3), (1)])

    return klab + jnp.transpose(klab, (1,0,3,2))

def form_B(geom, basis_set, cabs_set, f, k, fk_o1, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options):
    C_occ = C_obs.at[:, :ndocc].get()
    
    Uf = two_body_mo_computer(geom, "f12_double_commutator", basis_set, basis_set, basis_set, basis_set,\
                              C_occ, C_occ, C_occ, C_occ, xyz_path, deriv_order, options)
    F2 = form_F2(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)
    F = form_F(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)

    # Term 2
    terms = jnp.tensordot(F2, fk_o1, [(3), (1)])

    # Term 3
    terms -= jnp.tensordot(jnp.tensordot(F, k, [(3), (0)]), F, [(2, 3), (2, 3)])

    # Term 4
    terms -= jnp.tensordot(jnp.tensordot(F[:, :, :ndocc, :], f, [(3), (0)]), \
                           F[:, :, :ndocc, :], [(2, 3), (2, 3)])

    # Term 5
    terms += jnp.tensordot(jnp.tensordot(F[:, :, nobs:, :ndocc], f[:ndocc, :ndocc], [(3), (0)]), \
                           F[:, :, nobs:, :ndocc], [(2, 3), (2, 3)])

    # Term 6
    terms -= jnp.tensordot(jnp.tensordot(F[:, :, ndocc:nobs, :nobs], f[:nobs, :nobs], [(3), (0)]), \
                           F[:, :, ndocc:nobs, :nobs], [(2, 3), (2, 3)])

    # Term 7
    terms -= 2.0 * jnp.tensordot(jnp.tensordot(F[:, :, nobs:, :], f[:, :ndocc], [(3), (0)]), \
                                 F[:, :, nobs:, :ndocc], [(2, 3), (2, 3)])

    # Term 8
    terms -= 2.0 * jnp.tensordot(jnp.tensordot(F[:, :, ndocc:nobs, :nobs], f[:nobs, nobs:], [(3), (0)]), \
                                 F[:, :, ndocc:nobs, nobs:], [(2, 3), (2, 3)])

    B_nosymm = Uf + terms + jnp.transpose(terms, (1,0,3,2))

    return 0.5 * (B_nosymm + jnp.transpose(B_nosymm, (2,3,0,1)))
