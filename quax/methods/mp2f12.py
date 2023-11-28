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
    E_mp2, C_obs, eps = restricted_mp2(geom, basis_set, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order, return_aux_data=True)

    print("Running MP2-F12 Computation...")
    C_cabs = build_CABS(geom, basis_set, cabs_set, xyz_path, deriv_order, options)

    nobs = C_obs.shape[0]
    nri = C_cabs.shape[0]
	
    o, v, p, c, A = slice(0, ndocc), slice(ndocc, nobs), slice(0, nobs), slice(nobs, nri), slice(0, nri)

    eps_occ, eps_vir = eps[o], eps[v]

    # Fock
    h = form_h(geom, basis_set, cabs_set, C_obs, C_cabs, nobs, nri, xyz_path, deriv_order, options)
    G = form_G(geom, basis_set, cabs_set, C_obs, C_cabs, nobs, nri, xyz_path, deriv_order, options)
    f, fk, k = form_Fock(h, (G[A, o, A, o], G[A, o, o, A]))

    # V Intermediate
    FG = form_FG(geom, basis_set, C_obs, ndocc, xyz_path, deriv_order, options)
    F = form_F(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)
    V = form_V(FG, (F[o, o, o, c], F[o, o, p, p]), (G[o, o, o, c], G[o, o, p, p]))

    # X Intermediate
    F2 = form_F2(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options)
    X = form_X(F2[o, o, o, o], (F[o, o, o, c], F[o, o, p, p]))

    # C Intermediate
    C = form_C(F[o, o, v, c], f[v, c])

    # B Intermediate
    Uf = form_Uf(geom, basis_set, C_obs, ndocc, xyz_path, deriv_order, options)
    B = form_B(Uf, F2, (F, F[o, o, o, A], F[o, o, c, o], F[o, o, v, p], F[o, o, c, A], F[o, o, v, c]),\
               (f, f[o, o], f[p, p], f[A, o], f[p, c]), fk[o, A], k)

    D = -1.0 * jnp.reciprocal(eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1, 1) - eps_vir.reshape(-1, 1) - eps_vir)
    Dv = slice(0, nobs - ndocc)
    
    # indices = jnp.asarray(jnp.triu_indices(ndocc)).reshape(2,-1).T

    # def loop_energy(idx, f12_corr):
        # i, j = indices[idx]
    
    dE_mp2f12 = 0.0
    for i in range(ndocc):
        for j in range(i, ndocc):
            kd = jax.lax.cond(i == j, lambda: 1.0, lambda: 2.0)

            D_ij = D[i, j, Dv, Dv]

            V_ij = V[i, j, o, o]
            GD_ij = G[i, j, v, v] * D_ij
            V_ij -= jnp.tensordot(C, GD_ij, [(2, 3), (0, 1)])
            print(V_ij)

            V_s = 0.25 * (t_(i, j, i, j) + t_(i, j, j, i)) * kd * (V_ij[i, j] + V_ij[j, i])

            V_t = 0.25 * jax.lax.cond(i != j, lambda: (t_(i, j, i, j) - t_(i, j, j, i))
                                                   * kd * (V_ij[i, j] - V_ij[j, i]), lambda: 0.0)

            B_ij = B - (X * (f[i, i] + f[j, j]))
            CD_ij = jnp.einsum('mnab,ab->mnab', C, D_ij, optimize='optimal')
            B_ij -= jnp.tensordot(C, CD_ij, [(2, 3), (2, 3)])
            print(B_ij)

            B_s = 0.125 * (t_(i, j, i, j) + t_(i, j, j, i)) * kd \
                         * (B_ij[i, j, i, j] + B_ij[j, i, i, j]) \
                         * (t_(i, j, i, j) + t_(i, j, j, i)) * kd

            B_t = 0.125 * jax.lax.cond(i != j, lambda: (t_(i, j, i, j) - t_(i, j, j, i)) * kd
                                                     * (B_ij[i, j, i, j] - B_ij[j, i, i, j])
                                                     * (t_(i, j, i, j) - t_(i, j, j, i)) * kd,
                                                     lambda: 0.0)

            dE_mp2f12 += kd * (2.0 * V_s + B_s)         # Singlet Pair Energy
            dE_mp2f12 += 3.0 * kd * (2.0 * V_t + B_t)   # Triplet Pair Energy

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
    MO = jnp.dot(C1.T, jnp.dot(AO, C2))
    return MO

def form_h(geom, basis_set, cabs_set, C_obs, C_cabs, nobs, nri, xyz_path, deriv_order, options):
    tv = jnp.empty((nri, nri))

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
    AO = jnp.transpose(AO, (0,2,1,3))
    MO = partial_tei_transformation(AO, C1, C2, C3, C4)
    return MO

def form_G(geom, basis_set, cabs_set, C_obs, C_cabs, nobs, nri, xyz_path, deriv_order, options):
    eri = jnp.empty((nri, nobs, nri, nri))

    mo1 = two_body_mo_computer(geom, "eri", basis_set, basis_set, basis_set, basis_set,\
                              C_obs, C_obs, C_obs, C_obs, xyz_path, deriv_order, options)
    eri = eri.at[:nobs, :nobs, :nobs, :nobs].set(mo1) # <OO|OO>

    mo2 = two_body_mo_computer(geom, "eri", cabs_set, basis_set, basis_set, basis_set,\
                              C_cabs, C_obs, C_obs, C_obs, xyz_path, deriv_order, options)
    eri = eri.at[nobs:nri, :nobs, :nobs, :nobs].set(mo2) # <CO|OO>
    eri = eri.at[:nobs, :nobs, nobs:nri, :nobs].set(jnp.transpose(mo2, (2,3,0,1))) # <OO|CO>
    eri = eri.at[:nobs, :nobs, :nobs, nobs:nri].set(jnp.transpose(mo2, (3,2,1,0))) # <OO|OC>

    mo3 = two_body_mo_computer(geom, "eri", cabs_set, basis_set, basis_set, cabs_set,\
                              C_cabs, C_obs, C_obs, C_cabs, xyz_path, deriv_order, options)
    eri = eri.at[nobs:nri, :nobs, :nobs, nobs:nri].set(mo3) # <CO|OC>

    mo4 = two_body_mo_computer(geom, "eri", cabs_set, basis_set, cabs_set, basis_set,\
                              C_cabs, C_obs, C_cabs, C_obs, xyz_path, deriv_order, options)
    eri = eri.at[nobs:nri, :nobs, nobs:nri, :nobs].set(mo4) # <CO|CO>

    return eri

def form_F(geom, basis_set, cabs_set, C_obs, C_cabs, ndocc, nobs, nri, xyz_path, deriv_order, options):
    f12 = jnp.empty((ndocc, ndocc, nri, nri))
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
    f12_squared = jnp.empty((ndocc, ndocc, ndocc, nri))
    C_occ = C_obs.at[:, :ndocc].get()

    mo1 = two_body_mo_computer(geom, "f12_squared", basis_set, basis_set, basis_set, basis_set,\
                              C_occ, C_occ, C_occ, C_obs, xyz_path, deriv_order, options)
    f12_squared = f12_squared.at[:, :, :, :nobs].set(mo1) # <oo|oO>

    mo2 = two_body_mo_computer(geom, "f12_squared", basis_set, basis_set, basis_set, cabs_set,\
                              C_occ, C_occ, C_occ, C_cabs, xyz_path, deriv_order, options)
    f12_squared = f12_squared.at[:, :, :, nobs:].set(mo2) # <oo|oC>

    return f12_squared

def form_FG(geom, basis_set, C_obs, ndocc, xyz_path, deriv_order, options):
    C_occ = C_obs.at[:, :ndocc].get()

    f12g12 = two_body_mo_computer(geom, "f12g12", basis_set, basis_set, basis_set, basis_set,\
                                  C_occ, C_occ, C_occ, C_occ, xyz_path, deriv_order, options)
    return f12g12

def form_Uf(geom, basis_set, C_obs, ndocc, xyz_path, deriv_order, options):
    C_occ = C_obs.at[:, :ndocc].get()

    f12_double_commutator = two_body_mo_computer(geom, "f12_double_commutator",\
                                    basis_set, basis_set, basis_set, basis_set,\
                                    C_occ, C_occ, C_occ, C_occ, xyz_path, deriv_order, options)
    return f12_double_commutator

# Fock
def form_Fock(h, Fock_G):

    G_1o1o, G_1oo1 = Fock_G
    
    # Fock Matrix without Exchange
    fk = h + 2.0 * jnp.einsum('piqi->pq', G_1o1o, optimize='optimal')

    # Exchange
    k =  jnp.einsum('piiq->pq', G_1oo1, optimize='optimal')

    f = fk - k

    return f, fk, k

# F12 Intermediates
def form_V(FG, VX_F, V_G):
    
    G_oooc, G_oopq = V_G
    F_oooc, F_oopq = VX_F

    ijkl_1 = jnp.tensordot(G_oooc, F_oooc, [(2, 3), (2, 3)])
    ijkl_2 = jnp.transpose(ijkl_1, (1,0,3,2))
    ijkl_3 = jnp.tensordot(G_oopq, F_oopq, [(2, 3), (2, 3)])

    return FG - ijkl_1 - ijkl_2 - ijkl_3

def form_X(F2_oooo, VX_F):
    
    F_oooc, F_oopq = VX_F

    ijkl_1 = jnp.tensordot(F_oooc, F_oooc, [(2, 3), (2, 3)])
    ijkl_2 = jnp.transpose(ijkl_1, (1,0,3,2))
    ijkl_3 = jnp.tensordot(F_oopq, F_oopq, [(2, 3), (2, 3)])

    return F2_oooo - ijkl_1 - ijkl_2 - ijkl_3

def form_C(F_oovc, f_vc):

    klab = jnp.tensordot(F_oovc, f_vc, [(3), (1)])

    return klab + jnp.transpose(klab, (1,0,3,2))

def form_B(Uf, F2, B_F, B_f, fk_o1, k):

    F, F_ooo1, F_ooco, F_oovq, F_ooc1, F_oovc = B_F
    f, f_oo, f_pq, f_1o, f_pc = B_f

    # Term 2
    terms = jnp.tensordot(F2, fk_o1, [(3), (1)])

    # Term 3
    terms -= jnp.tensordot(jnp.tensordot(F, k, [(3), (0)]), F, [(2, 3), (2, 3)])

    # Term 4
    terms -= jnp.tensordot(jnp.tensordot(F_ooo1, f, [(3), (0)]), F_ooo1, [(2, 3), (2, 3)])

    # Term 5
    terms += jnp.tensordot(jnp.tensordot(F_ooco, f_oo, [(3), (0)]), F_ooco, [(2, 3), (2, 3)])

    # Term 6
    terms -= jnp.tensordot(jnp.tensordot(F_oovq, f_pq, [(3), (0)]), F_oovq, [(2, 3), (2, 3)])

    # Term 7
    terms -= 2.0 * jnp.tensordot(jnp.tensordot(F_ooc1, f_1o, [(3), (0)]), F_ooco, [(2, 3), (2, 3)])

    # Term 8
    terms -= 2.0 * jnp.tensordot(jnp.tensordot(F_oovq, f_pc, [(3), (0)]), F_oovc, [(2, 3), (2, 3)])


    B_nosymm = Uf + terms + jnp.transpose(terms, (1,0,3,2))

    return 0.5 * (B_nosymm + jnp.transpose(B_nosymm, (2,3,0,1)))
