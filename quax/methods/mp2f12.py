import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import fori_loop, cond

from .basis_utils import build_CABS
from .ints import compute_f12_oeints, compute_f12_teints, compute_dipole_ints, compute_quadrupole_ints
from .energy_utils import partial_tei_transformation, cartesian_product
from .mp2 import restricted_mp2

def restricted_mp2_f12(*args, options, deriv_order=0):
    if options['electric_field'] == 1:
        efield, geom, basis_set, cabs_set, nelectrons, nfrzn, nuclear_charges, xyz_path = args
        fields = (efield,)
        mp2_args = efield, geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path
    elif options['electric_field'] == 2:
        efield_grad, efield, geom, basis_set, cabs_set, nelectrons, nfrzn, nuclear_charges, xyz_path = args
        fields = (efield_grad, efield)
        mp2_args = efield_grad, efield, geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path
    else:
        geom, basis_set, cabs_set, nelectrons, nfrzn, nuclear_charges, xyz_path = args
        fields = None
        mp2_args = (geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)

    E_mp2, C_obs, eps, G = restricted_mp2(*mp2_args, options=options, deriv_order=deriv_order, return_aux_data=True)
    ndocc = nelectrons // 2
    ncore = nfrzn // 2
    eps_occ, eps_vir = eps[:ndocc], eps[ndocc:]

    print("Running MP2-F12 Computation...")
    C_cabs = build_CABS(geom, basis_set, cabs_set, xyz_path, deriv_order, options)
    C_mats = (C_obs[:, :ndocc], C_obs, C_cabs) # C_occ, C_obs, C_cabs

    nobs = C_obs.shape[0]
    spaces = (ndocc, nobs, C_cabs.shape[0]) # ndocc, nobs, nri

    # Fock
    f, fk, k = form_Fock(geom, basis_set, cabs_set, C_mats, spaces, fields, xyz_path, deriv_order, options)

    # V Intermediate
    V = form_V(geom, basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options)\

    # X Intermediate
    X = form_X(geom, basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options)

    # C Intermediate
    C = form_C(geom, basis_set, cabs_set, f[nobs:, ndocc:nobs], C_mats, spaces, xyz_path, deriv_order, options)

    # B Intermediate
    B = form_B(geom, basis_set, cabs_set, f, k, fk[:ndocc, :], C_mats, spaces, xyz_path, deriv_order, options)

    D = -1.0 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1, 1) - eps_vir.reshape(-1, 1) - eps_vir)
    G = jnp.swapaxes(G, 1, 2)

    indices = jnp.asarray(jnp.triu_indices(ndocc)).reshape(2,-1).T

    def loop_energy(idx, f12_corr):
        i, j = indices[idx]
        kd = cond(i == j, lambda: 1.0, lambda: 2.0)

        D_ij = D[i, j, :, :]

        GD_ij = jnp.einsum('ab,ab->ab', G[i - 1, j - 1, :, :], D_ij, optimize='optimal')
        V_ij = V[i, j, :, :] - jnp.einsum('klab,ab->kl', C, GD_ij, optimize='optimal')

        V_s = 0.25 * (t_(i, j, i, j) + t_(i, j, j, i)) * kd * (V_ij[i, j] + V_ij[j, i])

        V_t = 0.25 * cond(i != j, lambda: (t_(i, j, i, j) - t_(i, j, j, i))
                                               * kd * (V_ij[i, j] - V_ij[j, i]), lambda: 0.0)

        CD_ij = jnp.einsum('mnab,ab->mnab', C, D_ij, optimize='optimal')
        B_ij = B - (X * (f[i, i] + f[j, j])) - jnp.einsum('klab,mnab->klmn', C, CD_ij, optimize='optimal')

        B_s = 0.125 * (t_(i, j, i, j) + t_(i, j, j, i)) * kd \
                     * (B_ij[i, j, i, j] + B_ij[j, i, i, j]) \
                     * (t_(i, j, i, j) + t_(i, j, j, i)) * kd

        B_t = 0.125 * cond(i != j, lambda: (t_(i, j, i, j) - t_(i, j, j, i)) * kd
                                                 * (B_ij[i, j, i, j] - B_ij[j, i, i, j])
                                                 * (t_(i, j, i, j) - t_(i, j, j, i)) * kd,
                                                 lambda: 0.0)

        f12_corr += kd * (2.0 * V_s + B_s)         # Singlet Pair Energy
        f12_corr += 3.0 * kd * (2.0 * V_t + B_t)   # Triplet Pair Energy

        return f12_corr

    start = ndocc if ncore > 0 else 0
    dE_mp2f12 = fori_loop(start, indices.shape[0], loop_energy, 0.0)

    E_s = cabs_singles(f, spaces)

    return E_mp2 + dE_mp2f12 + E_s

# CABS Singles
def cabs_singles(f, spaces):
    ndocc, _, nri = spaces
    all_vir = nri - ndocc

    e_ij, C_ij = jnp.linalg.eigh(f[:ndocc, :ndocc])
    e_AB, C_AB = jnp.linalg.eigh(f[ndocc:, ndocc:])

    f_iA = C_ij.T @ f[:ndocc, ndocc:] @ C_AB

    indices = cartesian_product(jnp.arange(ndocc), jnp.arange(all_vir))

    def loop_singles(idx, singles):
        i, A = indices[idx]
        singles += 2 * f_iA[i, A]**2 / (e_ij[i] - e_AB[A])
        return singles
    E_s = fori_loop(0, indices.shape[0], loop_singles, 0.0)

    return E_s

# Fixed Amplitude Ansatz
@jax.jit
def t_(p, q, r, s):
    return jnp.select(
        [(p == q) & (p == r) & (p == s), (p == r) & (q == s), (p == s) & (q == r)],
        [0.5, 0.375, 0.125],
        default = jnp.nan
    )

# One-Electron Integrals
def one_body_mo_computer(geom, bs1, bs2, C1, C2, fields, xyz_path, deriv_order, options):
    """
    General one-body MO computer
    that computes the AOs and 
    transforms to MOs
    """
    T, V = compute_f12_oeints(geom, bs1, bs2, xyz_path, deriv_order, options, False)
    AO = T + V

    if options['electric_field'] == 1:
        Mu_XYZ = compute_dipole_ints(geom, bs1, bs2, xyz_path, deriv_order, options)
        AO += jnp.einsum('x,xij->ij', fields[0], Mu_XYZ, optimize = 'optimal')
    elif options['electric_field'] == 2:
        Mu_Th = compute_quadrupole_ints(geom, bs1, bs2, xyz_path, deriv_order, options)
        AO += jnp.einsum('x,xij->ij', fields[0], Mu_Th[:3, :, :], optimize = 'optimal')
        AO += jnp.einsum('x,xij->ij', fields[1][jnp.triu_indices(3)], Mu_Th[3:, :, :], optimize = 'optimal')

    MO = C1.T @ AO @ C2
    return MO

def form_h(geom, basis_set, cabs_set, C_mats, spaces, fields, xyz_path, deriv_order, options):
    _, nobs, nri = spaces
    _, C_obs, C_cabs = C_mats

    tv = jnp.zeros((nri, nri))

    mo1 = one_body_mo_computer(geom, basis_set, basis_set, C_obs, C_obs, fields, xyz_path, deriv_order, options)
    tv = tv.at[:nobs, :nobs].set(mo1) # <O|O>

    mo2 = one_body_mo_computer(geom, basis_set, cabs_set, C_obs, C_cabs, fields, xyz_path, deriv_order, options)
    tv = tv.at[:nobs, nobs:nri].set(mo2) # <O|C>
    tv = tv.at[nobs:nri, :nobs].set(mo2.T) # <C|O>

    mo3 = one_body_mo_computer(geom, cabs_set, cabs_set, C_cabs, C_cabs, fields, xyz_path, deriv_order, options)
    tv = tv.at[nobs:nri, nobs:nri].set(mo3) # <C|C>

    return tv

# Two-Electron Integrals
def two_body_mo_computer(geom, int_type, bs1, bs2, bs3, bs4, C1, C2, C3, C4, xyz_path, deriv_order, options):
    """
    General two-body MO computer
    that computes the AOs in chem notation,
    then transforms to MOs,
    and returns the MOs in phys notation
    """
    AO = compute_f12_teints(geom, bs1, bs3, bs2, bs4, int_type, xyz_path, deriv_order, options)
    MO = partial_tei_transformation(AO, C1, C3, C2, C4)
    MO = jnp.swapaxes(MO, 1, 2)
    return MO

def form_J(geom, basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options):
    ndocc, nobs, nri = spaces
    C_occ, C_obs, C_cabs = C_mats

    eri = jnp.zeros((nri, ndocc, nri, ndocc))

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

def form_K(geom, basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options):
    ndocc, nobs, nri = spaces
    C_occ, C_obs, C_cabs = C_mats

    eri = jnp.empty((nri, ndocc, ndocc, nri))

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

def form_ooO1(geom, int_type, basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options):
    ndocc, nobs, nri = spaces
    C_occ, C_obs, C_cabs = C_mats

    eri = jnp.zeros((ndocc, ndocc, nobs, nri))

    mo1 = two_body_mo_computer(geom, int_type, basis_set, basis_set, basis_set, basis_set,\
                              C_occ, C_occ, C_obs, C_obs, xyz_path, deriv_order, options)
    eri = eri.at[:, :, :, :nobs].set(mo1) # <oo|OO>

    mo2 = two_body_mo_computer(geom, int_type, basis_set, basis_set, basis_set, cabs_set,\
                               C_occ, C_occ, C_obs, C_cabs, xyz_path, deriv_order, options)
    eri = eri.at[:, :, :, nobs:].set(mo2) # <oo|OC>

    return eri

def form_F(geom, basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options):
    ndocc, nobs, nri = spaces
    C_occ, C_obs, C_cabs = C_mats

    f12 = jnp.zeros((ndocc, ndocc, nri, nri))

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

def form_F2(geom, basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options):
    ndocc, nobs, nri = spaces
    C_occ, C_obs, C_cabs = C_mats

    f12_squared = jnp.zeros((ndocc, ndocc, ndocc, nri))

    mo1 = two_body_mo_computer(geom, "f12_squared", basis_set, basis_set, basis_set, basis_set,\
                              C_occ, C_occ, C_occ, C_obs, xyz_path, deriv_order, options)
    f12_squared = f12_squared.at[:, :, :, :nobs].set(mo1) # <oo|oO>

    mo2 = two_body_mo_computer(geom, "f12_squared", basis_set, basis_set, basis_set, cabs_set,\
                              C_occ, C_occ, C_occ, C_cabs, xyz_path, deriv_order, options)
    f12_squared = f12_squared.at[:, :, :, nobs:].set(mo2) # <oo|oC>

    return f12_squared

# Fock
def form_Fock(geom, basis_set, cabs_set, C_mats, spaces, fields, xyz_path, deriv_order, options):

    fk = form_h(geom, basis_set, cabs_set, C_mats, spaces, fields, xyz_path, deriv_order, options)
    J = form_J(geom, basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options)
    K = form_K(geom, basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options)
    
    # Fock Matrix without Exchange
    fk += 2.0 * jnp.einsum('piqi->pq', J, optimize='optimal')

    # Exchange
    k =  jnp.einsum('piiq->pq', K, optimize='optimal')

    f = fk - k

    return f, fk, k

# F12 Intermediates
def form_V(geom, basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options):
    C_occ, _, _ = C_mats
    ndocc, nobs, _ = spaces
    
    FG = two_body_mo_computer(geom, "f12g12", basis_set, basis_set, basis_set, basis_set,\
                              C_occ, C_occ, C_occ, C_occ, xyz_path, deriv_order, options)
    G = form_ooO1(geom, "eri", basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options)
    F = form_ooO1(geom, "f12", basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options)

    ijkl_1 = jnp.einsum('ijmy,klmy->ijkl', G[:, :, :ndocc, nobs:], F[:, :, :ndocc, nobs:], optimize='optimal')
    ijkl_2 = jnp.transpose(ijkl_1, (1,0,3,2)) # ijxn,klxn->ijkl
    ijkl_3 = jnp.einsum('ijrs,klrs->ijkl', G[:, :, :nobs, :nobs], F[:, :, :nobs, :nobs], optimize='optimal')

    return FG - ijkl_1 - ijkl_2 - ijkl_3

def form_X(geom, basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options):
    C_occ, _, _ = C_mats
    ndocc, nobs, _ = spaces
    
    F2 = two_body_mo_computer(geom, "f12_squared", basis_set, basis_set, basis_set, basis_set,\
                              C_occ, C_occ, C_occ, C_occ, xyz_path, deriv_order, options)
    F = form_ooO1(geom, "f12", basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options)

    ijkl_1 = jnp.einsum('ijmy,klmy->ijkl', F[:, :, :ndocc, nobs:], F[:, :, :ndocc, nobs:], optimize='optimal')
    ijkl_2 = jnp.transpose(ijkl_1, (1,0,3,2)) # ijxn,klxn->ijkl
    ijkl_3 = jnp.einsum('ijrs,klrs->ijkl', F[:, :, :nobs, :nobs], F[:, :, :nobs, :nobs], optimize='optimal')

    return F2 - ijkl_1 - ijkl_2 - ijkl_3

def form_C(geom, basis_set, cabs_set, f_cv, C_mats, spaces, xyz_path, deriv_order, options):
    C_occ, C_obs, C_cabs = C_mats
    ndocc, nobs, _ = spaces

    F = two_body_mo_computer(geom, "f12", basis_set, basis_set, basis_set, cabs_set,\
                              C_occ, C_occ, C_obs, C_cabs, xyz_path, deriv_order, options)

    klab = jnp.einsum('klax,xb->klab', F[:, :, ndocc:nobs, :], f_cv, optimize='optimal')

    return klab + jnp.transpose(klab, (1,0,3,2))

def form_B(geom, basis_set, cabs_set, f, k, fk_o1, C_mats, spaces, xyz_path, deriv_order, options):
    C_occ, C_obs, C_cabs = C_mats
    ndocc, nobs, _ = spaces
    
    Uf = two_body_mo_computer(geom, "f12_double_commutator", basis_set, basis_set, basis_set, basis_set,\
                              C_occ, C_occ, C_occ, C_occ, xyz_path, deriv_order, options)
    F2 = form_F2(geom, basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options)
    F = form_F(geom, basis_set, cabs_set, C_mats, spaces, xyz_path, deriv_order, options)

    # Term 2
    terms = jnp.einsum('nmlP,kP->nmlk', F2, fk_o1)

    # Term 3
    terms -= jnp.einsum('nmQP,PR,lkQR->nmlk', F, k, F, optimize='optimal')

    # Term 4
    terms -= jnp.einsum('nmjP,PR,lkjR->nmlk', F[:, :, :ndocc, :], f, F[:, :, :ndocc, :], optimize='optimal')

    # Term 5
    terms += jnp.einsum('nmyi,ij,lkyj->nmlk', F[:, :, nobs:, :ndocc], f[:ndocc, :ndocc],\
                                              F[:, :, nobs:, :ndocc], optimize='optimal')

    # Term 6
    terms -= jnp.einsum('nmbp,pr,lkbr->nmlk', F[:, :, ndocc:nobs, :nobs], f[:nobs, :nobs],\
                                              F[:, :, ndocc:nobs, :nobs], optimize='optimal')

    # Term 7
    terms -= 2.0 * jnp.einsum('nmyi,iP,lkyP->nmlk', F[:, :, nobs:, :], f[:, :ndocc],\
                                                    F[:, :, nobs:, :ndocc], optimize='optimal')

    # Term 8
    terms -= 2.0 * jnp.einsum('nmbx,xq,lkbq->nmlk', F[:, :, ndocc:nobs, :nobs], f[:nobs, nobs:],\
                                                    F[:, :, ndocc:nobs, nobs:], optimize='optimal')

    B_nosymm = Uf + terms + jnp.transpose(terms, (1,0,3,2)) # nmlk->mnkl

    return 0.5 * (B_nosymm + jnp.transpose(B_nosymm, (2,3,0,1))) # mnkl + klmn