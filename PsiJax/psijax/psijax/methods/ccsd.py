import jax 
from jax.config import config; config.update("jax_enable_x64", True)
config.enable_omnistaging()
import jax.numpy as jnp
from jax.experimental import loops
import psi4

from .energy_utils import nuclear_repulsion, partial_tei_transformation, tei_transformation
from .hartree_fock import restricted_hartree_fock

def rccsd(geom, basis_name, xyz_path, nuclear_charges, charge, return_aux_data=False):
    # Do HF
    E_scf, C, eps, V = restricted_hartree_fock(geom, basis_name, xyz_path, nuclear_charges, charge, SCF_MAX_ITER=50, return_aux_data=True)

    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2
    nbf = V.shape[0]
    nvir = nbf - ndocc

    o = slice(0, ndocc)
    v = slice(ndocc, nbf)

    # Transform TEI's to MO basis
    V = tei_transformation(V,C)
    fock_Od = eps[o]
    fock_Vd = eps[v]

    # Save slices of two-electron repulsion integral
    V = jnp.swapaxes(V, 1,2)
    V = (V[o,o,o,o], V[o,o,o,v], V[o,o,v,v], V[o,v,o,v], V[o,v,v,v], V[v,v,v,v])

    # Oribital energy denominators 
    D = 1.0 / (fock_Od.reshape(-1,1,1,1) + fock_Od.reshape(-1,1,1) - fock_Vd.reshape(-1,1) - fock_Vd)
    d = 1.0 / (fock_Od.reshape(-1,1) - fock_Vd)

    # Initial Amplitudes
    T1 = jnp.zeros((ndocc,nvir))
    T2 = D*V[2]

    CC_MAX_ITER = 30
    iteration = 0
    E_ccsd = 1.0
    E_old = 0.0
    while abs(E_ccsd - E_old)  > 1e-9:
        E_old = E_ccsd * 1
        T1, T2 = rccsd_iter(T1, T2, V, d, D, ndocc, nvir)
        E_ccsd = rccsd_energy(T1,T2,V[2])
        iteration += 1
        if iteration == CC_MAX_ITER:
            break

    print(iteration, " CCSD iterations performed")
    #print("CCSD Correlation Energy:   ", E_ccsd)
    #print("CCSD Total Energy:         ", E_ccsd + E_scf)
    if return_aux_data:
        return E_scf + E_ccsd, T1, T2, V, fock_Od, fock_Vd
    else:
        return E_scf + E_ccsd

@jax.jit
def rccsd_energy(T1, T2, Voovv):
    E_ccsd = 0.0
    E_ccsd -= jnp.einsum('lc, kd, klcd -> ', T1, T1, Voovv, optimize = 'optimal')
    E_ccsd -= jnp.einsum('lkcd, klcd -> ', T2, Voovv, optimize = 'optimal')
    E_ccsd += 2.0*jnp.einsum('klcd, klcd -> ', T2, Voovv, optimize = 'optimal')
    E_ccsd += 2.0*jnp.einsum('lc, kd, lkcd -> ', T1, T1, Voovv, optimize = 'optimal')
    return E_ccsd

@jax.jit
def rccsd_iter(T1, T2, V, d, D, ndocc, nvir):
    Voooo, Vooov, Voovv, Vovov, Vovvv, Vvvvv = V

    newT1 = jnp.zeros(T1.shape)
    newT2 = jnp.zeros(T2.shape)

    # T1 equation
    newT1 -= jnp.einsum('kc, icka -> ia', T1, Vovov, optimize = 'optimal')
    newT1 += 2.0*jnp.einsum('kc, kica -> ia', T1, Voovv, optimize = 'optimal')
    newT1 -= jnp.einsum('kicd, kadc -> ia', T2, Vovvv, optimize = 'optimal')
    newT1 += 2.0*jnp.einsum('ikcd, kadc -> ia', T2, Vovvv, optimize = 'optimal')
    newT1 += -2.0*jnp.einsum('klac, klic -> ia', T2, Vooov, optimize = 'optimal')
    newT1 += jnp.einsum('lkac, klic -> ia', T2, Vooov, optimize = 'optimal')
    newT1 += -2.0*jnp.einsum('kc, la, lkic -> ia', T1, T1, Vooov, optimize = 'optimal')
    newT1 -= jnp.einsum('kc, id, kadc -> ia', T1, T1, Vovvv, optimize = 'optimal')
    newT1 += 2.0*jnp.einsum('kc, id, kacd -> ia', T1, T1, Vovvv, optimize = 'optimal')
    newT1 += jnp.einsum('kc, la, klic -> ia', T1, T1, Vooov, optimize = 'optimal')
    newT1 += -2.0*jnp.einsum('kc, ilad, lkcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += -2.0*jnp.einsum('kc, liad, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += jnp.einsum('kc, liad, lkcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += -2.0*jnp.einsum('ic, lkad, lkcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += jnp.einsum('ic, lkad, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += -2.0*jnp.einsum('la, ikdc, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += jnp.einsum('la, ikcd, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += jnp.einsum('kc, id, la, lkcd -> ia', T1, T1, T1, Voovv, optimize = 'optimal')
    newT1 += -2.0*jnp.einsum('kc, id, la, klcd -> ia', T1, T1, T1, Voovv, optimize = 'optimal')
    newT1 += 4.0*jnp.einsum('kc, ilad, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')

    # T2 equation
    newT2 += Voovv
    newT2 += jnp.einsum('ic, jd, cdab -> ijab', T1, T1, Vvvvv, optimize = 'optimal')
    newT2 += jnp.einsum('ijcd, cdab -> ijab', T2, Vvvvv, optimize = 'optimal')
    newT2 += jnp.einsum('ka, lb, ijkl -> ijab', T1, T1, Voooo, optimize = 'optimal')
    newT2 += jnp.einsum('klab, ijkl -> ijab', T2, Voooo, optimize = 'optimal')
    newT2 -= jnp.einsum('ic, jd, ka, kbcd -> ijab', T1, T1, T1, Vovvv, optimize = 'optimal')
    newT2 -= jnp.einsum('ic, jd, kb, kadc -> ijab', T1, T1, T1, Vovvv, optimize = 'optimal')
    newT2 += jnp.einsum('ic, ka, lb, lkjc -> ijab', T1, T1, T1, Vooov, optimize = 'optimal')
    newT2 += jnp.einsum('jc, ka, lb, klic -> ijab', T1, T1, T1, Vooov, optimize = 'optimal')
    newT2 += jnp.einsum('klac, ijdb, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*jnp.einsum('ikac, ljbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*jnp.einsum('lkac, ijdb, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('kiac, ljdb, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('ikac, ljbd, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*jnp.einsum('ikac, jlbd, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('kiac, ljbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*jnp.einsum('kiac, jlbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('ijac, lkbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*jnp.einsum('ijac, klbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('kjac, ildb, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += 4.0*jnp.einsum('ikac, jlbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('ijdc, lkab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('ic, jd, ka, lb, klcd -> ijab', T1, T1, T1, T1, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('ic, jd, lkab, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('ka, lb, ijdc, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO = -jnp.einsum('kb, jika -> ijab', T1, Vooov, optimize = 'optimal')
    P_OVVO += jnp.einsum('jc, icab -> ijab', T1, Vovvv, optimize = 'optimal')
    P_OVVO -= jnp.einsum('kiac, kjcb -> ijab', T2, Voovv, optimize = 'optimal')
    P_OVVO -= jnp.einsum('ic, ka, kjcb -> ijab', T1, T1, Voovv, optimize = 'optimal')
    P_OVVO -= jnp.einsum('ic, kb, jcka -> ijab', T1, T1, Vovov, optimize = 'optimal')
    P_OVVO += 2.0*jnp.einsum('ikac, kjcb -> ijab', T2, Voovv, optimize = 'optimal')
    P_OVVO -= jnp.einsum('ikac, jckb -> ijab', T2, Vovov, optimize = 'optimal')
    P_OVVO -= jnp.einsum('kjac, ickb -> ijab', T2, Vovov, optimize = 'optimal')
    P_OVVO -= 2.0*jnp.einsum('lb, ikac, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += jnp.einsum('lb, kiac, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO -= jnp.einsum('jc, ikdb, kacd -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO -= jnp.einsum('jc, kiad, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO -= jnp.einsum('jc, ikad, kbcd -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += jnp.einsum('jc, lkab, lkic -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += jnp.einsum('lb, ikac, kljc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO -= jnp.einsum('ka, ijdc, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += jnp.einsum('ka, ilcb, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += 2.0*jnp.einsum('jc, ikad, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO -= jnp.einsum('kc, ijad, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += 2.0*jnp.einsum('kc, ijad, kbcd -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += jnp.einsum('kc, ilab, kljc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO -= 2.0*jnp.einsum('kc, ilab, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += jnp.einsum('jkcd, ilab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    P_OVVO -= 2.0*jnp.einsum('kc, jd, ilab, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += jnp.einsum('kc, jd, ilab, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += -2.0*jnp.einsum('kc, la, ijdb, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += jnp.einsum('kc, la, ijdb, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += jnp.einsum('ic, ka, ljbd, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += -2.0*jnp.einsum('ic, ka, jlbd, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += jnp.einsum('ic, ka, ljdb, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += jnp.einsum('ic, lb, kjad, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += -2.0*jnp.einsum('ikdc, ljab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')

    newT2 += P_OVVO + jnp.transpose(P_OVVO, (1,0,3,2))

    newT1 *= d
    newT2 *= D
    return newT1, newT2

