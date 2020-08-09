import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
from jax.experimental import loops
import psi4
import numpy as onp

from .energy_utils import nuclear_repulsion, partial_tei_transformation, tei_transformation
from .hartree_fock import restricted_hartree_fock

def rccsd(geom, basis, nuclear_charges, charge, return_aux_data=False):
    # Do HF
    E_scf, C, eps, V, H = restricted_hartree_fock(geom, basis, nuclear_charges, charge, SCF_MAX_ITER=15, return_aux_data=True)

    nelectrons = int(np.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2
    nvir = V.shape[0] - ndocc

    o = slice(0, ndocc)
    v = slice(ndocc, V.shape[0])
    # Transform one-electron hamiltonian to MO basis
    H = np.einsum('up,vq,uv->pq', C, C, H)
    # Transform TEI's to MO basis
    V = tei_transformation(V,C)
    # Form MO fock matrix
    F = H + 2 * np.einsum('pqkk->pq', V[:,:,o,o]) - np.einsum('pkqk->pq', V[:,o,:,o])
    # Save diagonal terms
    fock_Od = np.diagonal(F)[o]
    fock_Vd = np.diagonal(F)[v]
    # Erase diagonal elements from original matrix
    F = F - np.diag(np.diag(F))

    # Save useful slices
    fock_OO = F[o,o]
    fock_VV = F[v,v]
    fock_OV = F[o,v]
    f = (fock_OO, fock_OV, fock_VV)

    # Save slices of two-electron repulsion integral
    V = np.swapaxes(V, 1,2)
    V = (V[o,o,o,o], V[o,o,o,v], V[o,o,v,v], V[o,v,o,v], V[o,v,v,v], V[v,v,v,v])

    # Auxilliary D matrix
    D = 1.0 / (fock_Od.reshape(-1,1,1,1) + fock_Od.reshape(-1,1,1) - fock_Vd.reshape(-1,1) - fock_Vd)
    d = 1.0 / (fock_Od.reshape(-1,1) - fock_Vd)

    # Initial Amplitudes
    T1 = f[1]*d
    T2 = D*V[2]

    # Pre iterations
    CC_MAX_ITER = 30
    iteration = 0
    E_ccsd = 1.0
    E_old = 0.0
    while abs(E_ccsd - E_old)  > 1e-9:
        E_old = E_ccsd * 1

        T1, T2 = rccsd_iter(T1, T2, f, V, d, D, ndocc, nvir)
        Voovv = V[2]
        E_ccsd = 0.
        E_ccsd += 2.0*np.einsum('kc, kc -> ', fock_OV, T1, optimize = 'optimal')
        E_ccsd += -1.0*np.einsum('lc, kd, klcd -> ', T1, T1, Voovv, optimize = 'optimal')
        E_ccsd += -1.0*np.einsum('lckd, klcd -> ', np.transpose(T2,(0,2,1,3)), Voovv, optimize = 'optimal')
        E_ccsd += 2.0*np.einsum('lckd, klcd -> ', np.transpose(T2,(1,2,0,3)), Voovv, optimize = 'optimal')
        E_ccsd += 2.0*np.einsum('lc, kd, lkcd -> ', T1, T1, Voovv, optimize = 'optimal')

        iteration += 1
        if iteration == CC_MAX_ITER:
            break

    print("CCSD Correlation Energy:   ", E_ccsd)
    print("CCSD Total Energy:         ", E_ccsd + E_scf)
    if return_aux_data:
        return E_scf + E_ccsd, T1, T2, V, fock_Od, fock_Vd
    else:
        return E_scf + E_ccsd

@jax.jit
def rccsd_iter(T1, T2, f, V, d, D, ndocc, nvir):
    Voooo, Vooov, Voovv, Vovov, Vovvv, Vvvvv = V
    fock_OO, fock_OV, fock_VV = f

    newT1 = np.zeros(T1.shape)
    newT2 = np.zeros(T2.shape)

    # T1 equation
    newT1 += fock_OV
    newT1 -= np.einsum('ik, ka -> ia', fock_OO, T1, optimize = 'optimal')
    newT1 += np.einsum('ca, ic -> ia', fock_VV, T1, optimize = 'optimal')
    newT1 -= np.einsum('kc, ic, ka -> ia', fock_OV, T1, T1, optimize = 'optimal')
    newT1 += 2.0*np.einsum('kc, ikac -> ia', fock_OV, T2, optimize = 'optimal')
    newT1 -= np.einsum('kc, kiac -> ia', fock_OV, T2, optimize = 'optimal')
    newT1 -= np.einsum('kc, icka -> ia', T1, Vovov, optimize = 'optimal')
    newT1 += 2.0*np.einsum('kc, kica -> ia', T1, Voovv, optimize = 'optimal')
    newT1 -= np.einsum('kicd, kadc -> ia', T2, Vovvv, optimize = 'optimal')
    newT1 += 2.0*np.einsum('ikcd, kadc -> ia', T2, Vovvv, optimize = 'optimal')
    newT1 += -2.0*np.einsum('klac, klic -> ia', T2, Vooov, optimize = 'optimal')
    newT1 += np.einsum('lkac, klic -> ia', T2, Vooov, optimize = 'optimal')
    newT1 += -2.0*np.einsum('kc, la, lkic -> ia', T1, T1, Vooov, optimize = 'optimal')
    newT1 -= np.einsum('kc, id, kadc -> ia', T1, T1, Vovvv, optimize = 'optimal')
    newT1 += 2.0*np.einsum('kc, id, kacd -> ia', T1, T1, Vovvv, optimize = 'optimal')
    newT1 += np.einsum('kc, la, klic -> ia', T1, T1, Vooov, optimize = 'optimal')
    newT1 += -2.0*np.einsum('kc, ilad, lkcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += -2.0*np.einsum('kc, liad, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += np.einsum('kc, liad, lkcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += -2.0*np.einsum('ic, lkad, lkcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += np.einsum('ic, lkad, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += -2.0*np.einsum('la, ikdc, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += np.einsum('la, ikcd, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += np.einsum('kc, id, la, lkcd -> ia', T1, T1, T1, Voovv, optimize = 'optimal')
    newT1 += -2.0*np.einsum('kc, id, la, klcd -> ia', T1, T1, T1, Voovv, optimize = 'optimal')
    newT1 += 4.0*np.einsum('kc, ilad, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')

    # T2 equation
    newT2 += Voovv
    newT2 += np.einsum('ic, jd, cdab -> ijab', T1, T1, Vvvvv, optimize = 'optimal')
    newT2 += np.einsum('ijcd, cdab -> ijab', T2, Vvvvv, optimize = 'optimal')
    newT2 += np.einsum('ka, lb, ijkl -> ijab', T1, T1, Voooo, optimize = 'optimal')
    newT2 += np.einsum('klab, ijkl -> ijab', T2, Voooo, optimize = 'optimal')
    newT2 -= np.einsum('ic, jd, ka, kbcd -> ijab', T1, T1, T1, Vovvv, optimize = 'optimal')
    newT2 -= np.einsum('ic, jd, kb, kadc -> ijab', T1, T1, T1, Vovvv, optimize = 'optimal')
    newT2 += np.einsum('ic, ka, lb, lkjc -> ijab', T1, T1, T1, Vooov, optimize = 'optimal')
    newT2 += np.einsum('jc, ka, lb, klic -> ijab', T1, T1, T1, Vooov, optimize = 'optimal')
    newT2 += np.einsum('klac, ijdb, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('ikac, ljbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('lkac, ijdb, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('kiac, ljdb, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ikac, ljbd, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('ikac, jlbd, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('kiac, ljbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('kiac, jlbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ijac, lkbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('ijac, klbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('kjac, ildb, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += 4.0*np.einsum('ikac, jlbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ijdc, lkab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ic, jd, ka, lb, klcd -> ijab', T1, T1, T1, T1, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ic, jd, lkab, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ka, lb, ijdc, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO = -1.0*np.einsum('ik, kjab -> ijab', fock_OO, T2, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('ca, ijcb -> ijab', fock_VV, T2, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('kb, jika -> ijab', T1, Vooov, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('jc, icab -> ijab', T1, Vovvv, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('kc, ic, kjab -> ijab', fock_OV, T1, T2, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('kc, ka, ijcb -> ijab', fock_OV, T1, T2, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('kiac, kjcb -> ijab', T2, Voovv, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('ic, ka, kjcb -> ijab', T1, T1, Voovv, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('ic, kb, jcka -> ijab', T1, T1, Vovov, optimize = 'optimal')
    P_OVVO += 2.0*np.einsum('ikac, kjcb -> ijab', T2, Voovv, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('ikac, jckb -> ijab', T2, Vovov, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('kjac, ickb -> ijab', T2, Vovov, optimize = 'optimal')
    P_OVVO += -2.0*np.einsum('lb, ikac, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('lb, kiac, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('jc, ikdb, kacd -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('jc, kiad, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('jc, ikad, kbcd -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('jc, lkab, lkic -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('lb, ikac, kljc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('ka, ijdc, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('ka, ilcb, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += 2.0*np.einsum('jc, ikad, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += -1.0*np.einsum('kc, ijad, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += 2.0*np.einsum('kc, ijad, kbcd -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('kc, ilab, kljc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += -2.0*np.einsum('kc, ilab, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('jkcd, ilab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    P_OVVO += -2.0*np.einsum('kc, jd, ilab, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('kc, jd, ilab, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += -2.0*np.einsum('kc, la, ijdb, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('kc, la, ijdb, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('ic, ka, ljbd, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += -2.0*np.einsum('ic, ka, jlbd, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('ic, ka, ljdb, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += 1.0*np.einsum('ic, lb, kjad, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += -2.0*np.einsum('ikdc, ljab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    
    newT2 += P_OVVO + np.transpose(P_OVVO, (1,0,3,2))

    newT1 *= d
    newT2 *= D
    return newT1, newT2

