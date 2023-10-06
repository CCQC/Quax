import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
import psi4

from .energy_utils import nuclear_repulsion, partial_tei_transformation, tei_transformation
from .hartree_fock import restricted_hartree_fock

def rccsd(geom, basis_set, xyz_path, nuclear_charges, charge, options, deriv_order=0, return_aux_data=False):
    # Do HF
    E_scf, C, eps, V = restricted_hartree_fock(geom, basis_set, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order, return_aux_data=True)

    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2
    nbf = V.shape[0]
    nvir = nbf - ndocc

    o = slice(0, ndocc)
    v = slice(ndocc, nbf)

    # Save slices of two-electron repulsion integrals in MO basis
    V = tei_transformation(V,C)
    V = jnp.swapaxes(V,1,2)
    V = (V[o,o,o,o], V[o,o,o,v], V[o,o,v,v], V[o,v,o,v], V[o,v,v,v], V[v,v,v,v])

    fock_Od = eps[o]
    fock_Vd = eps[v]

    # Oribital energy denominators 
    D = 1.0 / (fock_Od.reshape(-1, 1, 1, 1) + fock_Od.reshape(-1, 1, 1) - fock_Vd.reshape(-1, 1) - fock_Vd)
    d = 1.0 / (fock_Od.reshape(-1, 1) - fock_Vd)

    # Initial Amplitudes
    T1 = jnp.zeros((ndocc,nvir))
    T2 = D*V[2]

    maxit = options['maxit']
    iteration = 0
    E_ccsd = 1.0
    E_old = 0.0
    while abs(E_ccsd - E_old)  > 1e-9:
        E_old = E_ccsd * 1

        T1, T2 = rccsd_iter(T1, T2, V, d, D, ndocc, nvir)
        E_ccsd = rccsd_energy(T1, T2, V[2])

        iteration += 1
        if iteration == maxit:
            break

    print(iteration, " CCSD iterations performed")
    #print("CCSD Correlation Energy:   ", E_ccsd)
    #print("CCSD Total Energy:         ", E_ccsd + E_scf)
    if return_aux_data:
        return E_scf + E_ccsd, T1, T2, V, fock_Od, fock_Vd
    else:
        return E_scf + E_ccsd

# Not a lot of memory use here compared to ccsd iterations, safe to jit-compile this.
@jax.jit
def rccsd_energy(T1, T2, Voovv):
    E_ccsd = 0.0
    E_ccsd -= jnp.tensordot(T1, jnp.tensordot(T1, Voovv, [(0, 1), (1, 2)]), [(0, 1), (0, 1)])
    E_ccsd -= jnp.tensordot(T2, Voovv, [(0, 1, 2, 3), (1, 0, 2, 3)])
    E_ccsd += 2.0*jnp.tensordot(T2, Voovv, [(0, 1, 2, 3),(0, 1, 2, 3)])
    E_ccsd += 2.0*jnp.tensordot(T1, jnp.tensordot(T1, Voovv, [(0, 1), (0, 2)]), [(0, 1), (0, 1)])
    return E_ccsd

# Jit compiling ccsd is a BAD IDEA.
# TODO consider breaking up function and jit compiling those which do not use more memory than TEI transformation
def rccsd_iter(T1, T2, V, d, D, ndocc, nvir):
    Voooo, Vooov, Voovv, Vovov, Vovvv, Vvvvv = V

    newT1 = jnp.zeros(T1.shape)
    newT2 = jnp.zeros(T2.shape)

    # T1 equation
    newT1 += jnp.tensordot(T1, Voovv, [(0, 1), (0, 2)])
    newT1 += jnp.tensordot(T2, Vovvv, [(1, 2, 3), (0, 3, 2)])
    newT1 -= jnp.tensordot(Vooov, T2, [(0, 1, 3), (0, 1, 3)])
    newT1 -= jnp.einsum('kc, la, lkic -> ia', T1, T1, Vooov, optimize = 'optimal')
    newT1 += jnp.einsum('kc, id, kacd -> ia', T1, T1, Vovvv, optimize = 'optimal')
    newT1 -= jnp.einsum('kc, ilad, lkcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 -= jnp.einsum('kc, liad, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 -= jnp.einsum('ic, lkad, lkcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 -= jnp.einsum('la, ikdc, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 -= jnp.einsum('kc, id, la, klcd -> ia', T1, T1, T1, Voovv, optimize = 'optimal')
    newT1 += 2.0 * jnp.einsum('kc, ilad, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 *= 2.0

    newT1 -= jnp.tensordot(T1, Vovov, [(0, 1), (2, 1)])
    newT1 -= jnp.tensordot(T2, Vovvv, [(0, 2, 3), (0, 3, 2)])
    newT1 += jnp.tensordot(Vooov, T2, [(0, 1, 3), (1, 0, 3)])
    newT1 -= jnp.einsum('kc, id, kadc -> ia', T1, T1, Vovvv, optimize = 'optimal')
    newT1 += jnp.einsum('kc, la, klic -> ia', T1, T1, Vooov, optimize = 'optimal')
    newT1 += jnp.einsum('kc, liad, lkcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += jnp.einsum('ic, lkad, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += jnp.einsum('la, ikcd, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += jnp.einsum('kc, id, la, lkcd -> ia', T1, T1, T1, Voovv, optimize = 'optimal')

    # T2 equation
    newT2 -= jnp.einsum('ikac, ljbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 -= jnp.einsum('lkac, ijdb, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 -= jnp.einsum('ikac, jlbd, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 -= jnp.einsum('kiac, jlbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 -= jnp.einsum('ijac, klbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += 2.0 * jnp.einsum('ikac, jlbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 *= 2.0

    # Reducing Vvvvv contractions to tensordot is especially productive.
    # TODO try reducing Vovvv as well. Also check if removing jit makes this optimization moot...
    newT2 += Voovv
    newT2 += jnp.tensordot(T1, jnp.tensordot(T1, Vvvvv, [(1, ), (1, )]), [(1, ), (1, )])
    newT2 += jnp.tensordot(T2, Vvvvv, [(2, 3), (0, 1)])
    newT2 += jnp.einsum('ka, lb, ijkl -> ijab', T1, T1, Voooo, optimize = 'optimal')
    newT2 += jnp.tensordot(T2, Voooo, [(0, 1), (2, 3)]).transpose((2, 3, 0, 1))
    newT2 -= jnp.einsum('ic, jd, ka, kbcd -> ijab', T1, T1, T1, Vovvv, optimize = 'optimal')
    newT2 -= jnp.einsum('ic, jd, kb, kadc -> ijab', T1, T1, T1, Vovvv, optimize = 'optimal')
    newT2 += jnp.einsum('ic, ka, lb, lkjc -> ijab', T1, T1, T1, Vooov, optimize = 'optimal')
    newT2 += jnp.einsum('jc, ka, lb, klic -> ijab', T1, T1, T1, Vooov, optimize = 'optimal')
    newT2 += jnp.einsum('klac, ijdb, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('kiac, ljdb, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('ikac, ljbd, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('kiac, ljbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('ijac, lkbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('kjac, ildb, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('ijdc, lkab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('ic, jd, ka, lb, klcd -> ijab', T1, T1, T1, T1, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('ic, jd, lkab, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    newT2 += jnp.einsum('ka, lb, ijdc, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')

    P_OVVO  = jnp.tensordot(T2, Voovv, [(1, 3),(0, 2)]).transpose((0, 2, 1, 3))
    P_OVVO -= jnp.einsum('lb, ikac, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += jnp.einsum('jc, ikad, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += jnp.einsum('kc, ijad, kbcd -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO -= jnp.einsum('kc, ilab, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO -= jnp.einsum('kc, jd, ilab, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO -= jnp.einsum('kc, la, ijdb, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO -= jnp.einsum('ic, ka, jlbd, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO -= jnp.einsum('ikdc, ljab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    P_OVVO *= 2.0

    P_OVVO -= jnp.tensordot(T1, Vooov, [(0, ), (2, )]).transpose((2, 1, 3, 0))
    P_OVVO += jnp.tensordot(T1, Vovvv, [(1, ), (1, )]).transpose((1, 0, 2, 3))
    P_OVVO -= jnp.tensordot(T2, Voovv, [(0, 3), (0, 2)]).transpose((0, 2, 1, 3))
    P_OVVO -= jnp.einsum('ic, ka, kjcb -> ijab', T1, T1, Voovv, optimize = 'optimal')
    P_OVVO -= jnp.einsum('ic, kb, jcka -> ijab', T1, T1, Vovov, optimize = 'optimal')
    P_OVVO -= jnp.tensordot(T2, Vovov, [(1, 3), (2, 1)]).transpose((0, 2, 1, 3))
    P_OVVO -= jnp.tensordot(T2, Vovov, [(0, 3), (2, 1)]).transpose((2, 0, 1, 3))
    P_OVVO += jnp.einsum('lb, kiac, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO -= jnp.einsum('jc, ikdb, kacd -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO -= jnp.einsum('jc, kiad, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO -= jnp.einsum('jc, ikad, kbcd -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += jnp.einsum('jc, lkab, lkic -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += jnp.einsum('lb, ikac, kljc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO -= jnp.einsum('ka, ijdc, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += jnp.einsum('ka, ilcb, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO -= jnp.einsum('kc, ijad, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += jnp.einsum('kc, ilab, kljc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += jnp.einsum('jkcd, ilab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    P_OVVO += jnp.einsum('kc, jd, ilab, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += jnp.einsum('kc, la, ijdb, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += jnp.einsum('ic, ka, ljbd, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += jnp.einsum('ic, ka, ljdb, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += jnp.einsum('ic, lb, kjad, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')

    newT2 += P_OVVO 
    newT2 += P_OVVO.transpose((1, 0, 3, 2))

    newT1 *= d
    newT2 *= D
    return newT1, newT2

