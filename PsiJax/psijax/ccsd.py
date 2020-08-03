import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
from jax.experimental import loops
import psi4
import numpy as onp
from energy_utils import nuclear_repulsion, partial_tei_transformation, tei_transformation
from hartree_fock import restricted_hartree_fock

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
        #print(E_ccsd)

        iteration += 1
        if iteration == CC_MAX_ITER:
            break

    if return_aux_data:
        return E_scf + E_ccsd, T1, T2, V, fock_Od, fock_Vd
    else:
        return E_scf + E_ccsd
    #print("Testing (T)")
    #pT = parentheses_T(T1, T2, V, fock_Od, fock_Vd)
    #print(pT)

    
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

def parentheses_T(T1, T2, V, fock_Od, fock_Vd):
    Voooo, Vooov, Voovv, Vovov, Vovvv, Vvvvv = V

    # equations are in chemists, okay
    Vvvvo = np.transpose(Vovvv, (3,1,2,0))
    Vvooo = np.transpose(Vooov, (3,1,0,2))
    Vvovo = np.transpose(Voovv, (2,0,3,1))

    o,v = T1.shape
    W = np.zeros((v,v,v))
    V = np.zeros((v,v,v))


    pT = 0.0

    for i in range(o):
      for j in range(i+1):
        delta_ij = int(i == j)
        for k in range(j+1):
          delta_jk = int(j == k)

          #template
          #W ?= np.einsum(' ', Voooo[:,:,:,:], T2[:,:,:,:])
          #  Vvvvo_4i[b,d,a]*T2_1k_2j[c,d] 
          W  = np.einsum('bda,cd', Vvvvo[:,:,:,i], T2[k,j,:,:])

          #- Vvooo_2k_3j[c,l]*T2_1i[l,a,b] 
          W -= np.einsum('cl,lab', Vvooo[:,k,j,:], T2[i,:,:,:])

          #+  Vvvvo_4i[c,d,a]*T2_1j_2k[b,d] 
          W += np.einsum('cda,bd', Vvvvo[:,:,:,i], T2[j,k,:,:])

          #- Vvooo_2j_3k[b,l]*T2_1i[l,a,c] 
          W -= np.einsum('bl,lac', Vvooo[:,j,k,:], T2[i,:,:,:])

          #+  Vvvvo_4k[a,d,c]*T2_1j_2i[b,d] 
          W += np.einsum('adc,bd', Vvvvo[:,:,:,k], T2[j,i,:,:])

          #- Vvooo_2j_3i[b,l]*T2_1k[l,c,a]  
          W -= np.einsum('bl,lca', Vvooo[:,j,i,:], T2[k,:,:,:])

          #+  Vvvvo_4k[b,d,c]*T2_1i_2j[a,d] 
          W += np.einsum('bdc,ad', Vvvvo[:,:,:,k], T2[i,j,:,:])

          #- Vvooo_2i_3j[a,l]*T2_1k[l,c,b] 
          W -= np.einsum('al,lcb', Vvooo[:,i,j,:], T2[k,:,:,:])

          #+ Vvvvo_4j[c,d,b]*T2_1i_2k[a,d] 
          W += np.einsum('cdb,ad', Vvvvo[:,:,:,j], T2[i,k,:,:])

          #- Vvooo_2i_3k[a,l]*T2_1j[l,b,c]
          W -= np.einsum('al,lbc', Vvooo[:,i,k,:], T2[j,:,:,:])

          #+ Vvvvo_4j[a,d,b]*T2_1k_2i[c,d] 
          W += np.einsum('adb,cd', Vvvvo[:,:,:,j], T2[k,i,:,:])

          #- Vvooo_2k_3i[c,l]*T2_1j[l,b,a]) # jik bac
          W -= np.einsum('cl,lba', Vvooo[:,k,i,:], T2[j,:,:,:])

          V  = W + np.einsum('bc,a', Vvovo[:,j,:,k], T1[i,:]) \
                 + np.einsum('ac,b', Vvovo[:,i,:,k], T1[j,:]) \
                 + np.einsum('ab,c', Vvovo[:,i,:,j], T1[k,:])

          for a in range(v):
            for b in range(a+1):
              delta_ab = int(a == b)
              for c in range(b+1):
                delta_bc = int(b == c)
                Dd = fock_Od[i] + fock_Od[j] + fock_Od[k] - fock_Vd[a] - fock_Vd[b] - fock_Vd[c]
                X = W[a,b,c]*V[a,b,c] + W[a,c,b]*V[a,c,b] + W[b,a,c]*V[b,a,c]  \
                  + W[b,c,a]*V[b,c,a] + W[c,a,b]*V[c,a,b] + W[c,b,a]*V[c,b,a]
                Y = (V[a,b,c] + V[b,c,a] + V[c,a,b])
                Z = (V[a,c,b] + V[b,a,c] + V[c,b,a])
                E = (Y - 2*Z)*(W[a,b,c] + W[b,c,a] + W[c,a,b]) + (Z - 2*Y)*(W[a,c,b]+W[b,a,c]+W[c,b,a]) + 3*X
                pT += E * (2 - delta_ij - delta_jk)  / (Dd * (1 + delta_ab + delta_bc))
    return pT
                
#
#def parentheses_T(T1, T2, V, fock_Od, fock_Vd):
#    Voooo, Vooov, Voovv, Vovov, Vovvv, Vvvvv = V
#
#    o = T2.shape[0]
#    v = T2.shape[2]
#
#    t3c = np.zeros((o,o,o,v,v,v))
#    t3d = np.zeros((o,o,o,v,v,v))
#    # Disconnected TODO permute P(i/jk) P(a/bc)
#    # P(i/jk) * P(a/bc)
#    # (ijk - jik - kji) * (abc - bac - cba)
#    # (ijkabc - ijkbac - ijkcba - jikabc + jikbac + jikcba - kjiabc + kjicac + kjicba)
#    t3d += 8*np.einsum('ia, jkbc -> ', T1, Voovv, optimize = 'optimal')
#    t3d += -8*np.einsum('ia, jkcb -> ', T1, Voovv, optimize = 'optimal')
#
#    # Connected TODO permute P(i/jk) P(a/bc)
#    t3c += -8*np.einsum('kjae, iecb -> ', T2, Vovvv, optimize = 'optimal')
#    t3c += 8*np.einsum('jkae, iecb -> ', T2, Vovvv, optimize = 'optimal')
#    t3c += 8*np.einsum('kjae, iebc -> ', T2, Vovvv, optimize = 'optimal')
#    t3c += -8*np.einsum('jkae, iebc -> ', T2, Vovvv, optimize = 'optimal')
#    t3c += 8*np.einsum('mibc, jkma -> ', T2, Vooov, optimize = 'optimal')
#    t3c += -8*np.einsum('imbc, jkma -> ', T2, Vooov, optimize = 'optimal')
#    t3c += -8*np.einsum('mibc, kjma -> ', T2, Vooov, optimize = 'optimal')
#    t3c += 8*np.einsum('imbc, kjma -> ', T2, Vooov, optimize = 'optimal')
#
#    Dijkabc = fock_Od.reshape(-1, 1, 1, 1, 1, 1) + fock_Od.reshape(-1, 1, 1, 1, 1)  \
#            + fock_Od.reshape(-1, 1, 1, 1) - fock_Vd.reshape(-1, 1, 1) - fock_Vd.reshape(-1, 1) - fock_Vd
#    
#    tmp = (t3c + t3d) / Dijkabc
#
#    pert = (1.0/36) * np.einsum('ijkabc,ijkabc',t3c, tmp)
#    return pert




