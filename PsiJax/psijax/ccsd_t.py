import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
from jax.experimental import loops
import numpy as onp
from energy_utils import nuclear_repulsion, partial_tei_transformation, tei_transformation
from ccsd import rccsd 

#def parentheses_T(T1, T2, V, fock_Od, fock_Vd):
#    Voooo, Vooov, Voovv, Vovov, Vovvv, Vvvvv = V
#    # below equations are in chemists, so transpose 
#    Vvvvo = np.transpose(Vovvv, (3,1,2,0))
#    Vvooo = np.transpose(Vooov, (3,1,0,2))
#    Vvovo = np.transpose(Voovv, (2,0,3,1))
#    o,v = T1.shape
#    W = np.zeros((v,v,v))
#    V = np.zeros((v,v,v))
#    pT = 0.0
#    for i in range(o):
#      for j in range(i+1):
#        delta_ij = int(i == j)
#        for k in range(j+1):
#          delta_jk = int(j == k)
#          W  = np.einsum('bda,cd', Vvvvo[:,:,:,i], T2[k,j,:,:])
#          W -= np.einsum('cl,lab', Vvooo[:,k,j,:], T2[i,:,:,:])
#          W += np.einsum('cda,bd', Vvvvo[:,:,:,i], T2[j,k,:,:])
#          W -= np.einsum('bl,lac', Vvooo[:,j,k,:], T2[i,:,:,:])
#          W += np.einsum('adc,bd', Vvvvo[:,:,:,k], T2[j,i,:,:])
#          W -= np.einsum('bl,lca', Vvooo[:,j,i,:], T2[k,:,:,:])
#          W += np.einsum('bdc,ad', Vvvvo[:,:,:,k], T2[i,j,:,:])
#          W -= np.einsum('al,lcb', Vvooo[:,i,j,:], T2[k,:,:,:])
#          W += np.einsum('cdb,ad', Vvvvo[:,:,:,j], T2[i,k,:,:])
#          W -= np.einsum('al,lbc', Vvooo[:,i,k,:], T2[j,:,:,:])
#          W += np.einsum('adb,cd', Vvvvo[:,:,:,j], T2[k,i,:,:])
#          W -= np.einsum('cl,lba', Vvooo[:,k,i,:], T2[j,:,:,:])
#          V  = W + np.einsum('bc,a', Vvovo[:,j,:,k], T1[i,:]) \
#                 + np.einsum('ac,b', Vvovo[:,i,:,k], T1[j,:]) \
#                 + np.einsum('ab,c', Vvovo[:,i,:,j], T1[k,:])
#
#          for a in range(v):
#            for b in range(a+1):
#              delta_ab = int(a == b)
#              for c in range(b+1):
#                delta_bc = int(b == c)
#                Dd = fock_Od[i] + fock_Od[j] + fock_Od[k] - fock_Vd[a] - fock_Vd[b] - fock_Vd[c]
#                X = W[a,b,c]*V[a,b,c] + W[a,c,b]*V[a,c,b] + W[b,a,c]*V[b,a,c]  \
#                  + W[b,c,a]*V[b,c,a] + W[c,a,b]*V[c,a,b] + W[c,b,a]*V[c,b,a]
#                Y = (V[a,b,c] + V[b,c,a] + V[c,a,b])
#                Z = (V[a,c,b] + V[b,a,c] + V[c,b,a])
#                E = (Y - 2*Z)*(W[a,b,c] + W[b,c,a] + W[c,a,b]) + (Z - 2*Y)*(W[a,c,b]+W[b,a,c]+W[c,b,a]) + 3*X
#                pT += E * (2 - delta_ij - delta_jk)  / (Dd * (1 + delta_ab + delta_bc))
#    return pT

# Jittable Jax friendly version
#def parentheses_T(T1, T2, V, fock_Od, fock_Vd):
#    Voooo, Vooov, Voovv, Vovov, Vovvv, Vvvvv = V
#    # below equations are in chemists, so transpose 
#    Vvvvo = np.transpose(Vovvv, (3,1,2,0))
#    Vvooo = np.transpose(Vooov, (3,1,0,2))
#    Vvovo = np.transpose(Voovv, (2,0,3,1))
#    o,v = T1.shape
#    delta_o = np.eye(o)
#    delta_v = np.eye(v)
#    #W = np.zeros((v,v,v))
#    #V = np.zeros((v,v,v))
#
#    with loops.Scope() as s:
#      s.pT = 0.0
#      s.i, s.j, s.k = 0,0,0
#      s.a, s.b, s.c = 0,0,0
#      for _ in s.while_range(lambda: s.i < o):
#        s.j = 0
#        for _ in s.while_range(lambda: s.j < s.i+1):
#          delta_ij = delta_o[s.i,s.j] 
#          s.k = 0
#          for _ in s.while_range(lambda: s.k < s.j+1):
#            delta_jk = delta_o[s.j,s.k] 
#            W  = np.einsum('bda,cd', Vvvvo[:,:,:,s.i], T2[s.k,s.j,:,:])
#            W -= np.einsum('cl,lab', Vvooo[:,s.k,s.j,:], T2[s.i,:,:,:])
#            W += np.einsum('cda,bd', Vvvvo[:,:,:,s.i], T2[s.j,s.k,:,:])
#            W -= np.einsum('bl,lac', Vvooo[:,s.j,s.k,:], T2[s.i,:,:,:])
#            W += np.einsum('adc,bd', Vvvvo[:,:,:,s.k], T2[s.j,s.i,:,:])
#            W -= np.einsum('bl,lca', Vvooo[:,s.j,s.i,:], T2[s.k,:,:,:])
#            W += np.einsum('bdc,ad', Vvvvo[:,:,:,s.k], T2[s.i,s.j,:,:])
#            W -= np.einsum('al,lcb', Vvooo[:,s.i,s.j,:], T2[s.k,:,:,:])
#            W += np.einsum('cdb,ad', Vvvvo[:,:,:,s.j], T2[s.i,s.k,:,:])
#            W -= np.einsum('al,lbc', Vvooo[:,s.i,s.k,:], T2[s.j,:,:,:])
#            W += np.einsum('adb,cd', Vvvvo[:,:,:,s.j], T2[s.k,s.i,:,:])
#            W -= np.einsum('cl,lba', Vvooo[:,s.k,s.i,:], T2[s.j,:,:,:])
#            V  = W + np.einsum('bc,a', Vvovo[:,s.j,:,s.k], T1[s.i,:]) \
#                   + np.einsum('ac,b', Vvovo[:,s.i,:,s.k], T1[s.j,:]) \
#                   + np.einsum('ab,c', Vvovo[:,s.i,:,s.j], T1[s.k,:])
#
#            for a in range(v):
#              for b in range(a+1):
#                delta_ab = delta_v[a,b] 
#                for c in range(b+1):
#                  delta_bc = delta_v[b,c]
#                  Dd = fock_Od[s.i] + fock_Od[s.j] + fock_Od[s.k] - fock_Vd[a] - fock_Vd[b] - fock_Vd[c]
#                  X = W[a,b,c]*V[a,b,c] + W[a,c,b]*V[a,c,b] + W[b,a,c]*V[b,a,c]  \
#                    + W[b,c,a]*V[b,c,a] + W[c,a,b]*V[c,a,b] + W[c,b,a]*V[c,b,a]
#                  Y = (V[a,b,c] + V[b,c,a] + V[c,a,b])
#                  Z = (V[a,c,b] + V[b,a,c] + V[c,b,a])
#                  E = (Y - 2*Z)*(W[a,b,c] + W[b,c,a] + W[c,a,b]) + (Z - 2*Y)*(W[a,c,b]+W[b,a,c]+W[c,b,a]) + 3*X
#                  s.pT += E * (2 - delta_ij - delta_jk)  / (Dd * (1 + delta_ab + delta_bc))
#                  s.c += 1
#                s.b += 1
#              s.a += 1
#            s.k += 1
#          s.j += 1
#        s.i += 1
#    return s.pT

# Jittable Jax friendly version
def parentheses_T(T1, T2, V, fock_Od, fock_Vd):
    Voooo, Vooov, Voovv, Vovov, Vovvv, Vvvvv = V
    # below equations are in chemists, so transpose 
    Vvvvo = np.transpose(Vovvv, (3,1,2,0))
    Vvooo = np.transpose(Vooov, (3,1,0,2))
    Vvovo = np.transpose(Voovv, (2,0,3,1))
    o,v = T1.shape
    delta_o = np.eye(o)
    delta_v = np.eye(v)
    #W = np.zeros((v,v,v))
    #V = np.zeros((v,v,v))

    @jax.jit
    def inner_func(i,j,k):
        delta_ij = delta_o[i,j] 
        delta_jk = delta_o[j,k] 
        W  = np.einsum('bda,cd', Vvvvo[:,:,:,i], T2[k,j,:,:])
        W -= np.einsum('cl,lab', Vvooo[:,k,j,:], T2[i,:,:,:])
        W += np.einsum('cda,bd', Vvvvo[:,:,:,i], T2[j,k,:,:])
        W -= np.einsum('bl,lac', Vvooo[:,j,k,:], T2[i,:,:,:])
        W += np.einsum('adc,bd', Vvvvo[:,:,:,k], T2[j,i,:,:])
        W -= np.einsum('bl,lca', Vvooo[:,j,i,:], T2[k,:,:,:])
        W += np.einsum('bdc,ad', Vvvvo[:,:,:,k], T2[i,j,:,:])
        W -= np.einsum('al,lcb', Vvooo[:,i,j,:], T2[k,:,:,:])
        W += np.einsum('cdb,ad', Vvvvo[:,:,:,j], T2[i,k,:,:])
        W -= np.einsum('al,lbc', Vvooo[:,i,k,:], T2[j,:,:,:])
        W += np.einsum('adb,cd', Vvvvo[:,:,:,j], T2[k,i,:,:])
        W -= np.einsum('cl,lba', Vvooo[:,k,i,:], T2[j,:,:,:])
        V  = W + np.einsum('bc,a', Vvovo[:,j,:,k], T1[i,:]) \
               + np.einsum('ac,b', Vvovo[:,i,:,k], T1[j,:]) \
               + np.einsum('ab,c', Vvovo[:,i,:,j], T1[k,:])
    
        with loops.Scope() as s:
          s.pT_contribution = 0.0
          s.a, s.b, s.c = 0,0,0
          for _ in s.while_range(lambda: s.a < v):
            s.b = 0
            for _ in s.while_range(lambda: s.b < s.a + 1):
              s.c = 0
              for _ in s.while_range(lambda: s.c < s.b + 1):
                delta_ab = delta_v[s.a,s.b] 
                delta_bc = delta_v[s.b,s.c]
                Dd = fock_Od[i] + fock_Od[j] + fock_Od[k] - fock_Vd[s.a] - fock_Vd[s.b] - fock_Vd[s.c]
                X = W[s.a,s.b,s.c]*V[s.a,s.b,s.c] + W[s.a,s.c,s.b]*V[s.a,s.c,s.b] + W[s.b,s.a,s.c]*V[s.b,s.a,s.c]  \
                  + W[s.b,s.c,s.a]*V[s.b,s.c,s.a] + W[s.c,s.a,s.b]*V[s.c,s.a,s.b] + W[s.c,s.b,s.a]*V[s.c,s.b,s.a]
                Y = (V[s.a,s.b,s.c] + V[s.b,s.c,s.a] + V[s.c,s.a,s.b])
                Z = (V[s.a,s.c,s.b] + V[s.b,s.a,s.c] + V[s.c,s.b,s.a])
                E = (Y - 2*Z)*(W[s.a,s.b,s.c] + W[s.b,s.c,s.a] + W[s.c,s.a,s.b]) + (Z - 2*Y)*(W[s.a,s.c,s.b]+W[s.b,s.a,s.c]+W[s.c,s.b,s.a]) + 3*X
                s.pT_contribution += E * (2 - delta_ij - delta_jk)  / (Dd * (1 + delta_ab + delta_bc))
                s.c += 1
              s.b += 1
            s.a += 1
          return s.pT_contribution

    pT = 0.0
    for i in range(o):
      for j in range(i+1):
        for k in range(j+1):
          pT += inner_func(i,j,k)
    return pT


def rccsd_t(geom, basis, nuclear_charges, charge):
    E_ccsd, T1, T2, V, fock_Od, fock_Vd = rccsd(geom, basis, nuclear_charges, charge, return_aux_data=True)
    print("CCSD energy: ", E_ccsd)
    pT = parentheses_T(T1, T2, V, fock_Od, fock_Vd)
    print("(T) energy correction: ", pT)
    print("CCSD(T) total energy: ", E_ccsd + pT)
    return E_ccsd + pT
