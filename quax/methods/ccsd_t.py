import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental import loops

from .energy_utils import nuclear_repulsion, partial_tei_transformation, tei_transformation
from .ccsd import rccsd 
from ..integrals import integrals_utils

def perturbative_triples(T1, T2, V, fock_Od, fock_Vd):
    Voooo, Vooov, Voovv, Vovov, Vovvv, Vvvvv = V
    o,v = T1.shape
    delta_o = jnp.eye(o)
    delta_v = jnp.eye(v)

    def inner_func(i,j,k):
        delta_ij = delta_o[i,j] 
        delta_jk = delta_o[j,k] 
        W  = jnp.einsum('dab,cd', Vovvv[i,:,:,:], T2[k,j,:,:]) 
        W += jnp.einsum('dac,bd', Vovvv[i,:,:,:], T2[j,k,:,:]) 
        W += jnp.einsum('dca,bd', Vovvv[k,:,:,:], T2[j,i,:,:])  
        W += jnp.einsum('dcb,ad', Vovvv[k,:,:,:], T2[i,j,:,:])
        W += jnp.einsum('dbc,ad', Vovvv[j,:,:,:], T2[i,k,:,:])
        W += jnp.einsum('dba,cd', Vovvv[j,:,:,:], T2[k,i,:,:])
        W -= jnp.einsum('lc,lab', Vooov[:,k,j,:], T2[i,:,:,:])
        W -= jnp.einsum('lb,lac', Vooov[:,j,k,:], T2[i,:,:,:]) 
        W -= jnp.einsum('lb,lca', Vooov[:,j,i,:], T2[k,:,:,:])
        W -= jnp.einsum('la,lcb', Vooov[:,i,j,:], T2[k,:,:,:])
        W -= jnp.einsum('la,lbc', Vooov[:,i,k,:], T2[j,:,:,:])
        W -= jnp.einsum('lc,lba', Vooov[:,k,i,:], T2[j,:,:,:])
        V  = W + jnp.einsum('bc,a', Voovv[j,k,:,:], T1[i,:]) \
               + jnp.einsum('ac,b', Voovv[i,k,:,:], T1[j,:]) \
               + jnp.einsum('ab,c', Voovv[i,j,:,:], T1[k,:])


        delta_occ = 2 - delta_ij - delta_jk
        Dd_occ = fock_Od[i] + fock_Od[j] + fock_Od[k] 
        with loops.Scope() as s:
          s.pT_contribution = 0.0
          s.a, s.b, s.c = 0,0,0
          for _ in s.while_range(lambda: s.a < v): #TODO this could be converted to s.range, may improve autodiff performance
            s.b = 0
            for _ in s.while_range(lambda: s.b < s.a + 1):
              delta_vir = 1 + delta_v[s.a,s.b] 
              s.c = 0
              for _ in s.while_range(lambda: s.c < s.b + 1):
                delta_vir = delta_vir + delta_v[s.b,s.c]
                Dd = Dd_occ - (fock_Vd[s.a] + fock_Vd[s.b] + fock_Vd[s.c])
                X = W[s.a,s.b,s.c]*V[s.a,s.b,s.c] + W[s.a,s.c,s.b]*V[s.a,s.c,s.b] + W[s.b,s.a,s.c]*V[s.b,s.a,s.c]  \
                  + W[s.b,s.c,s.a]*V[s.b,s.c,s.a] + W[s.c,s.a,s.b]*V[s.c,s.a,s.b] + W[s.c,s.b,s.a]*V[s.c,s.b,s.a]
                Y = (V[s.a,s.b,s.c] + V[s.b,s.c,s.a] + V[s.c,s.a,s.b])
                Z = (V[s.a,s.c,s.b] + V[s.b,s.a,s.c] + V[s.c,s.b,s.a])
                E = (Y - 2*Z)*(W[s.a,s.b,s.c] + W[s.b,s.c,s.a] + W[s.c,s.a,s.b]) + (Z - 2*Y)*(W[s.a,s.c,s.b]+W[s.b,s.a,s.c]+W[s.c,s.b,s.a]) + 3*X
                s.pT_contribution += E * delta_occ / (Dd * delta_vir)
                s.c += 1
              s.b += 1
            s.a += 1
          return s.pT_contribution

    #with loops.Scope() as S:
    #  S.pT = 0.0
    #  S.j, S.k = 0, 0
    #  for i in S.range(o): # Uses lax.scan for fixed bounds (supposedly better for autodiff https://github.com/google/jax/issues/3850)
    #    S.j = 0
    #    for _ in S.while_range(lambda: S.j < i + 1): # while_loop for unknown bounds
    #      S.k = 0
    #      for _ in S.while_range(lambda: S.k < S.j + 1): 
    #        S.pT += inner_func(i,S.j,S.k)
    #        S.k += 1
    #      S.j += 1
    #  return S.pT
    with loops.Scope() as S:
      S.pT = 0.0
      S.i, S.j, S.k = 0, 0, 0
      for _ in S.while_range(lambda: S.i < o): 
        S.j = 0
        for _ in S.while_range(lambda: S.j < S.i + 1): 
          S.k = 0
          for _ in S.while_range(lambda: S.k < S.j + 1): 
            S.pT += inner_func(S.i,S.j,S.k)
            S.k += 1
          S.j += 1
        S.i += 1
      return S.pT

def vectorized_perturbative_triples(T1, T2, V, fock_Od, fock_Vd):
    Voooo, Vooov, Voovv, Vovov, Vovvv, Vvvvv = V
    # below equations are in chemists, so transpose 
    Vvvvo = jnp.transpose(Vovvv, (3,1,2,0))
    Vvooo = jnp.transpose(Vooov, (3,1,0,2))
    Vvovo = jnp.transpose(Voovv, (2,0,3,1))
    o,v = T1.shape
    delta_o = jnp.eye(o)
    delta_v = jnp.eye(v)
    # IDEA: Build up index arrays which mimic loop structure TODO regular numpy probably better here, with int16's
    occ_range = jnp.arange(o)
    vir_range = jnp.arange(v)
    occ_indices = cartesian_product(occ_range,occ_range,occ_range)
    i,j,k = occ_indices[:,0], occ_indices[:,1], occ_indices[:,2]
    occ_cond = (i <= j) & (j <= k)
    vir_indices = cartesian_product(vir_range,vir_range,vir_range)
    a,b,c = occ_indices[:,0], occ_indices[:,1], occ_indices[:,2]
    vir_cond = (a <= b) & (b <= c)
    # Now have all indices prepared
    occ_indices = occ_indices[occ_cond]
    vir_indices = vir_indices[vir_cond]
    i,j,k = occ_indices[:,0], occ_indices[:,1], occ_indices[:,2]
    a,b,c = occ_indices[:,0], occ_indices[:,1], occ_indices[:,2]

    @jax.jit
    def inner_func(i,j,k):
        delta_ij = delta_o[i,j] 
        delta_jk = delta_o[j,k] 
        W  = jnp.einsum('bda,cd', Vvvvo[:,:,:,i], T2[k,j,:,:])
        W -= jnp.einsum('cl,lab', Vvooo[:,k,j,:], T2[i,:,:,:])
        W += jnp.einsum('cda,bd', Vvvvo[:,:,:,i], T2[j,k,:,:])
        W -= jnp.einsum('bl,lac', Vvooo[:,j,k,:], T2[i,:,:,:])
        W += jnp.einsum('adc,bd', Vvvvo[:,:,:,k], T2[j,i,:,:])
        W -= jnp.einsum('bl,lca', Vvooo[:,j,i,:], T2[k,:,:,:])
        W += jnp.einsum('bdc,ad', Vvvvo[:,:,:,k], T2[i,j,:,:])
        W -= jnp.einsum('al,lcb', Vvooo[:,i,j,:], T2[k,:,:,:])
        W += jnp.einsum('cdb,ad', Vvvvo[:,:,:,j], T2[i,k,:,:])
        W -= jnp.einsum('al,lbc', Vvooo[:,i,k,:], T2[j,:,:,:])
        W += jnp.einsum('adb,cd', Vvvvo[:,:,:,j], T2[k,i,:,:])
        W -= jnp.einsum('cl,lba', Vvooo[:,k,i,:], T2[j,:,:,:])
        V  = W + jnp.einsum('bc,a', Vvovo[:,j,:,k], T1[i,:]) \
               + jnp.einsum('ac,b', Vvovo[:,i,:,k], T1[j,:]) \
               + jnp.einsum('ab,c', Vvovo[:,i,:,j], T1[k,:])

        delta_occ = 2 - delta_ij - delta_jk
        Dd = fock_Od[i] + fock_Od[j] + fock_Od[k] 
    
        with loops.Scope() as s:
          s.pT_contribution = 0.0
          # TODO is while looping better here?
          for vir_idx in s.range(vir_indices.shape[0]): 
            a,b,c = vir_indices[vir_idx]
            delta_ab = delta_v[a,b] 
            delta_bc = delta_v[b,c]
            #Dd = fock_Od[i] + fock_Od[j] + fock_Od[k] - fock_Vd[a] - fock_Vd[b] - fock_Vd[c]
            Dd -= fock_Vd[a] + fock_Vd[b] + fock_Vd[c]
            X = W[a,b,c]*V[a,b,c] + W[a,c,b]*V[a,c,b] + W[b,a,c]*V[b,a,c]  \
              + W[b,c,a]*V[b,c,a] + W[c,a,b]*V[c,a,b] + W[c,b,a]*V[c,b,a]
            Y = (V[a,b,c] + V[b,c,a] + V[c,a,b])
            Z = (V[a,c,b] + V[b,a,c] + V[c,b,a])
            E = (Y - 2*Z)*(W[a,b,c] + W[b,c,a] + W[c,a,b]) + (Z - 2*Y)*(W[a,c,b]+W[b,a,c]+W[c,b,a]) + 3*X
            #s.pT_contribution += E * (2 - delta_ij - delta_jk)  / (Dd * (1 + delta_ab + delta_bc))
            s.pT_contribution += E * (delta_occ)  / (Dd * (1 + delta_ab + delta_bc))
          return s.pT_contribution

    with loops.Scope() as S:
      S.pT = 0.0
      for occ_idx in S.range(occ_indices.shape[0]): 
        i,j,k = occ_indices[occ_idx]
        S.pT += inner_func(i,j,k)
      return S.pT
   
def rccsd_t(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=0):
    E_ccsd, T1, T2, V, fock_Od, fock_Vd = rccsd(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order, return_aux_data=True)
    pT = perturbative_triples(T1, T2, V, fock_Od, fock_Vd)
    #print("(T) energy correction:     ", pT)
    #print("CCSD(T) total energy:      ", E_ccsd + pT)
    return E_ccsd + pT

