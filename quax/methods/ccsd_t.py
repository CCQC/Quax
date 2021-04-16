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

def rccsd_t(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=0):
    E_ccsd, T1, T2, V, fock_Od, fock_Vd = rccsd(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order, return_aux_data=True)
    pT = perturbative_triples(T1, T2, V, fock_Od, fock_Vd)
    #print("(T) energy correction:     ", pT)
    #print("CCSD(T) total energy:      ", E_ccsd + pT)
    return E_ccsd + pT

