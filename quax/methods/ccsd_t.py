import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import while_loop

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

        def loop_a(arr0):
           a, b, c, pT_contribution = arr0
           b = 0

           def loop_b(arr1):
              a, b, c, pT_contribution = arr1
              c = 0
              delta_vir = 1 + delta_v[a,b]

              def loop_c(arr2):
                 a, b, c, pT_contribution = arr2
                 delta_vir = delta_vir + delta_v[b,c]
                 Dd = Dd_occ - (fock_Vd[a] + fock_Vd[b] + fock_Vd[c])
                 X = W[a,b,c]*V[a,b,c] + W[a,c,b]*V[a,c,b] + W[b,a,c]*V[b,a,c]  \
                   + W[b,c,a]*V[b,c,a] + W[c,a,b]*V[c,a,b] + W[c,b,a]*V[c,b,a]
                 Y = (V[a,b,c] + V[b,c,a] + V[c,a,b])
                 Z = (V[a,c,b] + V[b,a,c] + V[c,b,a])
                 E = (Y - 2*Z)*(W[a,b,c] + W[b,c,a] + W[c,a,b]) + (Z - 2*Y)*(W[a,c,b]+W[b,a,c]+W[c,b,a]) + 3*X
                 pT_contribution += E * delta_occ / (Dd * delta_vir)
                 c += 1
                 return (a, b, c, pT_contribution)

              a_, b_, c_, pT_contribution_ = while_loop(lambda arr2: arr2[2] < arr2[1] + 1, loop_c, (a, b, c, pT_contribution))
              b_ += 1
              return (a_, b_, c_, pT_contribution_)

           a_, b_, c_, pT_contribution_ = while_loop(lambda arr1: arr1[1] < arr1[0] + 1, loop_b, (a, b, c, pT_contribution))
           a_ += 1
           return (a_, b_, c_, pT_contribution_)

        a_, b_. c_, dE_pT = while_loop(lambda arr0: arr0[0] < v, loop_a, (0, 0, 0, 0.0)) # (a, b, c, pT_contribution)
        return dE_pT

    def loop_i(arr0):
       i, j, k, pT = arr0
       j = 0

       def loop_j(arr1):
          i, j, k, pT = arr1
          k = 0

          def loop_k(arr2):
             i, j, k, pT = arr2
             pT += inner_func(i, j, k)
             k += 1
             return (i, j, k, pT)

          i_, j_, k_, pT_ = while_loop(lambda arr2: arr2[2] < arr2[1] + 1, loop_k, (i, j, k, pT))
          j_ += 1
          return (i_, j_, k_, pT_)

       i_, j_, k_, pT_ = while_loop(lambda arr1: arr1[1] < arr1[0] + 1, loop_j, (i, j, k, pT))
       i_ += 1
       return (i_, j_, k_, pT_)

    i_, j_, k_, pT = while_loop(lambda arr0: arr0[0] < o, loop_i, (0, 0, 0, 0.0)) # (i, j, k, pT)
    return pT

def rccsd_t(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=0):
    E_ccsd, T1, T2, V, fock_Od, fock_Vd = rccsd(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order, return_aux_data=True)
    pT = perturbative_triples(T1, T2, V, fock_Od, fock_Vd)
    #print("(T) energy correction:     ", pT)
    #print("CCSD(T) total energy:      ", E_ccsd + pT)
    return E_ccsd + pT

