import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import while_loop

from .energy_utils import nuclear_repulsion, partial_tei_transformation, tei_transformation
from .ccsd import rccsd

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
           a_0, b_0, c_0, pT_contribution_0 = arr0
           b_0 = 0

           def loop_b(arr1):
              a_1, b_1, c_1, pT_contribution_1 = arr1
              c_1 = 0
              delta_vir = 1 + delta_v[a_1, b_1]

              def loop_c(arr2):
                 a_2, b_2, c_2, delta_vir_2, pT_contribution_2 = arr2
                 delta_vir_2 = delta_vir_2 + delta_v[b_2,c_2]
                 Dd = Dd_occ - (fock_Vd[a_2] + fock_Vd[b_2] + fock_Vd[c_2])
                 X = W[a_2, b_2, c_2] * V[a_2, b_2, c_2] + W[a_2, c_2, b_2] * V[a_2, c_2, b_2] + W[b_2, a_2, c_2] * V[b_2, a_2, c_2]  \
                   + W[b_2, c_2, a_2] * V[b_2, c_2, a_2] + W[c_2, a_2, b_2] * V[c_2, a_2, b_2] + W[c_2, b_2, a_2] * V[c_2, b_2, a_2]
                 Y = (V[a_2, b_2, c_2] + V[b_2, c_2, a_2] + V[c_2, a_2, b_2])
                 Z = (V[a_2, c_2, b_2] + V[b_2, a_2, c_2] + V[c_2, b_2, a_2])
                 E = (Y - 2 * Z) * (W[a_2, b_2, c_2] + W[b_2, c_2, a_2] + W[c_2, a_2, b_2]) \
                   + (Z - 2 * Y) * (W[a_2, c_2, b_2] + W[b_2, a_2, c_2] + W[c_2, b_2, a_2]) + 3 * X
                 pT_contribution_2 += E * delta_occ / (Dd * delta_vir_2)
                 c_2 += 1
                 return (a_2, b_2, c_2, delta_vir_2, pT_contribution_2)

              a_1_, b_1_, c_1_, delta_vir_, pT_contribution_1_ = while_loop(lambda arr2: arr2[2] < arr2[1] + 1, loop_c, (a_1, b_1, c_1, delta_vir, pT_contribution_1))
              b_1_ += 1
              return (a_1_, b_1_, c_1_, pT_contribution_1_)

           a_0_, b_0_, c_0_, pT_contribution_0_ = while_loop(lambda arr1: arr1[1] < arr1[0] + 1, loop_b, (a_0, b_0, c_0, pT_contribution_0))
           a_0_ += 1
           return (a_0_, b_0_, c_0_, pT_contribution_0_)

        a, b, c, dE_pT = while_loop(lambda arr0: arr0[0] < v, loop_a, (0, 0, 0, 0.0)) # (a, b, c, pT_contribution)
        return dE_pT

    def loop_i(arr0):
       i_0, j_0, k_0, pT_0 = arr0
       j_0 = 0

       def loop_j(arr1):
          i_1, j_1, k_1, pT_1 = arr1
          k_1 = 0

          def loop_k(arr2):
             i_2, j_2, k_2, pT_2 = arr2
             pT_2 += inner_func(i_2, j_2, k_2)
             k_2 += 1
             return (i_2, j_2, k_2, pT_2)

          i_1_, j_1_, k_1_, pT_1_ = while_loop(lambda arr2: arr2[2] < arr2[1] + 1, loop_k, (i_1, j_1, k_1, pT_1))
          j_1_ += 1
          return (i_1_, j_1_, k_1_, pT_1_)

       i_0_, j_0_, k_0_, pT_0_ = while_loop(lambda arr1: arr1[1] < arr1[0] + 1, loop_j, (i_0, j_0, k_0, pT_0))
       i_0_ += 1
       return (i_0_, j_0_, k_0_, pT_0_)

    i, j, k, pT = while_loop(lambda arr0: arr0[0] < o, loop_i, (0, 0, 0, 0.0)) # (i, j, k, pT)
    return pT

def rccsd_t(geom, basis_set, xyz_path, nuclear_charges, charge, options, deriv_order=0):
    E_ccsd, T1, T2, V, fock_Od, fock_Vd = rccsd(geom, basis_set, xyz_path, nuclear_charges, charge, options, deriv_order=deriv_order, return_aux_data=True)
    pT = perturbative_triples(T1, T2, V, fock_Od, fock_Vd)
    #print("(T) energy correction:     ", pT)
    #print("CCSD(T) total energy:      ", E_ccsd + pT)
    return E_ccsd + pT

