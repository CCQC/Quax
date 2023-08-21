import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.lax import fori_loop, while_loop
from functools import partial

from .integrals_utils import gaussian_product, boys, binomial_prefactor, factorials, double_factorials, neg_one_pow, cartesian_product, am_leading_indices, angular_momentum_combinations
from .basis_utils import flatten_basis_data, get_nbf

# Useful resources: Fundamentals of Molecular Integrals Evaluation, Fermann, Valeev https://arxiv.org/abs/2007.12057
#                   Gaussian-Expansion methods of molecular integrals Taketa, Huzinaga, O-ohata 

def overlap(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,prefactor):
    """
    Computes a single overlap integral. Taketa, Huzinaga, Oohata 2.12
    P = gaussian product of aa,A; bb,B
    PA_pow, PB_pow
        All powers of Pi-Ai or Pi-Bi packed into an array
        [[(Px-Ax)^0, (Px-Ax)^1, ... (Px-Ax)^max_am]
         [(Py-Ay)^0, (Py-Ay)^1, ... (Py-Ay)^max_am]
         [(Pz-Az)^0, (Pz-Az)^1, ... (Pz-Az)^max_am]]
    prefactor = jnp.exp(-aa * bb * jnp.dot(A-B,A-B) / gamma)
    """
    gamma = aa + bb
    prefactor *= (jnp.pi / gamma)**1.5

    wx = overlap_component(la,lb,PA_pow[0],PB_pow[0],gamma)
    wy = overlap_component(ma,mb,PA_pow[1],PB_pow[1],gamma)
    wz = overlap_component(na,nb,PA_pow[2],PB_pow[2],gamma)
    return prefactor * wx * wy * wz

def overlap_component(l1,l2,PAx,PBx,gamma):
    """
    The 1d overlap integral component. Taketa, Huzinaga, Oohata 2.12
    """
    K = 1 + (l1 + l2) // 2  

    def loop_i(arr):
       i, total = arr
       return (i+1, total + binomial_prefactor(2*i,l1,l2,PAx,PBx) * double_factorials[2*i-1] / (2*gamma)**i)

    i_accu, total_sum = while_loop(lambda arr: arr[0] < K, loop_i, (0, 0)) # (i, total)

    return total_sum

def kinetic(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,prefactor):
    """
    Computes a single kinetic energy integral.
    """
    gamma = aa + bb
    prefactor *= (jnp.pi / gamma)**1.5
    wx = overlap_component(la,lb,PA_pow[0],PB_pow[0],gamma)
    wy = overlap_component(ma,mb,PA_pow[1],PB_pow[1],gamma)
    wz = overlap_component(na,nb,PA_pow[2],PB_pow[2],gamma)
    wx_plus2 = overlap_component(la,lb+2,PA_pow[0],PB_pow[0],gamma)
    wy_plus2 = overlap_component(ma,mb+2,PA_pow[1],PB_pow[1],gamma)
    wz_plus2 = overlap_component(na,nb+2,PA_pow[2],PB_pow[2],gamma)
    wx_minus2 = overlap_component(la,lb-2,PA_pow[0],PB_pow[0],gamma)
    wy_minus2 = overlap_component(ma,mb-2,PA_pow[1],PB_pow[1],gamma)
    wz_minus2 = overlap_component(na,nb-2,PA_pow[2],PB_pow[2],gamma)

    term1 = bb*(2*(lb+mb+nb)+3) * wx * wy * wz 

    term2 = -2 * bb**2 * (wx_plus2*wy*wz + wx*wy_plus2*wz + wx*wy*wz_plus2)

    term3 = -0.5 * (lb * (lb-1) * wx_minus2 * wy * wz \
                  + mb * (mb-1) * wx * wy_minus2 * wz \
                  + nb * (nb-1) * wx * wy * wz_minus2)
    return prefactor * (term1 + term2 + term3)

def A_array(l1,l2,PA,PB,CP,g,A_vals):

    def loop_i(arr0):
       i, r, u, A = arr0
       Aterm = neg_one_pow[i] * binomial_prefactor(i,l1,l2,PA,PB) * factorials[i]
       r = i // 2

       def loop_r(arr1):
          i, r, u, Aterm, A = arr1
          u = (i - 2 * r) // 2

          def loop_u(arr2):
             i, r, u, Aterm, A = arr2
             I = i - 2 * r - u
             tmp = I - u
             fact_ratio = 1 / (factorials[r] * factorials[u] * factorials[tmp])
             Aterm *= neg_one_pow[u]  * CP[tmp] * (0.25 / g)**(r+u) * fact_ratio
             A = A.at[I].set(u)
             u -= 1
             return (i, r, u, Aterm, A)

          i_, r_, u_, Aterm_, A_ = while_loop(lambda arr2: arr2[1] > -1, loop_u, (i, r, u, Aterm, A))
          r_ -= 1
          return (i_, r_, u_, Aterm_, A_)

       i_, r_, u_, Aterm_, A_ = while_loop(lambda arr1: arr1[1] > -1, loop_r, (i, r, u, Aterm, A))
       i_ -= 1
       return (i_, r_, u_, A_)

    i_, r_, u_, A = while_loop(lambda arr0: arr0[0] > -1, loop_i, (l1 + l2, 0, 0, A_vals)) # (i, r, u, A)

    return A

def potential(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,Pgeom_pow,boys_eval,prefactor,charges,A_vals):
    """
    Computes a single electron-nuclear attraction integral
    """
    gamma = aa + bb
    prefactor *= -2 * jnp.pi / gamma

    def loop_val(n, val):
      Ax = A_array(la,lb,PA_pow[0],PB_pow[0],Pgeom_pow[n,0,:],gamma,A_vals)
      Ay = A_array(ma,mb,PA_pow[1],PB_pow[1],Pgeom_pow[n,1,:],gamma,A_vals)
      Az = A_array(na,nb,PA_pow[2],PB_pow[2],Pgeom_pow[n,2,:],gamma,A_vals)

      I, J, K, total = 0, 0, 0, 0
      def loop_I(arr0):
         I, J, K, val, total = arr0
         J = 0

         def loop_J(arr1):
            I, J, K, val, total = arr1
            K = 0

            def loop_K(arr2):
               I, J, K, val, total = arr2
               total += Ax[I] * Ay[J] * Az[K] * boys_eval[I + J + K, n]
               K += 1
               return (I, J, K, val, total)

            I_, J_, K_, val_, total_ = while_loop(lambda arr2: arr2[2] < na + nb + 1, loop_K, (I, J, K, val, total))
            J_ += 1
            return (I_, J_, K_, val_, total_)

         I_, J_, K_, val_, total_ = while_loop(lambda arr1: arr1[1] < ma + mb + 1, loop_J, (I, J, K, val, total))
         I_ += 1
         return (I_, J_, K_, val_, total_)

      I_, J_, K_, val_, total_ = while_loop(lambda arr0: arr0[0] < la + lb + 1, loop_I, (I, J, K, val, total))
      val_ += charges[n] * prefactor * total_
      return val_

    val = fori_loop(0, Pgeom_pow.shape[0], loop_val, 0)
    return val

def oei_arrays(geom, basis, charges):
    """
    Build one electron integral arrays (overlap, kinetic, and potential integrals)
    """
    coeffs, exps, atoms, ams, indices, dims = flatten_basis_data(basis)
    nbf = get_nbf(basis)
    nprim = coeffs.shape[0]
    max_am = jnp.max(ams)
    A_vals = jnp.zeros(2*max_am+1)

    # Save various AM distributions for indexing
    # Obtain all possible primitive quartet index combinations 
    primitive_duets = cartesian_product(jnp.arange(nprim), jnp.arange(nprim))
    STV = jnp.zeros((3,nbf,nbf))

    for n in range(primitive_duets.shape[0]):
       p1,p2 = primitive_duets[n]
       coef = coeffs[p1] * coeffs[p2]
       aa, bb = exps[p1], exps[p2]
       atom1, atom2 = atoms[p1], atoms[p2]
       am1, am2 = ams[p1], ams[p2]
       A, B = geom[atom1], geom[atom2]
       ld1, ld2 = am_leading_indices[am1], am_leading_indices[am2]

       gamma = aa + bb
       prefactor = jnp.exp(-aa * bb * jnp.dot(A-B,A-B) / gamma)
       P = (aa * A + bb * B) / gamma
       # Maximum angular momentum: hard coded
       #max_am = 3 # f function support
       # Precompute all powers up to 2+max_am of Pi-Ai, Pi-Bi.
       # We need 2+max_am since kinetic requires incrementing angluar momentum by +2
       PA_pow = jnp.power(jnp.broadcast_to(P-A, (max_am+3, 3)).T, jnp.arange(max_am+3))
       PB_pow = jnp.power(jnp.broadcast_to(P-B, (max_am+3, 3)).T, jnp.arange(max_am+3))

       # For potential integrals, we need the difference between
       # the gaussian product center P and ALL atoms in the molecule,
       # and then take all possible powers up to 2*max_am.
       # We pre-collect this into a 3d array, and then just pull out what we need via indexing in the loops, so they need not be recomputed.
       # The resulting array has dimensions (atom, cartesian component, power) so index (0, 1, 3) would return (Py - atom0_y)^3
       P_minus_geom = jnp.broadcast_to(P, geom.shape) - geom
       Pgeom_pow = jnp.power(jnp.transpose(jnp.broadcast_to(P_minus_geom, (2*max_am + 1,geom.shape[0],geom.shape[1])), (1,2,0)), jnp.arange(2*max_am + 1))
       # All possible jnp.dot(P-atom,P-atom)
       rcp2 = jnp.einsum('ij,ij->i', P_minus_geom, P_minus_geom)
       # All needed (and unneeded, for am < max_am) boys function evaluations
       boys_arg = jnp.broadcast_to(rcp2 * gamma, (2*max_am+1, geom.shape[0]))
       boys_nu = jnp.tile(jnp.arange(2*max_am+1), (geom.shape[0],1)).T
       boys_eval = boys(boys_nu,boys_arg)

       a, b = 0, 0
       def loop_a(arr0):
          a, b, oei = arr0
          b = 0

          def loop_b(arr1):
             a, b, oei = arr1
             # Gather angular momentum and index
             la,ma,na = angular_momentum_combinations[a + ld1]
             lb,mb,nb = angular_momentum_combinations[b + ld2]
             # To only create unique indices, need to have separate indices arrays for i and j.
             i = indices[p1] + a
             j = indices[p2] + b
             # Compute one electron integrals and add to appropriate index
             overlap_int = overlap(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,prefactor) * coef
             kinetic_int = kinetic(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,prefactor) * coef
             potential_int = potential(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,Pgeom_pow,boys_eval,prefactor,charges,A_vals) * coef
             oei = oei.at[([0,1,2],[i,i,i],[j,j,j])].set((overlap_int, kinetic_int, potential_int))
             b += 1
             return (a, b, oei)

          a_, b_, oei_ = while_loop(lambda arr1: arr1[1] < dims[p2], loop_b, (a, b, oei))
          a_ += 1
          return (a_, b_, oei_)

       a_, b_, oei_ = while_loop(lambda arr0: arr0[0] < dims[p1], loop_a, (a, b, STV))

       return oei_

    return STV[0], STV[1], STV[2]

