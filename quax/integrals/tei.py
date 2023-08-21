import jax 
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import fori_loop, while_loop

from .basis_utils import flatten_basis_data, get_nbf
from .integrals_utils import gaussian_product, boys, binomial_prefactor, cartesian_product, am_leading_indices, angular_momentum_combinations, fact_ratio2, neg_one_pow

def B_array(l1,l2,l3,l4,pa_pow,pb_pow,qc_pow,qd_pow,qp_pow,g1_pow,g2_pow,oodelta_pow,B_vals):
    #TODO can you do some reduction magic to reduce the number of loops?
    # Can you split it into two Scopes?
    # Can you convert  all or part of this to a tensor contraction?  
    # It does not appear to help to pull out binomial prefactors and compute outside loop.

    def loop_i1(arr0):
       i1, i2, r1, r2, u, B = arr0
       Bterm = binomial_prefactor(arr0[0],l1,l2,pa_pow,pb_pow)
       tmp = i1
       r1 = i1 // 2

       def loop_r1(arr1):
          i1, i2, r1, r2, u, B = arr1
          Bterm *= fact_ratio2[i1,r1]
          Bterm *= g1_pow[r1-i1]
          tmp -= 2 * r1
          i2 = l3 + l4

          def loop_i2(arr2):
             i1, i2, r1, r2, u, B = arr2
             Bterm *= neg_one_pow[i2]
             Bterm *= binomial_prefactor(i2,l3,l4,qc_pow,qd_pow)
             tmp += i2
             r2 = i2 // 2

             def loop_r2(arr3):
                i1, i2, r1, r2, u, B = arr3
                Bterm *= fact_ratio2[i2,r2]
                Bterm *= g2_pow[r2-i2]
                tmp -= 2 * r2
                u = tmp // 2

                def loop_u(arr4):
                   i1, i2, r1, r2, u, B = arr4
                   I = tmp - u
                   Bterm *= neg_one_pow[u]
                   Bterm *= fact_ratio2[tmp,u]
                   Bterm *= qp_pow[tmp - 2 * u]
                   Bterm *= oodelta_pow[I]
                   B = B.at[I].set(Bterm)
                   u -= 1
                   return (i1, i2, r1, r2, u, B)

                i1_, i2_, r1_, r2_, u_, B_ = while_loop(lambda arr4: arr4[4] > -1, loop_u, (i1, i2, r1, r2, u, B))
                r2_ -= 1
                return (i1_, i2_, r1_, r2_, u_, B_)

             i1_, i2_, r1_, r2_, u_, B_ = while_loop(lambda arr3: arr3[3] > -1, loop_r2, (i1, i2, r1, r2, u, B))
             i2_ -= 1
             return (i1_, i2_, r1_, r2_, u_, B_)

          i1_, i2_, r1_, r2_, u_, B_ = while_loop(lambda arr2: arr2[1] > -1, loop_i2, (i1, i2, r1, r2, u, B))
          r1_ -= 1
          return (i1_, i2_, r1_, r2_, u_, B_)

       i1_, i2_, r1_, r2_, u_, B_ = while_loop(lambda arr1: arr1[2] > -1, loop_r1, (i1, i2, r1, r2, u, B))
       i1_ -= 1
       return (i1_, i2_, r1_, r2_, u_, B_)

    i1, i2, r1, r2, u, B = while_loop(lambda arr0: arr0[0] > -1, loop_i1, (l1 + l2, 0, 0, 0, 0, B_vals)) # (i1, i2, r1, r2, u, B)
    return B

# def primitive_tei(La,Lb,Lc,Ld, A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4):
#     """
#     TODO can define a jvp rule for this, have it increment arguments appropriately
#     Computes a single contracted two electron integral.
#     given angular momentum vectors, centers, and single value exponents and contraction coefficients
#     """
#     # NOTE THIS FUNCTION IS NOT USED.
#     # For debugging. This is implementation is directly coded into tei_array
#     # in order to save some intermediates.
#     la, ma, na = La
#     lb, mb, nb = Lb
#     lc, mc, nc = Lc
#     ld, md, nd = Ld
#     xa,ya,za = A
#     xb,yb,zb = B
#     xc,yc,zc = C
#     xd,yd,zd = D

#     rab2 = jnp.dot(A-B,A-B)
#     rcd2 = jnp.dot(C-D,C-D)
#     coef = c1 * c2 * c3 * c4
#     xyzp = gaussian_product(aa,A,bb,B)
#     xyzq = gaussian_product(cc,C,dd,D)
#     xp,yp,zp = xyzp
#     xq,yq,zq = xyzq
#     rpq2 = jnp.dot(xyzp-xyzq,xyzp-xyzq)
#     gamma1 = aa + bb
#     gamma2 = cc + dd
#     delta = 0.25*(1/gamma1+1/gamma2)
#     Bx = B_array(la,lb,lc,ld,xp,xa,xb,xq,xc,xd,gamma1,gamma2,delta)
#     By = B_array(ma,mb,mc,md,yp,ya,yb,yq,yc,yd,gamma1,gamma2,delta)
#     Bz = B_array(na,nb,nc,nd,zp,za,zb,zq,zc,zd,gamma1,gamma2,delta)
#     boys_arg = 0.25*rpq2/delta
#     boys_eval = boys(jnp.arange(13), boys_arg) # supports up to f functions

#     with loops.Scope() as s:
#       s.I = 0
#       s.J = 0
#       s.K = 0
#       s.primitive = 0.
#       s.I = 0
#       for _ in s.while_range(lambda: s.I < la + lb + lc + ld + 1):
#         s.J = 0
#         for _ in s.while_range(lambda: s.J < ma + mb + mc + md + 1):
#           s.K = 0
#           for _ in s.while_range(lambda: s.K < na + nb + nc + nd + 1):
#             s.primitive += Bx[s.I] * By[s.J] * Bz[s.K] * boys_eval[s.I + s.J + s.K]
#             s.K += 1
#           s.J += 1
#         s.I += 1
#       value = 2*jax.lax.pow(jnp.pi,2.5)/(gamma1*gamma2*jnp.sqrt(gamma1+gamma2)) \
#               *jnp.exp(-aa*bb*rab2/gamma1) \
#               *jnp.exp(-cc*dd*rcd2/gamma2)*s.primitive*coef
#       return value

def tei_array(geom, basis):
    """
    Build two electron integral array from a jax.numpy array of the cartesian geometry in Bohr, 
    and a basis dictionary as defined by basis_utils.build_basis_set
    We have to loop over primitives rather than shells because JAX needs intermediates to be consistent 
    sizes in order to compile.
    """
    # Smush primitive data together into vectors
    coeffs, exps, atoms, ams, indices, dims = flatten_basis_data(basis)
    nbf = get_nbf(basis)
    max_am = jnp.max(ams)
    max_am_idx = max_am * 4 + 1 
    #TODO add excpetion raise if angular momentum is too high
    B_vals = jnp.zeros(4*max_am+1)  
    nprim = coeffs.shape[0]
    # Obtain all possible primitive quartet index combinations 
    primitive_quartets = cartesian_product(jnp.arange(nprim), jnp.arange(nprim), jnp.arange(nprim), jnp.arange(nprim))

    #print("Number of basis functions: ", nbf)
    #print("Number of primitve quartets: ", primitive_quartets.shape[0])

    #TODO Experimental: precompute quantities and lookup inside loop
    # Compute all possible Gaussian products for this basis set
    aa_plus_bb = jnp.broadcast_to(exps, (nprim,nprim)) + jnp.transpose(jnp.broadcast_to(exps, (nprim,nprim)), (1,0))
    aa_times_A = jnp.einsum('i,ij->ij', exps, geom[atoms])
    aaxA_plus_bbxB = aa_times_A[:,None,:] + aa_times_A[None,:,:]
    gaussian_products = jnp.einsum('ijk,ij->ijk', aaxA_plus_bbxB, 1/aa_plus_bb)  

    # Compute all rab2 (rcd2), every possible jnp.dot(A-B,A-B)
    natom = geom.shape[0]
    tmpA = jnp.broadcast_to(geom, (natom,natom,3))
    AminusB = (tmpA - jnp.transpose(tmpA, (1,0,2)))
    AmBdot = jnp.einsum('ijk,ijk->ij', AminusB, AminusB) # shape: (natom,natom)

    # Compute all differences between gaussian product centers with all atom centers
    tmpP = jnp.tile(gaussian_products, natom).reshape(nprim,nprim,natom,3)
    PminusA = tmpP - jnp.broadcast_to(geom, tmpP.shape)

    # Commpute all powers (up to max_am) of differences between gaussian product centers and atom centers
    # Shape: (nprim, nprim, natom, 3, max_am+1). In loop index PA_pow as [p1,p2,atoms[p1],:,:]
    PminusA_pow = jnp.power(jnp.transpose(jnp.broadcast_to(PminusA, (max_am+1,nprim,nprim,natom,3)), (1,2,3,4,0)), jnp.arange(max_am+1))

    def loop_prim_quartets(n, G):
      # Load in primitive indices, coeffs, exponents, centers, angular momentum index, and leading placement index in TEI array
      p1,p2,p3,p4 = primitive_quartets[n]
      coef = coeffs[p1] * coeffs[p2] * coeffs[p3] * coeffs[p4]
      aa, bb, cc, dd = exps[p1], exps[p2], exps[p3], exps[p4]
      ld1, ld2, ld3, ld4 = am_leading_indices[ams[p1]],am_leading_indices[ams[p2]],am_leading_indices[ams[p3]],am_leading_indices[ams[p4]]
      idx1, idx2, idx3, idx4 = indices[p1],indices[p2],indices[p3],indices[p4],
      #A, B, C, D = geom[atoms[p1]], geom[atoms[p2]], geom[atoms[p3]], geom[atoms[p4]]

      # Compute common intermediates before looping over AM distributions.
      # Avoids redundant recomputations/reassignment for all classes other than (ss|ss).
      #AB = A - B
      #CD = C - D
      #rab2 = jnp.dot(AB,AB)
      #rcd2 = jnp.dot(CD,CD)
      #P = (aa * A + bb * B) / gamma1
      #Q = (cc * C + dd * D) / gamma2
      gamma1 = aa + bb
      gamma2 = cc + dd

      #TODO
      P = gaussian_products[p1,p2]
      Q = gaussian_products[p3,p4]
      rab2 = AmBdot[atoms[p1],atoms[p2]]
      rcd2 = AmBdot[atoms[p3],atoms[p4]]
      #PA = PminusA[p1,p2,atoms[p1]]
      #PB = PminusA[p1,p2,atoms[p2]]
      #QC = PminusA[p3,p4,atoms[p3]]
      #QD = PminusA[p3,p4,atoms[p4]]
      #TODO

      PQ = P - Q
      rpq2 = jnp.dot(PQ,PQ)
      delta = 0.25*(1/gamma1+1/gamma2)
      boys_arg = 0.25 * rpq2 / delta
      boys_eval = boys(jnp.arange(max_am_idx), boys_arg)

      # Need all powers of Pi-Ai,Pi-Bi,Qi-Ci,Qi-Di (i=x,y,z) up to max_am and Qi-Pi up to max_am_idx
      # note: this computes unncessary quantities for lower angular momentum,
      # but avoids repeated computation of the same quantities in loops for higher angular momentum

      #PA_pow = jnp.power(jnp.broadcast_to(P-A, (max_am+1,3)).T, jnp.arange(max_am+1))
      #PB_pow = jnp.power(jnp.broadcast_to(P-B, (max_am+1,3)).T, jnp.arange(max_am+1))
      #QC_pow = jnp.power(jnp.broadcast_to(Q-C, (max_am+1,3)).T, jnp.arange(max_am+1))
      #QD_pow = jnp.power(jnp.broadcast_to(Q-D, (max_am+1,3)).T, jnp.arange(max_am+1))

      PA_pow = PminusA_pow[p1,p2,atoms[p1],:,:]
      PB_pow = PminusA_pow[p1,p2,atoms[p2],:,:]
      QC_pow = PminusA_pow[p3,p4,atoms[p3],:,:]
      QD_pow = PminusA_pow[p3,p4,atoms[p4],:,:]
      QP_pow = jnp.power(jnp.broadcast_to(Q-P, (max_am_idx,3)).T, jnp.arange(max_am_idx))

      # Gamma powers are negative, up to -(l1+l2).
      # Make array such that the given negative index returns the same negative power.
      g1_pow = jnp.power(4*gamma1, -jnp.roll(jnp.flip(jnp.arange(2*max_am+1)),1))
      g2_pow = jnp.power(4*gamma2, -jnp.roll(jnp.flip(jnp.arange(2*max_am+1)),1))
      oodelta_pow = jnp.power(1 / delta, jnp.arange(max_am_idx))  # l1 + l2 + l3 + l4 + 1

      prefactor = 34.986836655249726 / (gamma1*gamma2*jnp.sqrt(gamma1+gamma2)) \
                  * jnp.exp(-aa*bb*rab2/gamma1 + -cc*dd*rcd2/gamma2) * coef

      a, b, c, d = 0, 0, 0, 0
      def loop_a(arr0):
         a, b, c, d, G = arr0
         b = 0

         def loop_b(arr1):
            a, b, c, d, G = arr1
            c = 0

            def loop_c(arr2):
               a, b, c, d, G = arr2
               d = 0

               def loop_d(arr3):
                  a, b, c, d, G = arr3
                  # Collect angular momentum and index in G
                  la, ma, na = angular_momentum_combinations[a + ld1]
                  lb, mb, nb = angular_momentum_combinations[b + ld2]
                  lc, mc, nc = angular_momentum_combinations[c + ld3]
                  ld, md, nd = angular_momentum_combinations[d + ld4]
                  i = idx1 + a
                  j = idx2 + b
                  k = idx3 + c
                  l = idx4 + d
                  # Compute the primitive quartet tei and add to appropriate index in G
                  Bx = B_array(la,lb,lc,ld,PA_pow[0],PB_pow[0],QC_pow[0],QD_pow[0],QP_pow[0],g1_pow,g2_pow,oodelta_pow,B_vals)
                  By = B_array(ma,mb,mc,md,PA_pow[1],PB_pow[1],QC_pow[1],QD_pow[1],QP_pow[1],g1_pow,g2_pow,oodelta_pow,B_vals)
                  Bz = B_array(na,nb,nc,nd,PA_pow[2],PB_pow[2],QC_pow[2],QD_pow[2],QP_pow[2],g1_pow,g2_pow,oodelta_pow,B_vals)

                  I, J, K, primitive = 0, 0, 0, 0.0
                  def loop_I(arrI):
                     I, J, K, primitive = arrI
                     J = 0
                     tmp = Bx[I]

                     def loop_J(arrJ):
                        I, J, K, primitive = arrJ
                        K = 0
                        tmp *= By[J]

                        def loop_K(arrK):
                           I, J, K, primitive = arrK
                           tmp *= Bz[K] * boys_eval[I + J + K]
                           primitive += tmp
                           K += 1
                           return (I, J, K, primitive)

                        I_, J_, K_, primitive_ = while_loop(lambda arrK: arrK[2] < na + nb + nc + nd + 1, loop_K, (I, J, K, primitive))
                        J_ += 1
                        return (I_, J_, K_, primitive_)

                     I_, J_, K_, primitive_ = while_loop(lambda arrJ: arrJ[1] < ma + mb + mc + md + 1, loop_J, (I, J, K, primitive))
                     I_ += 1 # I
                     return (I_, J_, K_, primitive_)

                  I_, J_, K_, primitive_ = while_loop(lambda arrI: arrI[0] < la + lb + lc + ld + 1, loop_I, (I, J, K, primitive))

                  tei = prefactor * primitive_
                  G = G.at[i, j, k, l].set(tei)
                  d += 1
                  return (a, b, c, d, G)

               a_, b_, c_, d_, G_ = while_loop(lambda arr3: arr3[3] < dims[arr3[6]], loop_d, arr2)
               c_ += 1
               return (a_, b_, c_, d_, G_)

            a_, b_, c_, d_, G_ = while_loop(lambda arr2: arr2[2] < dims[arr2[5]], loop_c, arr1)
            b_ += 1
            return (a_, b_, c_, d_, G_)

         a_, b_, c_, d_, G_ = while_loop(lambda arr1: arr1[1] < dims[arr1[4]], loop_b, arr0)
         a_ += 1
         return (a_, b_, c_, d_, G_)

      a_, b_, c_, d_, G_ = while_loop(lambda arr0: arr0[0] < dims[p1], loop_a, (a, b, c, d, G))
      return G_

    G = fori_loop(0, primitive_quartets.shape[0], loop_prim_quartets, jnp.zeros((nbf,nbf,nbf,nbf)))
    return G

