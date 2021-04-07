import jax 
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental import loops

from .basis_utils import flatten_basis_data, get_nbf
from .integrals_utils import gaussian_product, boys, binomial_prefactor, cartesian_product, am_leading_indices, angular_momentum_combinations, fact_ratio2, neg_one_pow

def B_array(l1,l2,l3,l4,pa_pow,pb_pow,qc_pow,qd_pow,qp_pow,g1_pow,g2_pow,oodelta_pow,B_vals):
    #TODO can you do some reduction magic to reduce the number of loops?
    # Can you split it into two Scopes?
    # Can you convert  all or part of this to a tensor contraction?  
    # It does not appear to help to pull out binomial prefactors and compute outside loop.
    with loops.Scope() as s:
      s.B = B_vals
      s.i2 = 0
      s.r1 = 0
      s.r2 = 0
      s.u = 0 
      s.i1 = l1 + l2  
      for _ in s.while_range(lambda: s.i1 > -1):
        Bterm = binomial_prefactor(s.i1,l1,l2,pa_pow,pb_pow) 
        tmp = s.i1
        s.r1 = s.i1 // 2
        for _ in s.while_range(lambda: s.r1 > -1):
          Bterm *= fact_ratio2[s.i1,s.r1]
          Bterm *= g1_pow[s.r1-s.i1]
          tmp -= 2 * s.r1
          s.i2 = l3 + l4 
          for _ in s.while_range(lambda: s.i2 > -1):
            Bterm *= neg_one_pow[s.i2]
            Bterm *= binomial_prefactor(s.i2,l3,l4,qc_pow,qd_pow) 
            tmp += s.i2
            s.r2 = s.i2 // 2
            for _ in s.while_range(lambda: s.r2 > -1):
              Bterm *= fact_ratio2[s.i2,s.r2]
              Bterm *= g2_pow[s.r2-s.i2]
              tmp -= 2 * s.r2
              s.u = tmp // 2
              for _ in s.while_range(lambda: s.u > -1):
                I = tmp - s.u 
                Bterm *= neg_one_pow[s.u] 
                Bterm *= fact_ratio2[tmp,s.u]
                Bterm *= qp_pow[tmp - 2 * s.u]
                Bterm *= oodelta_pow[I]
                s.B = jax.ops.index_add(s.B, I, Bterm)
                s.u -= 1
              s.r2 -= 1
            s.i2 -= 1
          s.r1 -= 1
        s.i1 -= 1
      return s.B

def primitive_tei(La,Lb,Lc,Ld, A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4): 
    """
    TODO can define a jvp rule for this, have it increment arguments appropriately
    Computes a single contracted two electron integral. 
    given angular momentum vectors, centers, and single value exponents and contraction coefficients
    """
    # NOTE THIS FUNCTION IS NOT USED. 
    # For debugging. This is implementation is directly coded into tei_array 
    # in order to save some intermediates.
    la, ma, na = La
    lb, mb, nb = Lb
    lc, mc, nc = Lc
    ld, md, nd = Ld
    xa,ya,za = A 
    xb,yb,zb = B 
    xc,yc,zc = C 
    xd,yd,zd = D 

    rab2 = jnp.dot(A-B,A-B)
    rcd2 = jnp.dot(C-D,C-D)
    coef = c1 * c2 * c3 * c4
    xyzp = gaussian_product(aa,A,bb,B)
    xyzq = gaussian_product(cc,C,dd,D)
    xp,yp,zp = xyzp
    xq,yq,zq = xyzq
    rpq2 = jnp.dot(xyzp-xyzq,xyzp-xyzq)
    gamma1 = aa + bb
    gamma2 = cc + dd
    delta = 0.25*(1/gamma1+1/gamma2)
    Bx = B_array(la,lb,lc,ld,xp,xa,xb,xq,xc,xd,gamma1,gamma2,delta)
    By = B_array(ma,mb,mc,md,yp,ya,yb,yq,yc,yd,gamma1,gamma2,delta)
    Bz = B_array(na,nb,nc,nd,zp,za,zb,zq,zc,zd,gamma1,gamma2,delta)
    boys_arg = 0.25*rpq2/delta
    boys_eval = boys(jnp.arange(13), boys_arg) # supports up to f functions

    with loops.Scope() as s:
      s.I = 0
      s.J = 0  
      s.K = 0 
      s.primitive = 0.
      s.I = 0 
      for _ in s.while_range(lambda: s.I < la + lb + lc + ld + 1):
        s.J = 0 
        for _ in s.while_range(lambda: s.J < ma + mb + mc + md + 1):
          s.K = 0 
          for _ in s.while_range(lambda: s.K < na + nb + nc + nd + 1):
            s.primitive += Bx[s.I] * By[s.J] * Bz[s.K] * boys_eval[s.I + s.J + s.K]
            s.K += 1
          s.J += 1
        s.I += 1
      value = 2*jax.lax.pow(jnp.pi,2.5)/(gamma1*gamma2*jnp.sqrt(gamma1+gamma2)) \
              *jnp.exp(-aa*bb*rab2/gamma1) \
              *jnp.exp(-cc*dd*rcd2/gamma2)*s.primitive*coef
      return value

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

    with loops.Scope() as s:
      s.G = jnp.zeros((nbf,nbf,nbf,nbf))
      s.a = 0  # center A angular momentum iterator 
      s.b = 0  # center B angular momentum iterator 
      s.c = 0  # center C angular momentum iterator 
      s.d = 0  # center D angular momentum iterator 

      # Loop over primitive quartets, compute integral, add to appropriate index in G
      for prim_quar in s.range(primitive_quartets.shape[0]):
        # Load in primitive indices, coeffs, exponents, centers, angular momentum index, and leading placement index in TEI array
        p1,p2,p3,p4 = primitive_quartets[prim_quar] 
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

        # TODO is there symmetry here?
        s.a = 0
        for _ in s.while_range(lambda: s.a < dims[p1]):
          s.b = 0
          for _ in s.while_range(lambda: s.b < dims[p2]):
            s.c = 0
            for _ in s.while_range(lambda: s.c < dims[p3]):
              s.d = 0
              for _ in s.while_range(lambda: s.d < dims[p4]):
                # Collect angular momentum and index in G
                la, ma, na = angular_momentum_combinations[s.a + ld1]
                lb, mb, nb = angular_momentum_combinations[s.b + ld2]
                lc, mc, nc = angular_momentum_combinations[s.c + ld3]
                ld, md, nd = angular_momentum_combinations[s.d + ld4]
                i = idx1 + s.a
                j = idx2 + s.b
                k = idx3 + s.c
                l = idx4 + s.d
                # Compute the primitive quartet tei and add to appropriate index in G
                Bx = B_array(la,lb,lc,ld,PA_pow[0],PB_pow[0],QC_pow[0],QD_pow[0],QP_pow[0],g1_pow,g2_pow,oodelta_pow,B_vals)
                By = B_array(ma,mb,mc,md,PA_pow[1],PB_pow[1],QC_pow[1],QD_pow[1],QP_pow[1],g1_pow,g2_pow,oodelta_pow,B_vals)
                Bz = B_array(na,nb,nc,nd,PA_pow[2],PB_pow[2],QC_pow[2],QD_pow[2],QP_pow[2],g1_pow,g2_pow,oodelta_pow,B_vals)

                with loops.Scope() as S:
                  S.primitive = 0.
                  S.I = 0
                  S.J = 0
                  S.K = 0
                  for _ in S.while_range(lambda: S.I < la + lb + lc + ld + 1):
                    S.J = 0 
                    tmp = Bx[S.I] 
                    for _ in S.while_range(lambda: S.J < ma + mb + mc + md + 1):
                      S.K = 0 
                      tmp *= By[S.J] 
                      for _ in S.while_range(lambda: S.K < na + nb + nc + nd + 1):
                        tmp *= Bz[S.K] * boys_eval[S.I + S.J + S.K]
                        S.primitive += tmp
                        S.K += 1
                      S.J += 1
                    S.I += 1
                tei = prefactor * S.primitive
                s.G = jax.ops.index_add(s.G, jax.ops.index[i,j,k,l], tei) 

                s.d += 1
              s.c += 1
            s.b += 1
          s.a += 1
      return s.G

