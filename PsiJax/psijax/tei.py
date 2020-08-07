import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
from jax.experimental import loops
from basis_utils import flatten_basis_data, get_nbf
from integrals_utils import gaussian_product, boys, binomial_prefactor, cartesian_product, am_leading_indices, angular_momentum_combinations, fact_ratio2

#def B_array(l1,l2,l3,l4,pa,pb,qc,qd,qp,g1,g2,delta):
#    # This originally made arrays with argument-dependent shapes. Need fix size for jit compiling
#    # Hard code only up to f functions (fxxx, fxxx | fxxx, fxxx) => l1 + l2 + l3 + l4 + 1
#    g1 *= 4
#    g2 *= 4
#    oodelta = 1 / delta
#
#    with loops.Scope() as s:
#      s.B = np.zeros(13)
#      s.i2 = 0
#      s.r1 = 0
#      s.r2 = 0
#      s.u = 0 
#      s.i1 = l1 + l2  
#      for _ in s.while_range(lambda: s.i1 > -1):   
#        Bterm = binomial_prefactor(s.i1,l1,l2,pa,pb) 
#        s.r1 = s.i1 // 2
#        for _ in s.while_range(lambda: s.r1 > -1):
#          Bterm *= fact_ratio2[s.i1,s.r1] * (g1)**(s.r1-s.i1)
#          s.i2 = l3 + l4 
#          for _ in s.while_range(lambda: s.i2 > -1):
#            Bterm *= (-1)**s.i2 * binomial_prefactor(s.i2,l3,l4,qc,qd) 
#            s.r2 = s.i2 // 2
#            for _ in s.while_range(lambda: s.r2 > -1):
#              Bterm *= fact_ratio2[s.i2,s.r2] * (g2)**(s.r2-s.i2)
#              tmp = s.i1 + s.i2 - 2 * (s.r1 + s.r2)
#              s.u = (s.i1 + s.i2) // 2 - s.r1 - s.r2 
#              for _ in s.while_range(lambda: s.u > -1):
#                Bterm *= (-1)**s.u * fact_ratio2[tmp,s.u]
#                Bterm *= (qp)**(tmp - 2 * s.u)
#                Bterm *= oodelta**(tmp-s.u)
#                I = tmp - s.u 
#                s.B = jax.ops.index_add(s.B, I, Bterm)
#                s.u -= 1
#              s.r2 -= 1
#            s.i2 -= 1
#          s.r1 -= 1
#        s.i1 -= 1
#      return s.B

neg_one_pow = np.array([1,-1,1,-1,1,-1,1,-1,1,-1,1,-1])

def B_array(l1,l2,l3,l4,pa,pb,qc,qd,qp,g1,g2,delta):
    # This originally made arrays with argument-dependent shapes. Need fix size for jit compiling
    # Hard code only up to f functions (fxxx, fxxx | fxxx, fxxx) => l1 + l2 + l3 + l4 + 1

    oodelta_pow = np.power(1 / delta, np.arange(13))       # l1 + l2 + l3 + l4 + 1
    g1_pow = np.power(4 * g1, np.array([0,-6,-5,-4,-3,-2,-1])) # -(l1 + l2) -> -(l1 + l2) // 2 
    g2_pow = np.power(4 * g2, np.array([0,-6,-5,-4,-3,-2,-1])) # -(l3 + l4) -> -(l3 + l4) // 2 

    qp_pow = np.power(qp, np.arange(13))

    with loops.Scope() as s:
      s.B = np.zeros(13)
      s.i2 = 0
      s.r1 = 0
      s.r2 = 0
      s.u = 0 
      s.i1 = l1 + l2  
      for _ in s.while_range(lambda: s.i1 > -1):   
        Bterm = binomial_prefactor(s.i1,l1,l2,pa,pb) 
        s.r1 = s.i1 // 2
        for _ in s.while_range(lambda: s.r1 > -1):
          Bterm *= fact_ratio2[s.i1,s.r1] * g1_pow[s.r1-s.i1] 
          s.i2 = l3 + l4 
          for _ in s.while_range(lambda: s.i2 > -1):
            Bterm *= neg_one_pow[s.i2] * binomial_prefactor(s.i2,l3,l4,qc,qd) 
            s.r2 = s.i2 // 2
            for _ in s.while_range(lambda: s.r2 > -1):
              Bterm *= fact_ratio2[s.i2,s.r2] * g2_pow[s.r2-s.i2]
              tmp = s.i1 + s.i2 - 2 * (s.r1 + s.r2)
              s.u = tmp // 2
              for _ in s.while_range(lambda: s.u > -1):
                Bterm *= neg_one_pow[s.u] * fact_ratio2[tmp,s.u]
                Bterm *= qp_pow[tmp - 2 * s.u]
                I = tmp - s.u 
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

    rab2 = np.dot(A-B,A-B)
    rcd2 = np.dot(C-D,C-D)
    coef = c1 * c2 * c3 * c4
    xyzp = gaussian_product(aa,A,bb,B)
    xyzq = gaussian_product(cc,C,dd,D)
    xp,yp,zp = xyzp
    xq,yq,zq = xyzq
    rpq2 = np.dot(xyzp-xyzq,xyzp-xyzq)
    gamma1 = aa + bb
    gamma2 = cc + dd
    delta = 0.25*(1/gamma1+1/gamma2)
    Bx = B_array(la,lb,lc,ld,xp,xa,xb,xq,xc,xd,gamma1,gamma2,delta)
    By = B_array(ma,mb,mc,md,yp,ya,yb,yq,yc,yd,gamma1,gamma2,delta)
    Bz = B_array(na,nb,nc,nd,zp,za,zb,zq,zc,zd,gamma1,gamma2,delta)
    boys_arg = 0.25*rpq2/delta
    boys_eval = boys(np.arange(13), boys_arg) # supports up to f functions

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
            #s.primitive += Bx[s.I] * By[s.J] * Bz[s.K] * boys(s.I + s.J + s.K, boys_arg)
            s.primitive += Bx[s.I] * By[s.J] * Bz[s.K] * boys_eval[s.I + s.J + s.K]
            s.K += 1
          s.J += 1
        s.I += 1
      value = 2*jax.lax.pow(np.pi,2.5)/(gamma1*gamma2*np.sqrt(gamma1+gamma2)) \
              *np.exp(-aa*bb*rab2/gamma1) \
              *np.exp(-cc*dd*rcd2/gamma2)*s.primitive*coef
      return value

def tei_array(geom, basis):
    """
    Build two electron integral array from a jax.numpy array of the cartesian geometry in Bohr, 
    and a basis dictionary as defined by basis_utils.build_basis_set
    """
    # Smush primitive data together into vectors
    coeffs, exps, atoms, ams, indices, dims = flatten_basis_data(basis)
    nbf = get_nbf(basis)
    nprim = coeffs.shape[0]
    # Obtain all possible primitive quartet index combinations 
    primitive_quartets = cartesian_product(np.arange(nprim), np.arange(nprim), np.arange(nprim), np.arange(nprim))

    with loops.Scope() as s:
      s.G = np.zeros((nbf,nbf,nbf,nbf))
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
        A, B, C, D = geom[atoms[p1]], geom[atoms[p2]], geom[atoms[p3]], geom[atoms[p4]]

        # Compute common intermediates before looping over AM distributions.
        # Avoids redundant recomputations/reassignment for all classes other than (ss|ss).
        AB = A - B
        CD = C - D
        rab2 = np.dot(AB,AB)
        rcd2 = np.dot(CD,CD)
        gamma1 = aa + bb
        gamma2 = cc + dd
        P = (aa * A + bb * B) / gamma1
        Q = (cc * C + dd * D) / gamma2
        PQ = P - Q
        rpq2 = np.dot(PQ,PQ)
        PAx, PAy, PAz = P - A
        PBx, PBy, PBz = P - B
        QCx, QCy, QCz = Q - C
        QDx, QDy, QDz = Q - D
        QPx, QPy, QPz = Q - P
        delta = 0.25*(1/gamma1+1/gamma2)
        boys_arg = 0.25 * rpq2 / delta
        boys_eval = boys(np.arange(13), boys_arg) # NOTE supports f functions l1+l2+l3+l4+1
        prefactor = 34.986836655249726 / (gamma1*gamma2*np.sqrt(gamma1+gamma2)) \
                    * np.exp(-aa*bb*rab2/gamma1 + -cc*dd*rcd2/gamma2) * coef

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
                Bx = B_array(la,lb,lc,ld,PAx,PBx,QCx,QDx,QPx,gamma1,gamma2,delta)
                By = B_array(ma,mb,mc,md,PAy,PBy,QCy,QDy,QPy,gamma1,gamma2,delta)
                Bz = B_array(na,nb,nc,nd,PAz,PBz,QCz,QDz,QPz,gamma1,gamma2,delta)
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
                #tei = primitive_quartet(La,Lb,Lc,Ld,A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)
                s.G = jax.ops.index_add(s.G, jax.ops.index[i,j,k,l], tei) 
                s.d += 1
              s.c += 1
            s.b += 1
          s.a += 1
      return s.G

# Example evaluation and test against Psi4
import psi4
import numpy as onp
from basis_utils import build_basis_set
molecule = psi4.geometry("""
                         0 1
                         N 0.0 0.0 -0.849220457955
                         N 0.0 0.0  0.849220457955
                         units bohr
                         """)
geom = np.asarray(onp.asarray(molecule.geometry()))
basis_name = 'cc-pvtz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
G = tei_array(geom, basis_dict)

#mints = psi4.core.MintsHelper(basis_set)
#psi_G = np.asarray(onp.asarray(mints.ao_eri()))
#print("Matches Psi4: ", np.allclose(G, psi_G))
#

