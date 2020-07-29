import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
from jax.experimental import loops
from basis_utils import flatten_basis_data, get_nbf
from integrals_utils import gaussian_product, boys, binomial_prefactor, factorial, cartesian_product, am_leading_indices, angular_momentum_combinations

def fB(i,l1,l2,P,A,B,r,g): 
    return binomial_prefactor(i,l1,l2,P-A,P-B) * B0(i,r,g)

def B0(i,r,g): 
    return fact_ratio2(i,r) * (4*g)**(r-i)

def fact_ratio2(a,b):
    return factorial(a)/factorial(b)/factorial(a-2*b)


def B_term(i1,i2,r1,r2,u,l1,l2,l3,l4,Px,Ax,Bx,Qx,Cx,Dx,gamma1,gamma2,delta):
    val = fB(i1,l1,l2,Px,Ax,Bx,r1,gamma1) \
           * (-1)**i2 * fB(i2,l3,l4,Qx,Cx,Dx,r2,gamma2) \
           * (-1)**u * fact_ratio2(i1+i2-2*(r1+r2),u) \
           * (Qx-Px)**(i1+i2-2*(r1+r2)-2*u) \
           / delta**(i1+i2-2*(r1+r2)-u)
    return val

def B_array(l1,l2,l3,l4,p,a,b,q,c,d,g1,g2,delta):
    # This originally made arrays with argument-dependent shapes. Need fix size for jit compiling
    # Hard code only up to f functions (fxxx, fxxx | fxxx, fxxx) => l1 + l2 + l3 + l4 + 1
    with loops.Scope() as s:
      s.B = np.zeros(13)
      s.i2 = 0
      s.r1 = 0
      s.r2 = 0
      s.u = 0 
      s.i1 = l1 + l2  
      for _ in s.while_range(lambda: s.i1 > -1):   
        s.r1 = s.i1 // 2
        for _ in s.while_range(lambda: s.r1 > -1):
          s.i2 = l3 + l4 
          for _ in s.while_range(lambda: s.i2 > -1):
            s.r2 = s.i2 // 2
            for _ in s.while_range(lambda: s.r2 > -1):
              s.u = (s.i1 + s.i2) // 2 - s.r1 - s.r2 
              for _ in s.while_range(lambda: s.u > -1):
                I = s.i1 + s.i2 - 2 * (s.r1 + s.r2) - s.u 
                term = B_term(s.i1,s.i2,s.r1,s.r2,s.u,l1,l2,l3,l4,p,a,b,q,c,d,g1,g2,delta) 
                s.B = jax.ops.index_add(s.B, I, term)
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
    # Save various AM distributions for indexing
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
        p1,p2,p3,p4 = primitive_quartets[prim_quar] 
        c1, c2, c3, c4 = coeffs[p1], coeffs[p2], coeffs[p3], coeffs[p4]
        aa, bb, cc, dd = exps[p1], exps[p2], exps[p3], exps[p4]
        atom1, atom2, atom3, atom4 = atoms[p1], atoms[p2], atoms[p3], atoms[p4]
        A, B, C, D = geom[atom1], geom[atom2], geom[atom3], geom[atom4]
        am1,am2,am3,am4 = ams[p1], ams[p2], ams[p3], ams[p4]
        ld1, ld2, ld3, ld4 = am_leading_indices[am1],am_leading_indices[am2],am_leading_indices[am3],am_leading_indices[am4]

        # Compute common intermediates before looping over AM distributions.
        # Avoids redundant recomputations/reassignment for all classes other than (ss|ss).
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
        boys_arg = 0.25*rpq2/delta
        boys_eval = boys(np.arange(13), boys_arg)
        prefactor = 2*jax.lax.pow(np.pi,2.5)/(gamma1*gamma2*np.sqrt(gamma1+gamma2)) \
                     *np.exp(-aa*bb*rab2/gamma1) \
                     *np.exp(-cc*dd*rcd2/gamma2)*coef

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
                i = indices[p1] + s.a
                j = indices[p2] + s.b
                k = indices[p3] + s.c
                l = indices[p4] + s.d
                # Compute the primitive quartet two electron integral value and add to appropriate index in G
                Bx = B_array(la,lb,lc,ld,xp,xa,xb,xq,xc,xd,gamma1,gamma2,delta)
                By = B_array(ma,mb,mc,md,yp,ya,yb,yq,yc,yd,gamma1,gamma2,delta)
                Bz = B_array(na,nb,nc,nd,zp,za,zb,zq,zc,zd,gamma1,gamma2,delta)
                with loops.Scope() as S:
                  S.primitive = 0.
                  S.I = 0
                  S.J = 0
                  S.K = 0
                  for _ in S.while_range(lambda: S.I < la + lb + lc + ld + 1):
                    S.J = 0 
                    for _ in S.while_range(lambda: S.J < ma + mb + mc + md + 1):
                      S.K = 0 
                      for _ in S.while_range(lambda: S.K < na + nb + nc + nd + 1):
                        S.primitive += Bx[S.I] * By[S.J] * Bz[S.K] * boys_eval[S.I + S.J + S.K]
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
#import psi4
#import numpy as onp
#from basis_utils import build_basis_set
#molecule = psi4.geometry("""
#                         0 1
#                         C 0.0 0.0 -0.849220457955
#                         O 0.0 0.0  0.849220457955
#                         units bohr
#                         """)
#geom = np.asarray(onp.asarray(molecule.geometry()))
#basis_name = 'cc-pvdz'
#basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
#basis_dict = build_basis_set(molecule, basis_name)
#G = tei_array(geom, basis_dict)
#mints = psi4.core.MintsHelper(basis_set)
#psi_G = np.asarray(onp.asarray(mints.ao_eri()))
#print("Matches Psi4: ", np.allclose(G, psi_G))
#

