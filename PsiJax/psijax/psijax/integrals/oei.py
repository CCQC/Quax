import jax 
from jax.config import config; config.update("jax_enable_x64", True)
config.enable_omnistaging()
import jax.numpy as np
from jax.experimental import loops
from functools import partial

from .integrals_utils import gaussian_product, boys, binomial_prefactor, factorials, double_factorials, cartesian_product, am_leading_indices, angular_momentum_combinations
from .basis_utils import flatten_basis_data, get_nbf

def overlap(aa,La,A,bb,Lb,B):
    """
    Computes a single overlap integral. Taketa, Hunzinaga, Oohata 2.12
    """
    la,ma,na = La 
    lb,mb,nb = Lb 
    rab2 = np.dot(A-B,A-B)
    gamma = aa + bb
    P = gaussian_product(aa,A,bb,B)

    prefactor = (np.pi / gamma)**1.5 * np.exp(-aa * bb * rab2 / gamma)

    wx = overlap_component(la,lb,P[0]-A[0],P[0]-B[0],gamma)
    wy = overlap_component(ma,mb,P[1]-A[1],P[1]-B[1],gamma)
    wz = overlap_component(na,nb,P[2]-A[2],P[2]-B[2],gamma)
    return prefactor * wx * wy * wz

def overlap_component(l1,l2,PAx,PBx,gamma):
    """
    The 1d overlap integral component. Taketa, Hunzinaga, Oohata 2.12
    """
    K = 1 + (l1 + l2) // 2  
    with loops.Scope() as s:
      s.total = 0.
      s.i = 0
      for _ in s.while_range(lambda: s.i < K):
        s.total += binomial_prefactor(2*s.i,l1,l2,PAx,PBx) * double_factorials[2*s.i-1] / (2*gamma)**s.i
        s.i += 1
      return s.total

def kinetic(aa,La,A,bb,Lb,B):
    """
    Computes a single kinetic energy integral.
    """
    la,ma,na = La
    lb,mb,nb = Lb
    term1 = bb*(2*(lb+mb+nb)+3) * overlap(aa,La,A,bb,Lb,B)

    term2 = -2 * bb**2 * (overlap(aa,(la,ma,na), A, bb,(lb+2,mb,nb),B) \
                        + overlap(aa,(la,ma,na),A,bb,(lb,mb+2,nb),B) \
                        + overlap(aa,(la,ma,na),A,bb,(lb,mb,nb+2),B))

    term3 = -0.5 * (lb * (lb-1) * overlap(aa,(la,ma,na),A,bb,(lb-2,mb,nb),B) \
                  + mb * (mb-1) * overlap(aa,(la,ma,na),A,bb,(lb,mb-2,nb),B) \
                  + nb* (nb-1) * overlap(aa,(la,ma,na),A,bb,(lb,mb,nb-2),B))
    return term1 + term2 + term3

def A_term(i,r,u,l1,l2,PAx,PBx,CPx,gamma):
    """
    Taketa, Hunzinaga, Oohata 2.18
    """
    return (-1)**i * binomial_prefactor(i,l1,l2,PAx,PBx) * (-1)**u * factorials[i] * CPx**(i-2*r-2*u) * \
           (0.25 / gamma)**(r+u) / factorials[r] / factorials[u] / factorials[i-2*r-2*u]


def A_array(l1,l2,PA,PB,CP,g):
    with loops.Scope() as s:
      # Hard code only up to f functions (fxxx | fxxx) => l1 + l2 + 1 = 7
      s.A = np.zeros(7)
      s.i = 0
      s.r = 0
      s.u = 0 

      s.i = l1 + l2  
      for _ in s.while_range(lambda: s.i > -1):   
        s.r = s.i // 2
        for _ in s.while_range(lambda: s.r > -1):
          s.u = (s.i - 2 * s.r) // 2 
          for _ in s.while_range(lambda: s.u > -1):
            I = s.i - 2 * s.r - s.u 
            term = A_term(s.i,s.r,s.u,l1,l2,PA,PB,CP,g)
            s.A = jax.ops.index_add(s.A, I, term)
            s.u -= 1
          s.r -= 1
        s.i -= 1
      return s.A

def potential(aa, La, A, bb, Lb, B, geom, charges):
    """
    Computes a single electron-nuclear attraction integral
    """
    la,ma,na = La
    lb,mb,nb = Lb
    gamma = aa + bb
    P = gaussian_product(aa,A,bb,B)
    rab2 = np.dot(A-B,A-B)

    with loops.Scope() as s:
      s.val = 0.
      for i in s.range(geom.shape[0]):
        C = geom[i]
        rcp2 = np.dot(C-P,C-P)
        dPA = P-A
        dPB = P-B
        dPC = P-C

        Ax = A_array(la,lb,dPA[0],dPB[0],dPC[0],gamma)
        Ay = A_array(ma,mb,dPA[1],dPB[1],dPC[1],gamma)
        Az = A_array(na,nb,dPA[2],dPB[2],dPC[2],gamma)

        boys_arg = gamma * rcp2


        with loops.Scope() as S:
          S.total = 0.
          S.I = 0
          S.J = 0
          S.K = 0
          for _ in S.while_range(lambda: S.I < la + lb + 1):
            S.J = 0 
            for _ in S.while_range(lambda: S.J < ma + mb + 1):
              S.K = 0 
              for _ in S.while_range(lambda: S.K < na + nb + 1):
                S.total += Ax[S.I] * Ay[S.J] * Az[S.K] * boys(S.I + S.J + S.K, boys_arg)
                S.K += 1
              S.J += 1
            S.I += 1
        s.val += charges[i] * -2 * np.pi / gamma * np.exp(-aa * bb * rab2 / gamma) * S.total
      return s.val

def oei_arrays(geom, basis, charges):
    """
    Build one electron integral arrays (overlap, kinetic, and potential integrals)
    """
    coeffs, exps, atoms, ams, indices, dims = flatten_basis_data(basis)
    nbf = get_nbf(basis)
    nprim = coeffs.shape[0]

    # Save various AM distributions for indexing
    # Obtain all possible primitive quartet index combinations 
    primitive_duets = cartesian_product(np.arange(nprim), np.arange(nprim))
    with loops.Scope() as s:
      s.S = np.zeros((nbf,nbf))
      s.T = np.zeros((nbf,nbf))
      s.V = np.zeros((nbf,nbf))
      s.a = 0  # center A angular momentum iterator 
      s.b = 0  # center B angular momentum iterator 

      #NOTE this being a scan instead of while loop improves performance
      for prim_duet in s.range(primitive_duets.shape[0]):
        p1,p2 = primitive_duets[prim_duet] 
        coef = coeffs[p1] * coeffs[p2]
        aa, bb = exps[p1], exps[p2]
        atom1, atom2 = atoms[p1], atoms[p2] 
        A, B = geom[atom1], geom[atom2]
        am1, am2 = ams[p1], ams[p2]
        ld1, ld2 = am_leading_indices[am1], am_leading_indices[am2]
        xa,ya,za = A 
        xb,yb,zb = B 

        s.a = 0
        for _ in s.while_range(lambda: s.a < dims[p1]):
          s.b = 0
          for _ in s.while_range(lambda: s.b < dims[p2]):
            # Gather angular momentum and index
            La = angular_momentum_combinations[s.a + ld1]
            Lb = angular_momentum_combinations[s.b + ld2]
            i = indices[p1] + s.a
            j = indices[p2] + s.b
            # Compute one electron integrals and add to appropriate index
            overlap_int = overlap(aa,La,A,bb,Lb,B) * coef
            kinetic_int = kinetic(aa,La,A,bb,Lb,B) * coef
            potential_int = potential(aa,La,A,bb,Lb,B,geom,charges) * coef
            s.S = jax.ops.index_add(s.S, jax.ops.index[i,j], overlap_int) 
            s.T = jax.ops.index_add(s.T, jax.ops.index[i,j], kinetic_int) 
            s.V = jax.ops.index_add(s.V, jax.ops.index[i,j], potential_int) 
            s.b += 1
          s.a += 1
    return s.S,s.T,s.V

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
#charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])
#
#S,T,V = oei_arrays(geom, basis_dict, charges)
#
#mints = psi4.core.MintsHelper(basis_set)
#psi_S = np.asarray(onp.asarray(mints.ao_overlap()))
#psi_T = np.asarray(onp.asarray(mints.ao_kinetic()))
#psi_V = np.asarray(onp.asarray(mints.ao_potential()))
#print("Overlap matches Psi4: ", np.allclose(S, psi_S))
#print("Kinetic matches Psi4: ", np.allclose(T, psi_T))
#print("Potential matches Psi4: ", np.allclose(V, psi_V))





