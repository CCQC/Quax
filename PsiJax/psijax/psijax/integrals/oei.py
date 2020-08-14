import jax 
from jax.config import config; config.update("jax_enable_x64", True)
config.enable_omnistaging()
import jax.numpy as np
from jax.experimental import loops
from functools import partial

from .integrals_utils import gaussian_product, boys, binomial_prefactor, new_binomial_prefactor, factorials, double_factorials, neg_one_pow, cartesian_product, am_leading_indices, angular_momentum_combinations
from .basis_utils import flatten_basis_data, get_nbf

def overlap(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,prefactor):
    """
    Computes a single overlap integral. Taketa, Hunzinaga, Oohata 2.12
    P = gaussian product of aa,A; bb,B
    PA_pow, PB_pow
        All powers of Pi-Ai or Pi-Bi packed into an array
        [[(Px-Ax)^0, (Px-Ax)^1, ... (Px-Ax)^max_am]
         [(Py-Ay)^0, (Py-Ay)^1, ... (Py-Ay)^max_am]
         [(Pz-Az)^0, (Pz-Az)^1, ... (Pz-Az)^max_am]]
    prefactor = np.exp(-aa * bb * np.dot(A-B,A-B) / gamma)
    """
    gamma = aa + bb
    prefactor *= (np.pi / gamma)**1.5

    wx = overlap_component(la,lb,PA_pow[0],PB_pow[0],gamma)
    wy = overlap_component(ma,mb,PA_pow[1],PB_pow[1],gamma)
    wz = overlap_component(na,nb,PA_pow[2],PB_pow[2],gamma)
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
        s.total += new_binomial_prefactor(2*s.i,l1,l2,PAx,PBx) * double_factorials[2*s.i-1] / (2*gamma)**s.i
        s.i += 1
      return s.total

def kinetic(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,prefactor):
    """
    Computes a single kinetic energy integral.
    """
    gamma = aa + bb
    prefactor *= (np.pi / gamma)**1.5
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

def A_array(l1,l2,PA,PB,CP,g):
    with loops.Scope() as s:
      # Hard code only up to f functions (fxxx | fxxx) => l1 + l2 + 1 = 7
      s.A = np.zeros(7)
      s.i = 0
      s.r = 0
      s.u = 0 

      s.i = l1 + l2  
      for _ in s.while_range(lambda: s.i > -1):   
        Aterm = neg_one_pow[s.i] * new_binomial_prefactor(s.i,l1,l2,PA,PB) * factorials[s.i]
        s.r = s.i // 2
        for _ in s.while_range(lambda: s.r > -1):
          s.u = (s.i - 2 * s.r) // 2 
          for _ in s.while_range(lambda: s.u > -1):
            I = s.i - 2 * s.r - s.u 
            tmp = I - s.u
            fact_ratio = 1 / (factorials[s.r] * factorials[s.u] * factorials[tmp])
            #Aterm *= neg_one_pow[s.u]  * CP**(tmp) * (0.25 / g)**(s.r+s.u) * fact_ratio 
            Aterm *= neg_one_pow[s.u]  * CP[tmp] * (0.25 / g)**(s.r+s.u) * fact_ratio 
            s.A = jax.ops.index_add(s.A, I, Aterm)
            s.u -= 1
          s.r -= 1
        s.i -= 1
      return s.A

def potential(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,Pgeom_pow,rcp2,prefactor,charges):
    """
    Computes a single electron-nuclear attraction integral
    """
    gamma = aa + bb
    prefactor *= -2 * np.pi / gamma

    with loops.Scope() as s:
      s.val = 0.
      for i in s.range(Pgeom_pow.shape[0]):
        Ax = A_array(la,lb,PA_pow[0],PB_pow[0],Pgeom_pow[i,0,:],gamma)
        Ay = A_array(ma,mb,PA_pow[1],PB_pow[1],Pgeom_pow[i,1,:],gamma)
        Az = A_array(na,nb,PA_pow[2],PB_pow[2],Pgeom_pow[i,2,:],gamma)

        boys_arg = gamma * rcp2[i]
        boys_eval = boys(np.arange(7), boys_arg) # hard coded for f functions: l1 + l2 + 1

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
                S.total += Ax[S.I] * Ay[S.J] * Az[S.K] * boys_eval[S.I + S.J + S.K]
                S.K += 1
              S.J += 1
            S.I += 1
        s.val += charges[i] * prefactor * S.total
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

        gamma = aa + bb
        prefactor = np.exp(-aa * bb * np.dot(A-B,A-B) / gamma)
        P = (aa * A + bb * B) / gamma
        # Maximum angular momentum: hard coded
        max_am = 3 # f function support
        # Precompute all powers up to 2+max_am of Pi-Ai, Pi-Bi. 
        # We need 2+max_am since kinetic requires incrementing angluar momentum by +2
        PA_pow = np.power(np.broadcast_to(P-A, (max_am+3,3)).T, np.arange(max_am+3))
        PB_pow = np.power(np.broadcast_to(P-B, (max_am+3,3)).T, np.arange(max_am+3))

        # For potential integrals, we need the difference between 
        # the gaussian product center P and ALL atoms in the molecule, 
        # and then take all possible powers up to 2*max_am. 
        # We pre-collect this into a 3d array, and then just pull out what we need via indexing in the loops, so they need not be recomputed.
        # The resulting array has dimensions (atom, cartesian component, power) so index (0, 1, 3) would return (Py - atom0_y)^3
        P_minus_geom = np.broadcast_to(P, geom.shape) - geom
        Pgeom_pow = np.power(np.transpose(np.broadcast_to(P_minus_geom, (2*max_am + 1,geom.shape[0],geom.shape[1])), (1,2,0)), np.arange(2*max_am + 1))
        # All possible np.dot(P-atom,P-atom) 
        rcp2 = np.einsum('ij,ij->i', P_minus_geom, P_minus_geom)

        s.a = 0
        for _ in s.while_range(lambda: s.a < dims[p1]):
          s.b = 0
          for _ in s.while_range(lambda: s.b < dims[p2]):
            # Gather angular momentum and index
            la,ma,na = angular_momentum_combinations[s.a + ld1]
            lb,mb,nb = angular_momentum_combinations[s.b + ld2]
            i = indices[p1] + s.a
            j = indices[p2] + s.b
            # Compute one electron integrals and add to appropriate index
            overlap_int = overlap(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,prefactor) * coef
            kinetic_int = kinetic(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,prefactor) * coef
            potential_int = potential(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,Pgeom_pow,rcp2,prefactor,charges) * coef

            s.S = jax.ops.index_add(s.S, jax.ops.index[i,j], overlap_int) 
            s.T = jax.ops.index_add(s.T, jax.ops.index[i,j], kinetic_int) 
            s.V = jax.ops.index_add(s.V, jax.ops.index[i,j], potential_int) 
            s.b += 1
          s.a += 1
    return s.S,s.T,s.V

#import psi4
#import numpy as onp
#from basis_utils import build_basis_set,flatten_basis_data,get_nbf
#from integrals_utils import gaussian_product, boys, new_binomial_prefactor, binomial_prefactor, factorials, double_factorials, neg_one_pow, cartesian_product, am_leading_indices, angular_momentum_combinations
#molecule = psi4.geometry("""
#                         0 1
#                         N 0.0 0.0 -0.849220457955
#                         N 0.0 0.0  0.849220457955
#                         units bohr
#                         """)
#geom = np.asarray(onp.asarray(molecule.geometry()))
#basis_name = 'cc-pvtz'
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





