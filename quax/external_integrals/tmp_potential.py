# Temporary potential integrals since libint does allow beyond 2nd order at the moment.
import jax 
from jax.config import config; config.update("jax_enable_x64", True)
config.enable_omnistaging()
import jax.numpy as np
import numpy as onp
from jax.experimental import loops

from ..integrals.integrals_utils import boys, binomial_prefactor, gaussian_product, boys, factorials, double_factorials, neg_one_pow, cartesian_product, am_leading_indices, angular_momentum_combinations
from ..integrals.basis_utils import flatten_basis_data, get_nbf

def A_array(l1,l2,PA,PB,CP,g,A_vals):
    with loops.Scope() as s:
      # Hard code only up to f functions (fxxx | fxxx) => l1 + l2 + 1 = 7
      s.A = A_vals
      s.i = 0
      s.r = 0
      s.u = 0 

      s.i = l1 + l2  
      for _ in s.while_range(lambda: s.i > -1):   
        Aterm = neg_one_pow[s.i] * binomial_prefactor(s.i,l1,l2,PA,PB) * factorials[s.i]
        s.r = s.i // 2
        for _ in s.while_range(lambda: s.r > -1):
          s.u = (s.i - 2 * s.r) // 2 
          for _ in s.while_range(lambda: s.u > -1):
            I = s.i - 2 * s.r - s.u 
            tmp = I - s.u
            fact_ratio = 1 / (factorials[s.r] * factorials[s.u] * factorials[tmp])
            Aterm *= neg_one_pow[s.u]  * CP[tmp] * (0.25 / g)**(s.r+s.u) * fact_ratio 
            s.A = jax.ops.index_add(s.A, I, Aterm)
            s.u -= 1
          s.r -= 1
        s.i -= 1
      return s.A

@jax.jit
def potential(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,Pgeom_pow,boys_eval,prefactor,charges,A_vals):
    """
    Computes a single electron-nuclear attraction integral primitive
    """
    gamma = aa + bb
    prefactor *= -2 * np.pi / gamma

    with loops.Scope() as s:
      s.val = 0.
      for i in s.range(Pgeom_pow.shape[0]):
        Ax = A_array(la,lb,PA_pow[0],PB_pow[0],Pgeom_pow[i,0,:],gamma,A_vals)
        Ay = A_array(ma,mb,PA_pow[1],PB_pow[1],Pgeom_pow[i,1,:],gamma,A_vals)
        Az = A_array(na,nb,PA_pow[2],PB_pow[2],Pgeom_pow[i,2,:],gamma,A_vals)

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
                S.total += Ax[S.I] * Ay[S.J] * Az[S.K] * boys_eval[S.I + S.J + S.K, i]
                S.K += 1
              S.J += 1
            S.I += 1
        s.val += charges[i] * prefactor * S.total
      return s.val

def tmp_potential(geom, basis, charges):
    """
    Build one electron integral arrays (overlap, kinetic, and potential integrals)
    """
    coeffs, exps, atoms, ams, indices, dims = flatten_basis_data(basis)
    nbf = get_nbf(basis)
    nprim = coeffs.shape[0]
    max_am = np.max(ams)
    A_vals = np.zeros(2*max_am+1)

    # Save various AM distributions for indexing
    # Obtain all possible primitive duet index combinations 
    primitive_duets = cartesian_product(np.arange(nprim), np.arange(nprim))

    with loops.Scope() as s:
      s.V = np.zeros((nbf,nbf))
      s.a = 0  # center A angular momentum iterator 
      s.b = 0  # center B angular momentum iterator 

      for prim_duet in s.range(primitive_duets.shape[0]):
        p1,p2 = primitive_duets[prim_duet]
        coef = coeffs[p1] * coeffs[p2]
        aa, bb = exps[p1], exps[p2]
        atom1, atom2 = atoms[p1], atoms[p2]
        am1, am2 = ams[p1], ams[p2]
        A, B = geom[atom1], geom[atom2]
        ld1, ld2 = am_leading_indices[am1], am_leading_indices[am2]

        gamma = aa + bb
        prefactor = np.exp(-aa * bb * np.dot(A-B,A-B) / gamma)
        P = (aa * A + bb * B) / gamma
        # Maximum angular momentum: hard coded
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
        # All needed (and unneeded, for am < max_am) boys function evaluations
        boys_arg = np.broadcast_to(rcp2 * gamma, (2*max_am+1, geom.shape[0]))
        boys_nu = np.tile(np.arange(2*max_am+1), (geom.shape[0],1)).T
        boys_eval = boys(boys_nu,boys_arg)

        s.a = 0
        for _ in s.while_range(lambda: s.a < dims[p1]):
          s.b = 0
          for _ in s.while_range(lambda: s.b < dims[p2]):
            # Gather angular momentum and index
            la,ma,na = angular_momentum_combinations[s.a + ld1]
            lb,mb,nb = angular_momentum_combinations[s.b + ld2]
            # To only create unique indices, need to have separate indices arrays for i and j.
            i = indices[p1] + s.a
            j = indices[p2] + s.b
            # Compute one electron integrals and add to appropriate index
            potential_int = potential(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,Pgeom_pow,boys_eval,prefactor,charges,A_vals) * coef
            s.V = jax.ops.index_add(s.V, jax.ops.index[i,j],  potential_int)

            s.b += 1
          s.a += 1
      return s.V


