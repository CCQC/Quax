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

#def tmp_potential(geom, basis, charges):
#    """
#    Build one electron integral arrays (overlap, kinetic, and potential integrals)
#    """
#    coeffs, exps, atoms, ams, indices, dims = flatten_basis_data(basis)
#    print(indices)
#    nbf = get_nbf(basis)
#    nprim = coeffs.shape[0]
#    max_am = np.max(ams)
#    A_vals = np.zeros(2*max_am+1)
#
#    # Save various AM distributions for indexing
#    # Obtain all possible primitive quartet index combinations 
#    primitive_duets = cartesian_product(np.arange(nprim), np.arange(nprim))
#
#    # Shapes and constant arrays for jitted prepare_constants function
#    # so they are compile-time static
#    arr1 = np.arange(max_am + 3)
#    arr2 = np.arange(2*max_am + 1)
#    shape1 = (max_am + 3, 3)
#    shape2 = (2*max_am + 1,geom.shape[0],geom.shape[1])
#    shape3 = (2*max_am+1, geom.shape[0])
#
#    # Function for pre-evaluating primitive data
#    @jax.jit
#    def prepare_constants(A,B,aa,bb):
#        gamma = aa + bb
#        prefactor = np.exp(-aa * bb * np.dot(A-B,A-B) / gamma)
#        P = (aa * A + bb * B) / gamma
#        PA_pow = np.power(np.broadcast_to(P-A, shape1).T, arr1)
#        PB_pow = np.power(np.broadcast_to(P-B, shape1).T, arr1)
#        P_minus_geom = np.broadcast_to(P, geom.shape) - geom
#        Pgeom_pow = np.power(np.transpose(np.broadcast_to(P_minus_geom, shape2), (1,2,0)), arr2)
#        rcp2 = np.einsum('ij,ij->i', P_minus_geom, P_minus_geom)
#        boys_arg = np.broadcast_to(rcp2 * gamma, shape3)
#        boys_nu = np.tile(arr2, (geom.shape[0],1)).T
#        boys_eval = boys(boys_nu,boys_arg)
#        return PA_pow, PB_pow, Pgeom_pow, boys_eval, prefactor
#
#
#    with loops.Scope() as s:
#      s.V = np.zeros((nbf,nbf))
#      s.a = 0  # center A angular momentum iterator 
#      s.b = 0  # center B angular momentum iterator 
#      s.p1 = 0 # primitive index 1
#      s.p2 = 0 # primitive index 1
#
#      #for prim_duet in s.range(primitive_duets.shape[0]):
#        # Only compute this primitive duet if indices lie in upper triangle:
#        #p1,p2 = primitive_duets[prim_duet]
#        #for _ in s.cond_range(indices[p2] <= indices[p1]):
#      for _ in s.while_range(lambda: s.p1 < nprim):
#        s.p2 = 0
#        for _ in s.while_range(lambda: s.p2 <= s.p1):
#          coef = coeffs[s.p1] * coeffs[s.p2]
#          aa, bb = exps[s.p1], exps[s.p2]
#          atom1, atom2 = atoms[s.p1], atoms[s.p2]
#          am1, am2 = ams[s.p1], ams[s.p2]
#          A, B = geom[atom1], geom[atom2]
#          ld1, ld2 = am_leading_indices[am1], am_leading_indices[am2]
#
#          PA_pow, PB_pow, Pgeom_pow, boys_eval, prefactor = prepare_constants(A,B,aa,bb)
#
#          s.a = 0
#          for _ in s.while_range(lambda: s.a < dims[s.p1]):
#            s.b = 0
#            for _ in s.while_range(lambda: s.b < dims[s.p2]):
#              # Gather angular momentum and index
#              la,ma,na = angular_momentum_combinations[s.a + ld1]
#              lb,mb,nb = angular_momentum_combinations[s.b + ld2]
#              # To only create unique indices, need to have separate indices arrays for i and j.
#              i = indices[s.p1] + s.a
#              j = indices[s.p2] + s.b
#              # Compute one electron integrals and add to appropriate index
#              potential_int = potential(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,Pgeom_pow,boys_eval,prefactor,charges,A_vals) * coef
#
#              # Add to both upper and lower triangle only if not same primary index  
#              for _ in s.cond_range(indices[s.p1] != indices[s.p2]):
#                s.V = jax.ops.index_add(s.V, jax.ops.index[j,i],  potential_int)
#              s.V = jax.ops.index_add(s.V, jax.ops.index[i,j],  potential_int)
#              s.b += 1
#            s.a += 1
#          s.p2 +=1
#        s.p1 +=1
#      return s.V

# This one works great, until you want higher order derivs with a lot of BF's (>50)
#def tmp_potential(geom, basis, charges):
#    """
#    Computes nuclear-electron attraction potential integrals
#    """
#    nbf = get_nbf(basis)
#    nshells = len(basis)
#
#    # Only needed to get max am
#    coeffs, exps, atoms, ams, indices, dims = flatten_basis_data(basis)
#    max_am = np.max(ams)
#    A_vals = np.zeros(2*max_am+1)
#
#    # Shapes and constant arrays for jitted prepare_constants function
#    # so they are compile-time static
#    arr1 = np.arange(max_am + 3)
#    arr2 = np.arange(2*max_am + 1)
#    shape1 = (max_am + 3, 3)
#    shape2 = (2*max_am + 1,geom.shape[0],geom.shape[1])
#    shape3 = (2*max_am+1, geom.shape[0])
#
#    # Function for pre-evaluating primitive data
#    @jax.jit
#    def prepare_constants(A,B,aa,bb):
#        gamma = aa + bb
#        prefactor = np.exp(-aa * bb * np.dot(A-B,A-B) / gamma)
#        P = (aa * A + bb * B) / gamma
#        PA_pow = np.power(np.broadcast_to(P-A, shape1).T, arr1)
#        PB_pow = np.power(np.broadcast_to(P-B, shape1).T, arr1)
#        P_minus_geom = np.broadcast_to(P, geom.shape) - geom
#        Pgeom_pow = np.power(np.transpose(np.broadcast_to(P_minus_geom, shape2), (1,2,0)), arr2)
#        rcp2 = np.einsum('ij,ij->i', P_minus_geom, P_minus_geom)
#        boys_arg = np.broadcast_to(rcp2 * gamma, shape3)
#        boys_nu = np.tile(arr2, (geom.shape[0],1)).T
#        boys_eval = boys(boys_nu,boys_arg)
#        return PA_pow, PB_pow, Pgeom_pow, boys_eval, prefactor
#
#    V = np.zeros((nbf,nbf))
#    # These loops are quite fast, despite being Python loops, since they only index objects and run Jit-compiled functions
#    for i in range(nshells):
#        atom1 = basis[i]['atom']
#        A = geom[atom1]               # XYZ coordinates of center of this shell
#        exp1 = basis[i]['exp']        # Orbital exponents of this shell
#        coef1 = basis[i]['coef']      # Contraction coefficients of this shell
#        nprim1 = len(coef1)           # Number of primitives in this shell  
#        dim1 = basis[i]['idx_stride'] # Number of angular momentum components in this shell
#        am1 = basis[i]['am']          # Total angular momentum of this shell 
#        bf1 = basis[i]['idx']         # Base row index
#        ld1 = am_leading_indices[am1] # Angular momentum combinations lookup array index.
#        for j in range(i + 1):
#            atom2 = basis[j]['atom']
#            B = geom[atom2]                # XYZ coordinates of center of this shell
#            exp2 = basis[j]['exp']         # Orbital exponents of this shell
#            coef2 = basis[j]['coef']       # Contraction coefficients of this shell
#            nprim2 = len(coef2)            # Number of primitives in this shell
#            dim2 = basis[j]['idx_stride']  # Number of angular momentum components in this shell
#            am2 = basis[j]['am']           # Total angular momentum of this shell 
#            bf2 = basis[j]['idx']          # Base row index
#            ld2 = am_leading_indices[am2]  # Angular momentum combinations lookup array index.
#            # Contraction loop
#            for p in range(nprim1):
#                for q in range(nprim2):
#                    aa = exp1[p]
#                    bb = exp2[q]
#                    coef = coef1[p] * coef2[q]
#                    PA_pow, PB_pow, Pgeom_pow, boys_eval, prefactor = prepare_constants(A,B,aa,bb)
#                    # Angular momentum distribution loop
#                    for r in range(dim1):
#                        for s in range(dim2):
#                            la,ma,na = angular_momentum_combinations[r + ld1]
#                            lb,mb,nb = angular_momentum_combinations[s + ld2]
#                            potential_int = potential(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,Pgeom_pow,boys_eval,prefactor,charges,A_vals) * coef
#                            idx1 = bf1 + r
#                            idx2 = bf2 + s
#                            if i == j:
#                                V = jax.ops.index_add(V, jax.ops.index[idx1, idx2], potential_int)
#                            else:
#                                V = jax.ops.index_add(V, [[idx1, idx2], [idx2, idx1]], potential_int)
#    return V 

#
#import psi4
#import numpy as onp
#from basis_utils import build_basis_set,flatten_basis_data,get_nbf
#from integrals_utils import gaussian_product, boys, binomial_prefactor, factorials, double_factorials, neg_one_pow, cartesian_product, am_leading_indices, angular_momentum_combinations
#from pprint import pprint
#molecule = psi4.geometry("""
#                         0 1
#                         H 0.0 0.0 -0.849220457955
#                         H 0.0 0.0  0.849220457955
#                         units bohr
#                         """)
#geom = np.asarray(onp.asarray(molecule.geometry()))
#basis_name = 'cc-pvdz'
#basis_name = 'cc-pvtz'
#basis_name = 'cc-pvqz'
#basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
#basis_dict = build_basis_set(molecule, basis_name)
#pprint(basis_dict)

#charges = np.asarray([molecule.charge(i) for i in range(geom.shape[0])])
#V = tmp_potential(geom, basis_dict, charges)

#V_hess = jax.jacfwd(jax.jacfwd(tmp_potential))(geom,basis_dict,charges)

#V_quar = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(tmp_potential))))(geom,basis_dict,charges)

#mints = psi4.core.MintsHelper(basis_set)
#psi_V = np.asarray(onp.asarray(mints.ao_potential()))
#print(np.round(psi_V,3))
#print("Potential matches Psi4: ", np.allclose(V, psi_V))




