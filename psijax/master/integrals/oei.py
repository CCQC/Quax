import jax 
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
from jax.experimental import loops
from integrals_utils import gaussian_product, boys, binomial_prefactor, factorial, cartesian_product, am_leading_indices, angular_momentum_combinations
from functools import partial
np.set_printoptions(linewidth=400)

def double_factorial(n):
    '''Given integer, return double factorial n!! = n * (n-2) * (n-4) * ... '''
    with loops.Scope() as s:
      s.k = 1
      s.n = n
      for _ in s.while_range(lambda: s.n > 1):
        s.k *= s.n
        s.n -= 2
      return s.k

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
        s.total += binomial_prefactor(2*s.i,l1,l2,PAx,PBx) * double_factorial(2*s.i-1) / (2*gamma)**s.i
        s.i += 1
      return s.total

def kinetic(aa,La,A,bb,Lb,B):
    """
    Computes a single kinetic energy integral.
    """
    la,ma,na = La
    lb,mb,nb = Lb
    term0 = bb*(2*(lb+mb+nb)+3) * overlap(aa,La,A,bb,Lb,B)
    term1 = -2 * bb*2 * (overlap(aa,(l1,m1,n1),A, bb,(l2+2,m2,n2),B) \
             + overlap(aa,(la,ma,na),A,bb,(lb,mb+2,nb),B) \
             + overlap(aa,(la,ma,na),A,bb,(lb,mb,nb+2),B))

    term2 = -0.5*(l2*(l2-1)*overlap(aa,(la,ma,na),A,bb,(lb-2,mb,nb),B) \
                  + m2*(m2-1)*overlap(aa,(la,ma,na),A,bb,(lb,mb-2,nb),B) +
                  + n2*(n2-1)*overlap(aa,(la,ma,na),A,bb,(lb,mb,nb-2),B))
    return term0+term1+term2



def oei_arrays(geom, basis):
    '''
    Build one electron integral arrays (overlap, kinetic, and potential integrals)
    '''
    nshells = len(basis)
    coeffs = []
    exps = []
    atoms = []
    ams = []
    indices = []
    dims = []
    # Smush primitive data together into vectors
    nbf = 0
    for i in range(nshells):
        tmp_coeffs = basis[i]['coef']  
        tmp_exps = basis[i]['exp']  
        nbf += basis[i]['idx_stride']
        for j in tmp_coeffs:
            coeffs.append(j)
            atoms.append(basis[i]['atom'])
            ams.append(basis[i]['am'])
            indices.append(basis[i]['idx'])
            dims.append(basis[i]['idx_stride'])
        for j in tmp_exps:
            exps.append(j)
    coeffs = np.array(coeffs)
    exps = np.array(exps) 
    atoms = np.array(atoms)
    ams = np.array(ams)
    indices = np.array(indices)
    dims = np.array(dims)
    nprim = coeffs.shape[0]
    primitive_quartets = cartesian_product(np.arange(nprim), np.arange(nprim))

    # Save various AM distributions for indexing
    # Obtain all possible primitive quartet index combinations 
    primitive_duets = cartesian_product(np.arange(nprim), np.arange(nprim))
    with loops.Scope() as s:
      s.S = np.zeros((nbf,nbf))
      s.T = np.zeros((nbf,nbf))
      s.V = np.zeros((nbf,nbf))
      s.a = 0  # center A angular momentum iterator 
      s.b = 0  # center B angular momentum iterator 

      for prim_duet in s.range(primitive_duets.shape[0]):
        p1,p2 = primitive_duets[prim_duet] 
        c1, c2 = coeffs[p1], coeffs[p2]
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
            La = angular_momentum_combinations[s.a + ld1]
            Lb = angular_momentum_combinations[s.b + ld2]
            i = indices[p1] + s.a
            j = indices[p2] + s.b
            
            overlap_int = overlap(aa,La,A,bb,Lb,B) * c1 * c2
            s.S = jax.ops.index_add(s.S, jax.ops.index[i,j], overlap_int) 
            s.b += 1
          s.a += 1

    return s.S
    #return s.S,s.T,s.V

import psi4
import numpy as onp
from basis_utils import build_basis_set
molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         units bohr
                         """)
geom = np.asarray(onp.asarray(molecule.geometry()))
basis_name = 'cc-pvdz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
S = oei_arrays(geom, basis_dict)
mints = psi4.core.MintsHelper(basis_set)
psi_S = np.asarray(onp.asarray(mints.ao_overlap()))
print(S)
print(psi_S)
print("Matches Psi4: ", np.allclose(S, psi_S))


