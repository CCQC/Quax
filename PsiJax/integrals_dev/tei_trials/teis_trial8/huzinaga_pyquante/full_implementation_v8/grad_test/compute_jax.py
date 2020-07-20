import psi4
import jax.numpy as np
import jax
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set,homogenize_basisdict
from integrals_utils import cartesian_product, old_cartesian_product, find_unique_shells, am_vectors
from functools import partial
from jax.experimental import loops
from pprint import pprint
from tei import primitive_quartet, gaussian_product, B_array,boys

molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         units bohr
                         """)

# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))

basis_name = 'sto-3g'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
max_prim = basis_set.max_nprimitive()
max_am = basis_set.max_am()
nprim = basis_set.nprimitive()
nbf = basis_set.nbf()
print("Number of basis functions: ", nbf)
print("Number of primitives: ", nprim)

def experiment(geom, basis):
    nshells = len(basis)
    coeffs = []
    exps = []
    atoms = []
    ams = []
    indices = []
    dims = []
    # Smush primitive data together into vectors
    for i in range(nshells):
        tmp_coeffs = basis[i]['coef']  
        tmp_exps = basis[i]['exp']  
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
    # Save various AM distributions for indexing
    angular_momentum = np.array([[0,0,0], 
                                 [1,0,0],                             
                                 [0,1,0],
                                 [0,0,1],
                                 [2,0,0],
                                 [1,1,0],
                                 [1,0,1],
                                 [0,2,0],
                                 [0,1,1],
                                 [0,0,2], 
                                 [3,0,0],
                                 [2,1,0],
                                 [2,0,1],
                                 [1,2,0],
                                 [1,1,1],
                                 [1,0,2],
                                 [0,3,0],
                                 [0,2,1],
                                 [0,1,2],
                                 [0,0,3]])
    leading_indices = np.array([0,1,4,10])
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
        ld1, ld2, ld3, ld4 = leading_indices[am1],leading_indices[am2],leading_indices[am3],leading_indices[am4]

        # compute common intermediates before looping over AM. Avoids recomputations for all classes other than (ss|ss).
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
                #La = angular_momentum[s.a + ld1]
                #Lb = angular_momentum[s.b + ld2]
                #Lc = angular_momentum[s.c + ld3]
                #Ld = angular_momentum[s.d + ld4]
                i = indices[p1] + s.a
                j = indices[p2] + s.b
                k = indices[p3] + s.c
                l = indices[p4] + s.d
                # Compute the primitive
                la, ma, na = angular_momentum[s.a + ld1]
                lb, mb, nb = angular_momentum[s.b + ld2]
                lc, mc, nc = angular_momentum[s.c + ld3]
                ld, md, nd = angular_momentum[s.d + ld4]
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

#G = experiment(geom, basis_dict)
#mints = psi4.core.MintsHelper(basis_set)
#psi_G = np.asarray(onp.asarray(mints.ao_eri()))
#print("Matches Psi4: ", np.allclose(G, psi_G))


#gradG = jax.jacfwd(jax.jacfwd(experiment, 0))(geom,basis_dict)
#print(gradG)
#print(gradG.shape)
#
#print("Indices which are incorrect:")
#problem_idx = onp.vstack(onp.where(~onp.isclose(G, psi_G))).T
#print(problem_idx)
#print(problem_idx.shape)
#
#for idx in problem_idx:
#    i,j,k,l = idx
#    print(G[i,j,k,l],psi_G[i,j,k,l])





