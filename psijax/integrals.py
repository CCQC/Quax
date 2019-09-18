import jax
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)

np.set_printoptions(linewidth=400)

def vectorized_oei(geom, basis, nbf_per_atom, charge_per_atom):
    '''Computes overlap, kinetic, and potential energy integrals over s orbital basis functions
       Parameters
       ----------
       basis : torch.tensor() of shape (n,), where n is number of basis functions
            A vector of orbital exponents in the same order as the atom order in 'geom'.
            That is, you must concatentate the basis functions of each atom together before passing to this function.
       geom : torch.tensor() of shape (N x 3), where N is number of atoms
            The cartesian coordinates 
       nbf_per_atom: torch.tensor() of shape (N,)
            The number of basis functions for each atom (so we know which center (cartesian coordinate) goes with which basis function)
       charge_per_atom: torch.tensor() of shape (N,)
            The charge of the nucleus for each atom.
       NOTE: In the future you should just make a single argument which contains all orbital exponents and their corresponding centers pre-prepared
    '''
    # SETUP AND OVERLAP INTEGRALS
    nbf = basis.shape[0]
    # 'centers' are the cartesian centers ((nbf,3) array) corresponding to each basis function, in the same order as the 'basis' vector
    #TODO
    centers = np.repeat(geom, 4, axis=0).reshape(-1,3) # TODO currently can only repeat each center the same number of times => only works for when all atoms have same # of basis functions
    # Construct Normalization constant product array, Na * Nb component
    norm = (2 * basis / np.pi)**(3/4)
    normtensor = np.outer(norm,norm) # outer product => every possible combination of Na * Nb
    # Construct pi / aa + bb ** 3/2 term
    aa_times_bb = np.outer(basis,basis)
    #aa_plus_bb = basis.expand(nbf,-1) + torch.transpose(basis.expand(nbf,-1),0,1) # doesnt copy data, unlike repeat(). may not work, but very efficient
    aa_plus_bb = np.broadcast_to(basis, (nbf,nbf)) + np.transpose(np.broadcast_to(basis, (nbf,nbf)), (1,0))
    term = (np.pi / aa_plus_bb) ** (3/2)
    ## Construct gaussian product coefficient array, c = exp(A-B dot A-B) * ((-aa * bb) / (aa + bb))
    tmpA = np.broadcast_to(centers, (nbf,nbf,3))
    AminusB = tmpA - np.transpose(tmpA, (1,0,2)) #caution: tranpose shares memory with original array. changing one changes the other
    AmBAmB = np.einsum('ijk,ijk->ij', AminusB, AminusB)
    coeff = np.exp(AmBAmB * (-aa_times_bb / aa_plus_bb))
    S = normtensor * coeff * term
    # KINETIC INTEGRALS
    P = aa_times_bb / aa_plus_bb
    T = S * (3 * P + 2 * P * P * -AmBAmB)
    # Construct gaussian product center array, R = (aa * A + bb * B) / (aa + bb)
    # First construct every possible sum of exponential-weighted cartesian centers, aa*A + bb*B 
    aatimesA = np.einsum('i,ij->ij', basis,centers)
    # This is a 3D tensor (nbf,nbf,3), where each row is a unique sum of two exponent-weighted cartesian centers
    numerator = aatimesA[:,None,:] + aatimesA[None,:,:]
    R = np.einsum('ijk,ij->ijk', numerator, 1/aa_plus_bb)
    ## Now we must subtract off the atomic coordinates, for each atom, introducing yet another dimension, where we expand according to number of atoms
    R_per_atom = np.broadcast_to(R, (geom.shape[0],) + R.shape)
    expanded_geom = np.transpose(np.broadcast_to(geom, (nbf,nbf) + geom.shape), (2,1,0,3))
    # Subtract off atom coordinates
    Rminusgeom = R_per_atom - expanded_geom
    # Now contract along the coordinate dimension, and weight by aa_plus_bb. This is the boys function argument.
    ##arg = (aa+bb) * np.dot(R - atom, R - atom)
    contracted = np.einsum('ijkl,ijkl->ijk', Rminusgeom,Rminusgeom)
    boys_arg = np.einsum('ijk,jk->ijk', contracted, aa_plus_bb)
    # Now evaluate the boys function on all elements, multiply by charges, and then sum the atom dimension
    # it is safe to sum here, since every other operation in the integral expression is linear
    ##F = boys(torch.tensor(0.0), boys_arg)
    F = np.zeros_like(boys_arg)
    mask1 = boys_arg <= 1e-8
    mask2 = boys_arg > 1e-8
    F = jax.ops.index_update(F,mask1, 1 - boys_arg[mask1] / 3)
    F = jax.ops.index_update(F,mask2, jax.scipy.special.erf(np.sqrt(boys_arg[mask2])) * np.sqrt(np.pi) / (2 * np.sqrt(boys_arg[mask2])))
    #F[mask1] = 1 - boys_arg[mask1] / 3 
    #F[mask2] = np.erf(np.sqrt(boys_arg[mask2])) * np.sqrt(np.pi) / (2 * np.sqrt(boys_arg[mask2]))
    Fcharge = -charge_per_atom[:,None,None] * F[:,...]
    Ffinal = np.sum(Fcharge, axis=0)
    V = Ffinal * normtensor * coeff * 2 * np.pi / aa_plus_bb
    return S, T, V

def vectorized_tei(geom, basis, nbf_per_atom):
    '''Computes two electron integrals over s orbital basis functions
       Parameters
       ----------
       geom : torch.tensor() of shape (N x 3), where N is number of atoms
            The cartesian coordinates in the same order as the concatenated basis functions 
       basis : torch.tensor() of shape (n,), where n is number of basis functions
            A vector of orbital exponents in the same order as the atom order in 'geom'.
            That is, you must concatentate the basis functions of each atom together before passing to this function.
       nbf_per_atom: torch.tensor() of shape (N,)
            The number of basis functions for each atom (so we know which center (cartesian coordinate) goes with which basis function)
    '''
    nbf = basis.shape[0]
    # 'centers' are the cartesian centers ((nbf,3) array) corresponding to each basis function, in the same order as the 'basis' vector
    #TODO
    centers = np.repeat(geom, 4, axis=0).reshape(-1,3) # TODO currently can only repeat each center the same number of times => only works for when all atoms have same # of basis functions
    # Construct every combination of normalization constants 
    norm = (2 * basis / np.pi)**(3/4)
    normtensor = np.einsum('i,j,k,l',norm,norm,norm,norm)
    # Obtain miscellaneous terms 
    # (i,l,j,k) + (l,i,j,k) ---> (i+l,i+l,j+j,k+k) ---> (A+D,D+A,C+C,B+B) which is just (A+D,A+D,C+C,B+B)
    tmp1 = np.broadcast_to(basis, (nbf,nbf,nbf,nbf))
    aa_plus_bb = tmp1.transpose((0,3,1,2)) + tmp1.transpose((3,0,1,2))
    cc_plus_dd = aa_plus_bb.transpose((2,3,0,1))
    aa_times_bb = tmp1.transpose((0,3,1,2)) * tmp1.transpose((3,0,1,2))
    cc_times_dd = aa_times_bb.transpose((2,3,0,1))
    # Obtain gaussian product coefficients
    tmp2 = np.broadcast_to(centers, (nbf,nbf,nbf,nbf,3))
    AminusB = tmp2.transpose((0,3,1,2,4)) - tmp2.transpose((3,0,1,2,4))
    CminusD = AminusB.transpose((2,3,0,1,4))
    # 'dot' the cartesian dimension
    contract_AminusB = np.einsum('ijklm,ijklm->ijkl', AminusB,AminusB)
    c1 = np.exp(contract_AminusB * -aa_times_bb / aa_plus_bb)
    contract_CminusD = np.einsum('ijklm,ijklm->ijkl', CminusD,CminusD)
    c2 = np.exp(contract_CminusD * -cc_times_dd / cc_plus_dd)
    # Obtain gaussian product centers Rp = (aa * A + bb * B) / (aa + bb);  Rq = (cc * C + dd * D) / (cc + dd)
    weighted_centers = np.einsum('ijkl,ijklm->ijklm', tmp1, tmp2)
    tmpAB = weighted_centers.transpose((0,3,1,2,4)) + weighted_centers.transpose((3,0,1,2,4))
    tmpCD = tmpAB.transpose((2,3,0,1,4))
    Rp = np.einsum('ijklm,ijkl->ijklm', tmpAB, 1/aa_plus_bb)
    Rq = np.einsum('ijklm,ijkl->ijklm', tmpCD, 1/cc_plus_dd)
    delta = 1 / (4 * aa_plus_bb) + 1 / (4 * cc_plus_dd)
    boys_arg = np.einsum('ijklm,ijklm->ijkl', Rp-Rq, Rp-Rq) / (4 * delta)
    F = np.zeros_like(boys_arg)
    mask1 = boys_arg <= 1e-8
    mask2 = boys_arg > 1e-8
    F = jax.ops.index_update(F,mask1, 1 - boys_arg[mask1] / 3)
    F = jax.ops.index_update(F,mask2, jax.scipy.special.erf(np.sqrt(boys_arg[mask2])) * np.sqrt(np.pi) / (2 * np.sqrt(boys_arg[mask2])))
    G = F * c1 * c2 * normtensor * 2 * np.pi**2 / (aa_plus_bb * cc_plus_dd) * np.sqrt(np.pi / (aa_plus_bb + cc_plus_dd))
    return G

def nuclear_repulsion(atom1, atom2):
    ''' warning : hard coded for H2'''
    Za = 1.0
    Zb = 1.0
    return Za*Zb / np.linalg.norm(atom1-atom2)

def orthogonalizer(S):
    '''Compute overlap to the negative 1/2 power'''
    eigval, eigvec = np.linalg.eigh(S)
    cutoff = 1.0e-12
    above_cutoff = (abs(eigval) > cutoff * np.max(abs(eigval)))
    val = 1 / np.sqrt(eigval[above_cutoff])
    vec = eigvec[:, above_cutoff]
    A = vec.dot(np.diag(val)).dot(vec.T)
    return A

geom = np.array([0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]).reshape(-1,3)
basis = np.array([0.5, 0.4, 0.3, 0.2])
full_basis = np.concatenate((basis,basis))
nbf_per_atom = np.array([basis.shape[0],basis.shape[0]])
charge_per_atom = np.array([1.0,1.0])

#S, T, V = vectorized_oei(geom, full_basis, nbf_per_atom, charge_per_atom)
#A = orthogonalizer(S)
#Enuc = nuclear_repulsion(geom[0],geom[1])

G = vectorized_tei(geom,full_basis, nbf_per_atom)



