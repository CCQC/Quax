import torch
import numpy as np
import math
torch.set_printoptions(precision=12,linewidth=300)
torch.set_default_dtype(torch.float64)
ang2bohr = 1 / 0.52917720859

@torch.jit.script
def gp(aa,bb,A,B):
    '''Gaussian product theorem. Returns center and coefficient of product'''
    R = (aa * A + bb * B) / (aa + bb)
    c = torch.exp(torch.dot(A-B,A-B) * (-aa * bb / (aa + bb)))
    return R,c

@torch.jit.script
def normalize(aa):
    '''Normalization constant for s primitive basis functions. Argument is orbital exponent coefficient'''
    N = (2*aa/math.pi)**(3/4)
    return N

@torch.jit.script
def torchboys(nu, arg):
    '''Pytorch can only exactly compute F0 boys function using the error function relation, the rest would have to
    be determined recursively'''
    if arg < torch.tensor(1e-8):
        boys =  1 / (2 * nu + 1) - arg / (2 * nu + 3)
    else:
        boys = torch.erf(torch.sqrt(arg)) * math.sqrt(math.pi) / (2 * torch.sqrt(arg))
    return boys

@torch.jit.script
def boys(nu, arg):
    '''Alternative boys function expansion. Not exact.'''
    boys = 0.5 * torch.exp(-arg) * (1 / (nu + 0.5)) * (1 + (arg / (nu+1.5)) *\
                                                          (1 + (arg / (nu+2.5)) *\
                                                          (1 + (arg / (nu+3.5)) *\
                                                          (1 + (arg / (nu+4.5)) *\
                                                          (1 + (arg / (nu+5.5)) *\
                                                          (1 + (arg / (nu+6.5)) *\
                                                          (1 + (arg / (nu+7.5)) *\
                                                          (1 + (arg / (nu+8.5)) *\
                                                          (1 + (arg / (nu+9.5)) *\
                                                          (1 + (arg / (nu+10.5))*\
                                                          (1 + (arg / (nu+11.5)))))))))))))
    return boys


@torch.jit.script
def vectorized_oei(basis, geom, nbf_per_atom, charge_per_atom):
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
    nbf = torch.numel(basis)
    # 'centers' are the cartesian centers ((nbf,3) array) corresponding to each basis function, in the same order as the 'basis' vector
    centers = geom.repeat_interleave(nbf_per_atom, dim=0).reshape(-1,3)
    # Construct Normalization constant product array, Na * Nb component
    norm = (2 * basis / math.pi)**(3/4)
    normtensor = torch.ger(norm,norm) # outer product => every possible combination of Na * Nb
    # Construct pi / aa + bb ** 3/2 term
    aa_times_bb = torch.ger(basis,basis)
    aa_plus_bb = basis.expand(nbf,-1) + torch.transpose(basis.expand(nbf,-1),0,1) # doesnt copy data, unlike repeat(). may not work, but very efficient
    term = (math.pi / aa_plus_bb) ** (3/2)
    # Construct gaussian product coefficient array, c = exp(A-B dot A-B) * ((-aa * bb) / (aa + bb))
    tmpA = centers.expand(nbf,nbf,3)
    AminusB = tmpA - torch.transpose(tmpA, 0,1) #caution: tranpose shares memory with original array. changing one changes the other
    AmBAmB = torch.einsum('ijk,ijk->ij', AminusB, AminusB)
    coeff = torch.exp(AmBAmB * (-aa_times_bb / aa_plus_bb))
    S = normtensor * coeff * term
    # KINETIC INTEGRALS
    P = aa_times_bb / aa_plus_bb
    T = S * (3 * P + 2 * P * P * -AmBAmB)
    # Construct gaussian product center array, R = (aa * A + bb * B) / (aa + bb)
    # First construct every possible sum of exponential-weighted cartesian centers, aa*A + bb*B 
    aatimesA = torch.einsum('i,ij->ij', basis,centers)
    # This is a 3D tensor (nbf,nbf,3), where each row is a unique sum of two exponent-weighted cartesian centers
    numerator = aatimesA[:,None,:] + aatimesA[None,:,:]
    R = torch.einsum('ijk,ij->ijk', numerator, 1/aa_plus_bb)
    # Now we must subtract off the atomic coordinates, for each atom, introducing yet another dimension, where we expand according to number of atoms
    R_per_atom = R.expand(geom.size()[0],-1,-1,-1)
    expanded_geom = torch.transpose(geom.expand(nbf,nbf,-1,-1), 0,2)
    # Subtract off atom coordinates
    Rminusgeom = R_per_atom - expanded_geom
    # Now contract along the coordinate dimension, and weight by aa_plus_bb. This is the boys function argument.
    # arg = (aa+bb) * torch.dot(R - atom, R - atom)
    contracted = torch.einsum('ijkl,ijkl->ijk', Rminusgeom,Rminusgeom)
    boys_arg = torch.einsum('ijk,jk->ijk', contracted, aa_plus_bb)
    # Now evaluate the boys function on all elements, multiply by charges, and then sum the atom dimension
    # it is safe to sum here, since every other operation in the integral expression is linear
    #F = boys(torch.tensor(0.0), boys_arg)
    F = torch.zeros_like(boys_arg)
    mask1 = boys_arg <= 1e-8
    mask2 = boys_arg > 1e-8
    F[mask1] = 1 - boys_arg[mask1] / 3 
    F[mask2] = torch.erf(torch.sqrt(boys_arg[mask2])) * math.sqrt(math.pi) / (2 * torch.sqrt(boys_arg[mask2]))

    Fcharge = -charge_per_atom[:,None,None] * F[:,...]
    Ffinal = torch.sum(Fcharge, dim=0)
    V = Ffinal * normtensor * coeff * 2 * math.pi / aa_plus_bb
    return S, T, V

@torch.jit.script
def vectorized_tei(basis, geom, nbf_per_atom):
    '''Computes two electron integrals over s orbital basis functions
       Parameters
       ----------
       basis : torch.tensor() of shape (n,), where n is number of basis functions
            A vector of orbital exponents in the same order as the atom order in 'geom'.
            That is, you must concatentate the basis functions of each atom together before passing to this function.
       geom : torch.tensor() of shape (N x 3), where N is number of atoms
            The cartesian coordinates in the same order as the concatenated basis functions 
       nbf_per_atom: torch.tensor() of shape (N,)
            The number of basis functions for each atom (so we know which center (cartesian coordinate) goes with which basis function)
    '''
    nbf = torch.numel(basis)
    # 'centers' are the cartesian centers ((nbf,3) array) corresponding to each basis function, in the same order as the 'basis' vector
    centers = geom.repeat_interleave(nbf_per_atom, dim=0).reshape(-1,3)
    # Construct every combination of normalization constants 
    norm = (2 * basis / math.pi)**(3/4)
    normtensor = torch.einsum('i,j,k,l',norm,norm,norm,norm) 
    # Obtain miscellaneous terms 
    # (i,l,j,k) + (l,i,j,k) ---> (i+l,i+l,j+j,k+k) ---> (A+D,D+A,C+C,B+B) which is just (A+D,A+D,C+C,B+B)
    tmp1 = basis.expand(nbf,nbf,nbf,-1)
    aa_plus_bb = tmp1.permute(0,3,1,2) + tmp1.permute(3,0,1,2)
    cc_plus_dd = aa_plus_bb.permute(2,3,0,1)
    aa_times_bb = tmp1.permute(0,3,1,2) * tmp1.permute(3,0,1,2)
    cc_times_dd = aa_times_bb.permute(2,3,0,1)  
    # Obtain gaussian product coefficients
    tmp2 = centers.expand(nbf,nbf,nbf,nbf,3)
    AminusB = tmp2.permute(0,3,1,2,4) - tmp2.permute(3,0,1,2,4) 
    CminusD = AminusB.permute(2,3,0,1,4)
    # 'dot' the cartesian dimension
    contract_AminusB = torch.einsum('ijklm,ijklm->ijkl', AminusB,AminusB)
    c1 = torch.exp(contract_AminusB * -aa_times_bb / aa_plus_bb)
    contract_CminusD = torch.einsum('ijklm,ijklm->ijkl', CminusD,CminusD)
    c2 = torch.exp(contract_CminusD * -cc_times_dd / cc_plus_dd)
    # Obtain gaussian product centers Rp = (aa * A + bb * B) / (aa + bb);  Rq = (cc * C + dd * D) / (cc + dd)
    weighted_centers = torch.einsum('ijkl,ijklm->ijklm', tmp1, tmp2)
    #del tmp1; del tmp2
    tmpAB = weighted_centers.permute(0,3,1,2,4) + weighted_centers.permute(3,0,1,2,4) 
    tmpCD = tmpAB.permute(2,3,0,1,4)                                                  
    Rp = torch.einsum('ijklm,ijkl->ijklm', tmpAB, 1/aa_plus_bb) 
    Rq = torch.einsum('ijklm,ijkl->ijklm', tmpCD, 1/cc_plus_dd) 
    delta = 1 / (4 * aa_plus_bb) + 1 / (4 * cc_plus_dd)
    boys_arg = torch.einsum('ijklm,ijklm->ijkl', Rp-Rq, Rp-Rq) / (4 * delta)
    #F = boys(torch.tensor(0.0), boys_arg)
    F = torch.zeros_like(boys_arg)
    mask1 = boys_arg <= 1e-8
    mask2 = boys_arg > 1e-8
    F[mask1] = 1 - boys_arg[mask1] / 3 
    F[mask2] = torch.erf(torch.sqrt(boys_arg[mask2])) * math.sqrt(math.pi) / (2 * torch.sqrt(boys_arg[mask2]))
    G = F * c1 * c2 * normtensor * 2 * math.pi**2 / (aa_plus_bb * cc_plus_dd) * torch.sqrt(math.pi / (aa_plus_bb + cc_plus_dd))
    return G

# Naive (non-vectorized) implementation
@torch.jit.script
def overlap(aa, bb, A, B):
    '''Computes a single overlap integral over two primitive s-orbital basis functions'''
    Na = normalize(aa)
    Nb = normalize(bb)
    R,c = gp(aa,bb,A,B)
    S = Na * Nb * c * (math.pi / (aa + bb)) ** (3/2)
    return S

@torch.jit.script
def kinetic(aa,bb,A,B):
    '''Computes a single kinetic energy integral over two primitive s-orbital basis functions'''
    P = (aa * bb) / (aa + bb)
    ab = -1.0 * torch.dot(A-B, A-B)
    K = overlap(aa,bb,A,B) * (3 * P + 2 * P * P * ab)
    return K

@torch.jit.script
def potential(aa,bb,A,B,atom,charge):
    '''Computes a single electron-nuclear potential energy integral over two primitive s-orbital basis functions'''
    g = aa + bb
    eps = 1 / (4 * g)
    P, c = gp(aa,bb,A,B)
    arg = g * torch.dot(P - atom, P - atom)
    Na = normalize(aa)
    Nb = normalize(bb)
    F = torchboys(torch.tensor(0.0), arg)
    #print(arg,F[0], boys(torch.tensor(0.0), arg)[0])
    V = -charge * F * Na * Nb * c * 2 * math.pi / g
    return V

@torch.jit.script
def eri(aa,bb,cc,dd,A,B,C,D):
    '''Computes a single two electron integral over 4 s-orbital basis functions on 4 centers'''
    g1 = aa + bb
    g2 = cc + dd
    Rp = (aa * A + bb * B) / (aa + bb)
    tmpc1 = torch.dot(A-B, A-B) * ((-aa * bb) / (aa + bb))
    c1 = torch.exp(tmpc1)
    Rq = (cc * C + dd * D) / (cc + dd)
    tmpc2 = torch.dot(C-D, C-D) * ((-cc * dd) / (cc + dd))
    c2 = torch.exp(tmpc2)

    Na, Nb, Nc, Nd = normalize(aa), normalize(bb), normalize(cc), normalize(dd)
    delta = 1 / (4 * g1) + 1 / (4 * g2)
    arg = torch.dot(Rp - Rq, Rp - Rq) / (4 * delta)
    F = torchboys(torch.tensor(0.0), arg)
    G = F * Na * Nb * Nc * Nd * c1 * c2 * 2 * math.pi**2 / (g1 * g2) * torch.sqrt(math.pi / (g1 + g2))
    return G

# Can't jit this, too much control flow 
def build_oei(basisA, basisB, A, B, mode):
    '''Builds overlap/kinetic/potential one-electron integral matrix of diatomic molecule with s-orbital basis functions, based on 'mode' argument'''
    nbfA = torch.numel(basisA)
    nbfB = torch.numel(basisB)
    nbf = nbfA + nbfB
    I = torch.zeros((nbf,nbf))
    basis = torch.cat((basisA,basisB), dim=0).reshape(-1,1)
    An = A.repeat(nbfA).reshape(nbfA, 3)
    Bn = B.repeat(nbfB).reshape(nbfB, 3)
    centers = torch.cat((An,Bn),dim=0)
    if mode == 'kinetic':
        for i,b1 in enumerate(basis):
            for j in range(i+1):
                I[i,j] = kinetic(b1, basis[j], centers[i], centers[j])
                I[j,i] = I[i,j]
    if mode == 'overlap':
        for i,b1 in enumerate(basis):
            for j in range(i+1):
                I[i,j] = overlap(b1, basis[j], centers[i], centers[j])
                I[j,i] = I[i,j]
    if mode == 'potential':
        for i,b1 in enumerate(basis):
            for j in range(i+1):
                I[i,j] = potential(b1, basis[j], centers[i], centers[j], A, torch.tensor(1.0)) + \
                         potential(b1, basis[j], centers[i], centers[j], B, torch.tensor(1.0))
                I[j,i] = I[i,j]
    return I

# Can't jit this, too much control flow 
def build_tei(basisA, basisB, basisC, basisD, A, B, C, D):
    '''Builds two-electron integral 4d tensor for a diatomic molecule with s-orbital basis functions'''
    nbfA, nbfB = torch.numel(basisA), torch.numel(basisB)
    nbf = nbfA + nbfB 
    G = torch.zeros((nbf,nbf,nbf,nbf))
    basis = torch.cat((basisA,basisB,basisC,basisD), dim=0).reshape(-1,1)
    An = A.repeat(nbfA).reshape(nbfA, 3)
    Bn = B.repeat(nbfB).reshape(nbfB, 3)
    Cn = C.repeat(nbfA).reshape(nbfA, 3)
    Dn = D.repeat(nbfB).reshape(nbfB, 3)
    centers = torch.cat((An,Bn,Cn,Dn),dim=0)
    # WARNING: 8 fold redundancy
    for i in range(nbf):
        for j in range(nbf):
            for k in range(nbf):
                for l in range(nbf):
                    G[i,j,k,l] = eri(basis[i], basis[j], basis[k], basis[l], centers[i], centers[j], centers[k], centers[l])
    return G

@torch.jit.script
def nuclear_repulsion(atom1, atom2):
    ''' warning : hard coded for H2'''
    Za = 1.0
    Zb = 1.0
    return Za*Zb / torch.norm(atom1-atom2) 

@torch.jit.script
def orthogonalizer(S):
    '''Compute overlap to the negative 1/2 power'''
    #eigval, eigvec = torch.symeig(A1, eigenvectors=True)
    #d12 = torch.diag(torch.sqrt(eigval))
    #A = torch.chain_matmul(eigvec, d12, torch.t(eigvec))
    # More stable for small eigenvalues:
    eigval, eigvec = torch.symeig(S, eigenvectors=True)
    cutoff = 1.0e-12
    above_cutoff = (abs(eigval) > cutoff * torch.max(abs(eigval)))
    val = torch.rsqrt(eigval[above_cutoff])
    vec = eigvec[:, above_cutoff]
    A = torch.chain_matmul(vec, torch.diag(val), vec.t())
    return A

def matrix_sqrt(S):
    '''Compute matrix to 1/2 power'''
    eigval, eigvec = torch.symeig(S, eigenvectors=True)
    cutoff = 1.0e-12
    above_cutoff = (abs(eigval) > cutoff * torch.max(abs(eigval)))
    val = torch.sqrt(eigval[above_cutoff])
    vec = eigvec[:, above_cutoff]
    A = torch.chain_matmul(vec, torch.diag(val), vec.t())
    return A


# Define coordinates in Bohr as Torch tensors, turn on gradient tracking.  
tmpgeom1 = [0.000000000000,0.000000000000,-0.849220457955,0.000000000000,0.000000000000,0.849220457955]
tmpgeom2 = [torch.tensor(i, requires_grad=True) for i in tmpgeom1]
geom = torch.stack(tmpgeom2).reshape(2,3)
# Define some basis function exponents
basis0 = torch.tensor([0.5], requires_grad=False)
basis1 = torch.tensor([0.5, 0.4], requires_grad=False)
basis2 = torch.tensor([0.5, 0.4, 0.3, 0.2], requires_grad=False)
basis3 = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05], requires_grad=False)
basis4 = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001], requires_grad=False)
basis5 = torch.rand(50)
# Load converged Psi4 Densities for basis sets 1 through 4
#D0 = torch.from_numpy(np.load('psi4_densities/D0.npy'))
#D1 = torch.from_numpy(np.load('psi4_densities/D1.npy'))
#D2 = torch.from_numpy(np.load('psi4_densities/D2.npy'))
#D3 = torch.from_numpy(np.load('psi4_densities/D3.npy'))
#D4 = torch.from_numpy(np.load('psi4_densities/D4.npy'))


#@torch.jit.script
def hartree_fock(basis,geom,F):
    ndocc = 1 #hard coded
    full_basis = torch.cat((basis,basis))
    nbf_per_atom = torch.tensor([basis.size()[0],basis.size()[0]])
    charge_per_atom = torch.tensor([1.0,1.0])
    Enuc = nuclear_repulsion(geom[0], geom[1])
    S, T, V = vectorized_oei(full_basis, geom, nbf_per_atom, charge_per_atom)
    G = vectorized_tei(full_basis,geom,nbf_per_atom)
    A = orthogonalizer(S)

    H = T + V
    for i in range(15):
        Fp = torch.chain_matmul(A, F, A)
        eps, C2 = torch.symeig(Fp, eigenvectors=True)       
        #C2 = C2.detach()
        #eps = eps.detach()
        C = torch.matmul(A, C2) # At this point, the MO coefficients, and therefore the density, become connected to the Geometry
        #C = C.detach()        # We know this because if you detach here it breaks SCF gradients, but if you don't the scf gradients are correct
        D = torch.einsum('pi,qi->pq', C[:,:ndocc], C[:,:ndocc])
        # Density determined. Build fock
        J = torch.einsum('pqrs,rs->pq', G, D)
        K = torch.einsum('prqs,rs->pq', G, D)
        F = H + J * 2 - K
        E_scf = torch.einsum('pq,pq->', F + H, D) + Enuc
        E_mp2 = mp2(eps,G,C)
        mp2_grad = torch.autograd.grad(E_mp2, geom, create_graph=True)[0]
        print(mp2_grad)
        print('\n')
        #scf_grad = torch.autograd.grad(E_scf, geom,create_graph=True)[0]
        #E_mp2 = mp2(eps,G,C)
        #mp2_grad = torch.autograd.grad(E_mp2, geom, create_graph=True)[0]
    #E_mp2 = mp2(eps,G,C)
    #mp2_grad = torch.autograd.grad(E_mp2, geom, create_graph=True)[0]
    #print(mp2_grad)
    #print('\n')

    # Now get MP2 stuff 
    #Fp = torch.chain_matmul(A, F, A)
    #eps, C = torch.symeig(Fp, eigenvectors=True)       
    #C = torch.matmul(A, C)
    #E_mp2 = mp2(eps,G,C)
    #grad = torch.autograd.grad(E_mp2, geom, create_graph=True)
    #print(grad)

@torch.jit.script
def mp2(eps, G, C):
    ndocc = 1 #hard coded
    eps_occ, eps_vir = eps[:ndocc], eps[ndocc:]
    e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) - eps_vir.reshape(-1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir)
    Gmo = torch.einsum('pqrs,sl,rk,qj,pi->ijkl', G, C[:,ndocc:],C[:,:ndocc],C[:,ndocc:],C[:,:ndocc])
    E_mp2 = torch.einsum('iajb,iajb,iajb->', Gmo, Gmo, e_denom) + torch.einsum('iajb,iajb,iajb->', Gmo - Gmo.permute(0,3,2,1), Gmo, e_denom) 
    return E_mp2


#D = torch.from_numpy(np.load('D2.npy'))
#C = torch.from_numpy(np.load('C2.npy'))
#eps = torch.from_numpy(np.load('eps2.npy'))
F = torch.from_numpy(np.load('F2.npy'))
#print(F)
#dumb_hartree_fock(basis2,geom,C)
hartree_fock(basis2,geom,F)
#print(E)
#print(torch.autograd.grad(E,geom))

#mp2(basis2,geom,C,eps)

