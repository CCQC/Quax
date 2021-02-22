import jax
import jax.numpy as np
import numpy as onp
onp.set_printoptions(linewidth=500)
from jax.config import config; config.update("jax_enable_x64", True)

@jax.jit
def gp(aa,bb,A,B):
    '''Gaussian product theorem. Returns center and coefficient of product'''
    R = (aa * A + bb * B) / (aa + bb)
    c = np.exp(np.dot(A-B,A-B) * (-aa * bb / (aa + bb)))
    return R,c

def boys(arg):
    '''
    F0(x) boys function. When x near 0, use taylor expansion, 
       F0(x) = sum over k to infinity:  (-x)^k / (k!(2k+1))
    Otherwise,
       F0(x) = sqrt(pi/(4x)) * erf(sqrt(x))
    '''
    if arg < 1e-8:
        #NOTE This expansion must go to same order as angular momentum, otherwise potential/eri integrals are wrong. 
        # This currently just supports up to g functions. (arg**4 term)
        boys = 1 - (arg / 3) + (arg**2 / 10) - (arg**3 / 42) + (arg**4 / 216)
    else:
        boys = jax.scipy.special.erf(np.sqrt(arg)) * np.sqrt(np.pi / (4 * arg))
    return boys

def double_factorial(n):
    '''The double factorial function for small Python integer `n`.'''
    return np.prod(np.asarray(np.arange(n, 1, -2)))

def normalize(aa,ax,ay,az):
    '''
    Normalization constant for gaussian basis function. 
    aa : orbital exponent
    ax : angular momentum component x
    ay : angular momentum component y
    az : angular momentum component z
    '''
    f = np.sqrt(double_factorial(2*ax-1) * double_factorial(2*ay-1) * double_factorial(2*az-1))
    N = (2*aa/np.pi)**(3/4) * (4 * aa)**((ax+ay+az)/2) / f
    return N

def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
    ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * ((Ax - Cx)**2 + (Ay - Cy)**2 + (Az - Cz)**2))) / (aa + bb))
    return ss

def kinetic_ss(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    P = (aa * bb) / (aa + bb)
    ab = -1.0 * np.dot(A-C, A-C)
    K = overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, aa, bb) * (3 * P + 2 * P * P * ab)
    return K

def potential_ss(Ax, Ay, Az, Cx, Cy, Cz, geom, charge, aa, bb):
    '''
    Computes a single unnormalized electron-nuclear 
    potential energy integral over two primitive 
    s-orbital basis functions
    Ax,Ay,Az,Cx,Cy,Cz: cartesian coordinates of centers
    aa,bb: gaussian primitive exponents
    geom: Nx3 array of cartesian geometry 
    charge: N array of charges
    '''
    A = np.array([Ax, Ay, Az])
    B = np.array([Cx, Cy, Cz])
    g = aa + bb
    eps = 1 / (4 * g)
    P, c = gp(aa,bb,A,B)
    V = 0
    # For every atom
    for i in range(geom.shape[0]):
        arg = g * np.dot(P - geom[i], P - geom[i])
        F = boys(arg)
        V += -charge[i] * F * c * 2 * np.pi / g
    return V

def eri_ss(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,aa,bb,cc,dd):
    '''Computes a single unnormalized 
    two electron integral over 4 s-orbital basis functions on 4 centers'''
    A = np.array([Ax, Ay, Az])
    B = np.array([Bx, By, Bz])
    C = np.array([Cx, Cy, Cz])
    D = np.array([Dx, Dy, Dz])
    g1 = aa + bb
    g2 = cc + dd
    Rp = (aa * A + bb * B) / (aa + bb)
    tmpc1 = np.dot(A-B, A-B) * ((-aa * bb) / (aa + bb))
    c1 = np.exp(tmpc1)
    Rq = (cc * C + dd * D) / (cc + dd)
    tmpc2 = np.dot(C-D, C-D) * ((-cc * dd) / (cc + dd))
    c2 = np.exp(tmpc2)
    delta = 1 / (4 * g1) + 1 / (4 * g2)
    arg = np.dot(Rp - Rq, Rp - Rq) / (4 * delta)
    F = boys(arg)
    G = F * c1 * c2 * 2 * np.pi**2 / (g1 * g2) * np.sqrt(np.pi / (g1 + g2))
    return G

def cartesian_product(*arrays):
    '''Generalized cartesian product
       Used to find all *indices* of values in an ERI tensor of size (nbf,nbf,nbf,nbf) 
       given 4 arrays:
       (np.arange(nbf), np.arange(nbf), np.arange(nbf), np.arange(nbf))'''
    la = len(arrays)
    dtype = onp.find_common_type([a.dtype for a in arrays], [])
    arr = onp.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(onp.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def find_unique_tei_indices(nbf):
    '''Finds a set of indices of an ERI Tensor corresponding to 
    a UNIQUE set two-electron integrals.'''
    # NOTE probably a faster way to do this, i.e. organically generate the set of unique indices instead of first generating all indices and filtering with boolean masks
    v = onp.arange(nbf,dtype=np.int16) #int16 reduces memory by half, no need for large integers, it will not exceed nbf
    indices = cartesian_product(v,v,v,v)
    size = indices.shape[0]
    batch_size = int(size/4) # batch size
    # Evaluate indices (in batches to save memory) in 'canonical' order, i>=j, k>=l, IJ>=KL
    def get_mask(a,b):
        cond1 = (indices[a:b,0] >= indices[a:b,1]) & (indices[a:b,2] >= indices[a:b,3])
        cond2 = indices[a:b,0] * (indices[a:b,0] + 1)/2 + indices[a:b,1] >= indices[a:b,2]*(indices[a:b,2]+1)/2 + indices[a:b,3]
        mask = cond1 & cond2
        return mask
    mask1 = get_mask(0,batch_size)
    mask2 = get_mask(batch_size, 2 * batch_size)
    mask3 = get_mask(2 * batch_size, 3 * batch_size)

    a = 3 * batch_size
    cond1 = (indices[a:,0] >= indices[a:,1]) & (indices[a:,2] >= indices[a:,3])
    cond2 = indices[a:,0] * (indices[a:,0] + 1)/2 + indices[a:,1] >= indices[a:,2]*(indices[a:,2]+1)/2 + indices[a:,3]
    mask4 = cond1 & cond2
    mask = np.hstack((mask1,mask2,mask3,mask4))
    # Keep non-batched version here, for clarity:
    #cond1 = (indices[:,0] >= indices[:,1]) & (indices[:,2] >= indices[:,3]) 
    #cond2 = indices[:,0] * (indices[:,0] + 1)/2 + indices[:,1] >= indices[:,2]*(indices[:,2]+1)/2 + indices[:,3]
    #mask = cond1 & cond2 
    return np.asarray(indices[mask,:])

def recursively_promote_angular_momentum(args, start_am, target_am, exponent_vector, current, old=None, dim=6):
    '''
    An arbitrary angular momentum one electron integral function factory.
    Takes in some base s-orbital integral function, as well as target angular momentum vector
    Uses the fact that (a+1i|b) = 1/2 alpha * [ d/dAi (a|b) + ai (a-1i|b)]
    where alpha is the exponent on a, ai is the current angular momentum at index i in (a|b)

    Parameters
    ----------
    args: tuple
        Arguments of the function `current`, for overlap is (Ax, Ay, Az, Cx, Cy, Cz, aa, bb)
    start_am : onp.array
        Starting angular momentum vector (bra|ket):= [ax,ay,az,bx,by,bz] for integral (a|b). 
        Always will start with (s|s) := onp.array([0,0,0,0,0,0]), and this function will recursively construct
        function which computes integral with `target_am`
    target_am: onp.array
        Target angular momentum vector. Recursion will halt when we construct a function which computes the integral
        (a|b) with this desired angular momentum on the basis functions.
        A (px|dyz) integral target am would be, for example, [1,0,0,0,1,1].
    current: function 
        Current function for computing a one electron integral with arguments (Ax, Ay, Az, Cx, Cy, Cz, aa, bb)
        Starts with (s|s) function, gets promoted through the recursion.
    old: function
        Function for computing previous (a-1|b) or (a|b-1) function.
    Returns
    -------
    A function which computes the one electron integral with proper angular momentum on the basis functions
    NOTE: could just optionally return the evaluated integral!!!
    '''
    for idx in range(dim): 
        if start_am[idx] != target_am[idx]:
            # Define coefficients of terms
            ai = start_am[idx]
            alpha = exponent_vector[idx]
            # Build one-increment higher angular momentum function
            if ai == 0:
                def new(*args): 
                    return (1/(2*alpha)) * (jax.grad(current,idx)(*args))
            else:
                def new(*args): 
                    # Uh oh... `old` is incorrect when going from [1,0,0,0,0,0] to [1,0,0,1,0,0], right?
                    return (1/(2*alpha)) * (jax.grad(current,idx)(*args) + ai * old(*args))
              
            # Increment angular momentum vector by one in the proper place.
            promotion = onp.zeros(dim)
            promotion[idx] += 1
            return recursively_promote_angular_momentum(args, start_am + promotion, target_am, exponent_vector, new, current, dim=dim)
        else:
            continue
    return current

# Create exponents vector for H: S 0.5 and H: D 0.5
geom = np.array([[0.0,0.0,-0.849220457955],
                 [0.0,0.0, 0.849220457955]])
charge = np.array([1.0,1.0])

# S S test
exponents = np.repeat(0.5, 2)
nbf_per_atom = np.array([1,1])
centers = np.repeat(geom, nbf_per_atom, axis=0)
angular_momentum = np.array([[0,0,0], [0,0,0]])
nbf = exponents.shape[0]

## S P test
#exponents = np.repeat(0.5, 4)
#nbf_per_atom = np.array([1,3])
#centers = np.repeat(geom, nbf_per_atom, axis=0)
#angular_momentum = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
#nbf = exponents.shape[0]

## S D test
#exponents = np.repeat(0.5, 7)
#nbf_per_atom = np.array([1,6])
#centers = np.repeat(geom, nbf_per_atom, axis=0)
#angular_momentum = np.array([[0,0,0], [2,0,0], [1,1,0], [1,0,1], [0,2,0], [0,1,1], [0,0,2]])
#nbf = exponents.shape[0]

## P P test 
#exponents = np.repeat(0.5, 6)
#nbf_per_atom = np.array([3,3])
#angular_momentum = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,0],[0,1,0],[0,0,1]])
#nbf = exponents.shape[0]

## D D test
#exponents = np.repeat(0.5, 12)
#nbf_per_atom = np.array([6,6])
#centers = np.repeat(geom, nbf_per_atom, axis=0)
#angular_momentum = np.array([[2,0,0], [1,1,0], [1,0,1], [0,2,0], [0,1,1], [0,0,2], [2,0,0], [1,1,0], [1,0,1], [0,2,0], [0,1,1], [0,0,2]])
#nbf = exponents.shape[0]


## F F test
#exponents = np.repeat(0.5, 20)
#nbf_per_atom = np.array([10,10])
#centers = np.repeat(geom, nbf_per_atom, axis=0)
#angular_momentum = np.array([[3,0,0], [2,1,0], [1,2,0], [2,0,1], [1,0,2], [1,1,1], [0,3,0], [0,2,1], [0,1,2], [0,0,3],[3,0,0], [2,1,0], [1,2,0], [2,0,1], [1,0,2], [1,1,1], [0,3,0], [0,2,1], [0,1,2], [0,0,3]])
#nbf = exponents.shape[0]

##########################
# ONE-ELECTRON INTEGRALS #
##########################

def compute_oei(geom, exponents, nbf_per_atom, angular_momentum, nbf):
    centers = np.repeat(geom, nbf_per_atom, axis=0)
    S = np.zeros((nbf,nbf))
    T = np.zeros((nbf,nbf))
    V = np.zeros((nbf,nbf))
    
    for i in range(nbf):
        for j in range(i+1):
            aa = exponents[i]
            bb = exponents[j]
            Ax, Ay, Az = centers[i]
            Bx, By, Bz = centers[j]
            pi, pj, pk = angular_momentum[i]
            qi, qj, qk = angular_momentum[j]
    
            start_am = onp.array([0,0,0,0,0,0])
            target_am = onp.array([pi,pj,pk,qi,qj,qk])
            args = (Ax,Ay,Az,Bx,By,Bz,aa,bb)
            exponent_vector = np.array([aa,aa,aa,bb,bb,bb])
            ## NOTE for (d|d) only. Get from Psi4.
            #Na = 0.489335770373359
            #Nb = 0.489335770373359
            ## NOTE for (f|f) only. Get from Psi4.
            #Na = 0.3094831149945914
            #Nb = 0.3094831149945914
            Na = normalize(args[-2], pi, pj, pk)
            Nb = normalize(args[-1], qi, qj, qk)
    
            overlap_func = recursively_promote_angular_momentum(args, start_am, target_am, exponent_vector, overlap_ss, old=None, dim=6)
            overlap = Na * Nb * overlap_func(*args) 
            S = jax.ops.index_update(S, jax.ops.index[i,j], overlap)
            S = jax.ops.index_update(S, jax.ops.index[j,i], overlap)
    
            kinetic_func = recursively_promote_angular_momentum(args, start_am, target_am, exponent_vector, kinetic_ss, old=None, dim=6)
            kinetic = Na * Nb * kinetic_func(*args) 
            T = jax.ops.index_update(T, jax.ops.index[i,j], kinetic)
            T = jax.ops.index_update(T, jax.ops.index[j,i], kinetic)
    
            args = (Ax,Ay,Az,Bx,By,Bz,geom,charge,aa,bb)
            potential_func = recursively_promote_angular_momentum(args, start_am, target_am, exponent_vector, potential_ss, old=None, dim=6)
            potential = Na * Nb * potential_func(*args) 
            V = jax.ops.index_update(V, jax.ops.index[i,j], potential)
            V = jax.ops.index_update(V, jax.ops.index[j,i], potential)
    return S, T, V

##########################
# TWO-ELECTRON INTEGRALS #
##########################

def compute_tei(geom, exponents, nbf_per_atom, angular_momentum, nbf):
    centers = np.repeat(geom, nbf_per_atom, axis=0)
    unique_tei_indices = find_unique_tei_indices(nbf)
    unique_teis = []
    for idx in unique_tei_indices:
        i,j,k,l = idx
        aa,bb,cc,dd = exponents[i], exponents[j], exponents[k], exponents[l]
        Ax,Ay,Az = centers[i]
        Bx,By,Bz = centers[j]
        Cx,Cy,Cz = centers[k]
        Dx,Dy,Dz = centers[l]
    
        pi, pj, pk = angular_momentum[i]
        qi, qj, qk = angular_momentum[j]
        ri, rj, rk = angular_momentum[k]
        si, sj, sk = angular_momentum[l]
    
        start_am = onp.array([0,0,0,0,0,0,0,0,0,0,0,0])
        target_am = onp.array([pi,pj,pk,qi,qj,qk,ri,rj,rk,si,sj,sk])
        exponent_vector = np.array([aa,aa,aa,bb,bb,bb,cc,cc,cc,dd,dd,dd])
        args = (Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,aa,bb,cc,dd)
        eri_func = recursively_promote_angular_momentum(args, start_am, target_am, exponent_vector, eri_ss, old=None, dim=12)
    
        Na = normalize(aa, pi, pj, pk)
        Nb = normalize(bb, qi, qj, qk)
        Nc = normalize(cc, pi, pj, pk)
        Nd = normalize(dd, qi, qj, qk)
        eri_integral = Na * Nb * Nc * Nd * eri_func(*args) 
        unique_teis.append(eri_integral)
        #print(eri_integral)
    
    unique_teis = np.asarray(unique_teis)
    return unique_teis


# Test
print("Amount of time for {} basis functions".format(nbf))
print(nbf)
S,T,V = compute_oei(geom, exponents, nbf_per_atom, angular_momentum, nbf)
print(S)
I = compute_tei(geom, exponents, nbf_per_atom, angular_momentum, nbf)


