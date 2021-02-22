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

#def boys(arg):
#    '''
#    F0(x) boys function. When x near 0, use taylor expansion, 
#       F0(x) = sum over k to infinity:  (-x)^k / (k!(2k+1))
#    Otherwise,
#       F0(x) = sqrt(pi/(4x)) * erf(sqrt(x))
#    '''
#    if arg < 1e-8:
#        #NOTE This expansion must go to same order as angular momentum, otherwise potential/eri integrals are wrong. 
#        # This currently just supports up to g functions. (arg**4 term)
#        boys = 1 - (arg / 3) + (arg**2 / 10) - (arg**3 / 42) + (arg**4 / 216)
#    else:
#        boys = jax.scipy.special.erf(np.sqrt(arg)) * np.sqrt(np.pi / (4 * arg))
#    return boys

@jax.jarrett
def boys(arg):
    '''Alternative boys function expansion. Not exact.'''
    boys = 0.5 * np.exp(-arg) * (1 / (0.5)) * (1 + (arg / (1.5)) *\
                                                          (1 + (arg / (2.5)) *\
                                                          (1 + (arg / (3.5)) *\
                                                          (1 + (arg / (4.5)) *\
                                                          (1 + (arg / (5.5)) *\
                                                          (1 + (arg / (6.5)) *\
                                                          (1 + (arg / (7.5)) *\
                                                          (1 + (arg / (8.5)) *\
                                                          (1 + (arg / (9.5)) *\
                                                          (1 + (arg / (10.5))*\
                                                          (1 + (arg / (11.5)))))))))))))
    return boys

def double_factorial(n):
    '''The double factorial function for small Python integer `n`.'''
    return np.prod(np.arange(n, 1, -2))

def normalize(aa,ax,ay,az):
    '''
    Normalization constant for gaussian basis function. 
    aa : orbital exponent
    ax : angular momentum component x
    ay : angular momentum component y
    az : angular momentum component z
    '''
    #f = np.sqrt(double_factorial(2*ax-1) * double_factorial(2*ay-1) * double_factorial(2*az-1))
    # TODO TODO 
    f = 1
    N = (2*aa/np.pi)**(3/4) * (4 * aa)**((ax+ay+az)/2) / f
    return N

def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * np.dot(A-C, A-C)) / (aa + bb)))
    #ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * ((Ax - Cx)**2 + (Ay - Cy)**2 + (Az - Cz)**2))) / (aa + bb))
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

def recursively_promote_oei(args, start_am, target_am, current, old=None):
    '''
    An arbitrary angular momentum one electron integral function factory.
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
    for idx in range(6): 
        if start_am[idx] != target_am[idx]:
            # Define coefficients of terms
            ai = start_am[idx]
            if idx <= 2:
                alpha = args[-2] 
            else:
                alpha = args[-1] 
            # Build one-increment higher angular momentum function
            if ai == 0:
                def new(*args): 
                    return (1/(2*alpha)) * (jax.grad(current,idx)(*args))
            else:
                def new(*args): 
                    # Uh oh... `old` is incorrect when going from [1,0,0,0,0,0] to [1,0,0,1,0,0], right?
                    return (1/(2*alpha)) * (jax.grad(current,idx)(*args) + ai * old(*args))
            # Increment angular momentum vector by one in the proper place.
            promotion = onp.zeros(6)
            promotion[idx] += 1
            return recursively_promote_oei(args, start_am + promotion, target_am, new, current)

        else:
            continue
    return current

#def recursively_promote_oei(args, start_am, target_am, current, old=None):

def oei_0(args, func):
    return func(*args)

def oei_1(args, target_am, exponent_vector, func):
    mask = np.where(target_am != 0, True,False)
    indices = np.arange(6)[mask]
    # may have to sort?
    grad_indices = np.repeat(indices, target_am[mask])
    idx = int(grad_indices[0])
    alpha = exponent_vector[idx]
    def new(*args): 
        return (1/(2*alpha)) * (jax.grad(func,argnums=idx)(*args))
    return new(*args)

def oei_2(args, target_am, exponent_vector, func):
    '''Returns a OEI with  '''
    increment_count_vector = np.zeros(6)
    mask = np.where(target_am != 0, True,False)
    indices = np.arange(6)[mask]
    grad_indices = np.repeat(indices, target_am[mask])

    idx1 = int(grad_indices[0])
    alpha1 = exponent_vector[idx1]
    def new(*args): 
        return (1/(2*alpha1)) * jax.grad(func,idx1)(*args)
    # after first pass, you need to track which indices are incremented for ai factor
    increment_count_vector = jax.ops.index_add(increment_count_vector,idx1,1)

    idx2 = int(grad_indices[1])
    ai2 = increment_count_vector[idx2]
    alpha2 = exponent_vector[idx2]
    
    def new2(*args): 
        return (1/(2*alpha2)) * (jax.grad(new,idx2)(*args) + ai2 * func(*args))
    return new2(*args)

def oei_3(args, target_am, exponent_vector, func):
    increment_count_vector = np.zeros(6)
    mask = np.where(target_am != 0, True,False)
    indices = np.arange(6)[mask]
    grad_indices = np.repeat(indices, target_am[mask])

    idx1 = int(grad_indices[0])
    alpha1 = exponent_vector[idx1]
    def new(*args): 
        return (1/(2*alpha1)) * jax.grad(func,idx1)(*args)
    increment_count_vector = jax.ops.index_add(increment_count_vector,idx1,1)

    idx2 = int(grad_indices[1])
    ai2 = increment_count_vector[idx2]
    alpha2 = exponent_vector[idx2]
    def new2(*args): 
        return (1/(2*alpha2)) * (jax.grad(new,idx2)(*args) + ai2 * func(*args))
    increment_count_vector = jax.ops.index_add(increment_count_vector,idx2,1)

    # Get index, alpha, ai
    idx3 = int(grad_indices[2])
    ai3 = increment_count_vector[idx3]
    alpha3 = exponent_vector[idx3]
    def new3(*args): 
        return (1/(2*alpha3)) * (jax.grad(new2,idx3)(*args) + ai3 * new(*args))
    return new3(*args)

def oei_4(args, target_am, exponent_vector, func):
    # Would have to preprocess this TODO TODO
    increment_count_vector = np.zeros(6)
    #TEMP TODO
    #mask = np.where(target_am != 0, True,False)
    #indices = np.arange(6)[mask]
    #grad_indices = np.repeat(indices, target_am[mask])
    #TEMP TODO
    grad_indices = [0, 0, 3,3]
    # Get index, alpha, ai
    idx1 = int(grad_indices[0])
    alpha1 = exponent_vector[idx1]
    def new(*args): 
        return (1/(2*alpha1)) * jax.grad(func,idx1)(*args)
    increment_count_vector = jax.ops.index_add(increment_count_vector,idx1,1)

    # Get index, alpha, ai
    idx2 = int(grad_indices[1])
    ai2 = increment_count_vector[idx2]
    alpha2 = exponent_vector[idx2]
    def new2(*args): 
        return (1/(2*alpha2)) * (jax.grad(new,idx2)(*args) + ai2 * func(*args))
    increment_count_vector = jax.ops.index_add(increment_count_vector,idx2,1)

    # Get index, alpha, ai
    idx3 = int(grad_indices[2])
    ai3 = increment_count_vector[idx3]
    alpha3 = exponent_vector[idx3]
    def new3(*args): 
        return (1/(2*alpha3)) * (jax.grad(new2,idx3)(*args) + ai3 * new(*args))
    increment_count_vector = jax.ops.index_add(increment_count_vector,idx3,1)


    idx4 = int(grad_indices[3])
    #ai = increment_count_vector[idx4]
    ai4 = increment_count_vector[idx4]
    alpha4 = exponent_vector[idx4]
    def new4(*args): 
        return (1/(2*alpha4)) * (jax.grad(new3,idx4)(*args) + ai4 * new2(*args))
    return new4(*args)


#def oei_2(args,func):
#    alpha = 0.5
#    mask = np.where(target_am != 0, True,False)
#    indices = np.arange(6)[mask]
#    grad_indices = np.repeat(indices, target_am[mask])
#
#    idx1 = int(grad_indices[0])
#    def new(*args): 
#        return (1/(2*alpha)) * jax.grad(func,idx1)(*args)
#
#    idx2 = int(grad_indices[1])
#    def new2(*args): 
#        #return (1/(2*alpha)) * (jax.grad(new,idx)(*args) + ai * func(*args))
#        return (1/(2*alpha)) * (jax.grad(new,idx2)(*args))
#    return new2(*args)
 
# Create exponents vector for H: S 0.5 and H: D 0.5
geom = np.array([[0.0,0.0,-0.849220457955],
                 [0.0,0.0, 0.849220457955]])
charge = np.array([1.0,1.0])

## S S test
#exponents = np.repeat(0.5, 2)
#nbf_per_atom = np.array([1,1])
#centers = np.repeat(geom, nbf_per_atom, axis=0)
#angular_momentum = np.array([[0,0,0], [0,0,0]])
#nbf = exponents.shape[0]

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
#centers = np.repeat(geom, nbf_per_atom, axis=0)
#angular_momentum = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,0],[0,1,0],[0,0,1]])
#nbf = exponents.shape[0]

# D D test
exponents = np.repeat(0.5, 12)
nbf_per_atom = np.array([6,6])
centers = np.repeat(geom, nbf_per_atom, axis=0)
angular_momentum = np.array([[2,0,0], [1,1,0], [1,0,1], [0,2,0], [0,1,1], [0,0,2], [2,0,0], [1,1,0], [1,0,1], [0,2,0], [0,1,1], [0,0,2]])
nbf = exponents.shape[0]


## F F test
#exponents = np.repeat(0.5, 20)
#nbf_per_atom = np.array([10,10])
#centers = np.repeat(geom, nbf_per_atom, axis=0)
#angular_momentum = np.array([[3,0,0], [2,1,0], [1,2,0], [2,0,1], [1,0,2], [1,1,1], [0,3,0], [0,2,1], [0,1,2], [0,0,3],[3,0,0], [2,1,0], [1,2,0], [2,0,1], [1,0,2], [1,1,1], [0,3,0], [0,2,1], [0,1,2], [0,0,3]])
#nbf = exponents.shape[0]

S = onp.zeros((nbf,nbf))
T = onp.zeros((nbf,nbf))
V = onp.zeros((nbf,nbf))
#
indices = []
for i in range(nbf):
    for j in range(i+1):
        indices.append([i,j])

indices = np.asarray(indices)
#print(indices)

def get_overlap(idx):
    i,j = idx
    aa = exponents[i]
    bb = exponents[j]
    Ax, Ay, Az = centers[i]
    Bx, By, Bz = centers[j]
    pi, pj, pk = angular_momentum[i]
    qi, qj, qk = angular_momentum[j]
    start_am = np.array([0,0,0,0,0,0])
    target_am = np.array([pi,pj,pk,qi,qj,qk])
    args = (Ax,Ay,Az,Bx,By,Bz,aa,bb)
    Na = normalize(args[-2], pi, pj, pk)
    Nb = normalize(args[-1], qi, qj, qk)
    #overlap_func = promote_oei(args, target_am, np.array([aa,aa,aa,bb,bb,bb]), np.zeros(6), overlap_ss, overlap_ss)
    #overlap_func = test(args, start_am, target_am, overlap_ss, old=None)
    #overlap_func = test(args, start_am, target_am, overlap_ss, old=overlap_ss)
    #overlap_func = recursively_promote_oei(args, start_am, target_am, overlap_ss, old=None)

    #if np.sum(target_am) == 0:
    #    overlap2 = oei_0(args,overlap_ss)
    #    print(overlap2*Na*Nb)
    #if np.sum(target_am) == 1:
    #    overlap2 = oei_1(args,overlap_ss)
    #    print(overlap2*Na*Nb)
    #if np.sum(target_am) == 2:
    #    overlap2 = oei_2(args, target_am, np.array([aa,aa,aa,bb,bb,bb]), overlap_ss)
    #    print(overlap2*Na*Nb)
    #if np.sum(target_am) == 3:
    #    overlap2 = oei_3(args, target_am, np.array([aa,aa,aa,bb,bb,bb]), overlap_ss)
    #    print(overlap2*Na*Nb)
    if np.sum(target_am) == 4:
        overlap2 = oei_4(args, target_am, np.array([aa,aa,aa,bb,bb,bb]), overlap_ss)
    overlap = Na * Nb * overlap2
    return overlap
##
#vectorized_overlap = jax.jit(jax.vmap(get_overlap, (0,)))
#overlaps = vectorized_overlap(indices)
#print(overlaps)
#TODO turn back on
overlaps = jax.lax.map(get_overlap, indices)
print(overlaps)

def dummy(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
    return 1

#for i in range(nbf):
#    for j in range(i+1):
#        aa = exponents[i]
#        bb = exponents[j]
#        Ax, Ay, Az = centers[i]
#        Bx, By, Bz = centers[j]
#        #pi, pj, pk = angular_momentum[i]
#        #qi, qj, qk = angular_momentum[j]
#        pi, pj, pk = angular_momentum[i][0], angular_momentum[i][1], angular_momentum[i][2]
#        qi, qj, qk = angular_momentum[j][0], angular_momentum[j][1], angular_momentum[j][2] 
#
#        start_am = onp.array([0,0,0,0,0,0])
#        #target_am = onp.array([pi,pj,pk,qi,qj,qk])
#        target_am = np.array([pi,pj,pk,qi,qj,qk])
#        #target_am = np.asarray(onp.array([pi,pj,pk,qi,qj,qk]))
#        args = (Ax,Ay,Az,Bx,By,Bz,aa,bb)
#
#        overlap_func = recursively_promote_oei(args, start_am, target_am, overlap_ss, old=None)
#        Na = normalize(args[-2], pi, pj, pk)
#        Nb = normalize(args[-1], qi, qj, qk)
#        overlap = Na * Nb * overlap_func(*args) 
#        print(overlap)

        #if np.sum(target_am) == 0:
        #    overlap2 = oei_0(args,overlap_ss)
        #    print(overlap2*Na*Nb)
        #if np.sum(target_am) == 1:
        #    overlap2 = oei_1(args,overlap_ss)
        #    print(overlap2*Na*Nb)
        #if np.sum(target_am) == 2:
        #    overlap2 = oei_2(args, target_am, np.array([aa,aa,aa,bb,bb,bb]), overlap_ss)
        #    print(overlap2*Na*Nb)
        #if np.sum(target_am) == 3:
        #    overlap2 = oei_3(args, target_am, np.array([aa,aa,aa,bb,bb,bb]), overlap_ss)
        #    print(overlap2*Na*Nb)
        #if np.sum(target_am) == 4:
        #    overlap2 = oei_4(args, target_am, np.array([aa,aa,aa,bb,bb,bb]), overlap_ss)
        #    print(overlap2*Na*Nb)


#        overlap = oei_2(args, overlap_ss)
        #overlap = oei_1(args, target_am, np.array([aa,aa,aa,bb,bb,bb]), overlap_ss)
#
        #overlap = Na * Nb * overlap_func(Ax,Ay,Az,Bx,By,Bz,aa,bb) 
        #overlap = Na * Nb * overlap_func(*args) 
        #overlap = overlap_func(*args) 
        #print(overlap)
#        S[i,j] = overlap
#
#        kinetic_func = recursively_promote_oei(args, start_am, target_am, kinetic_ss, old=None)
#        kinetic = Na * Nb * kinetic_func(*args) 
#        T[i,j] = kinetic
#
#        args = (Ax,Ay,Az,Bx,By,Bz,geom,charge,aa,bb)
#        potential_func = recursively_promote_oei(args, start_am, target_am, potential_ss, old=None)
#        potential = Na * Nb * potential_func(*args) 
#        print(potential)
#        V[i,j] = potential 
#        
#print(S)
#print(T)
#print(V)
#

