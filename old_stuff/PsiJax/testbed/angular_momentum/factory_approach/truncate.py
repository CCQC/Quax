import jax
import jax.numpy as np
import numpy as onp
onp.set_printoptions(linewidth=500)
from jax.config import config; config.update("jax_enable_x64", True)

def double_factorial(n):
    '''The double factorial function for small Python integer `n`.'''
    return np.prod(np.arange(n, 1, -2))

@jax.jit
def gp(aa,bb,A,B):
    '''Gaussian product theorem. Returns center and coefficient of product'''
    R = (aa * A + bb * B) / (aa + bb)
    c = np.exp(np.dot(A-B,A-B) * (-aa * bb / (aa + bb)))
    return R,c

@jax.jit
def odd_double_factorial(x): # this ones jittable, roughly equal speed, makes `normalize` also jittable.
    n = (x + 1)/2
    return 2**n * np.exp(jax.scipy.special.gammaln(n + 0.5)) / (np.pi**(0.5))

@jax.jit
def normalize(aa,ax,ay,az):
    '''
    Normalization constant for gaussian basis function. 
    aa : orbital exponent
    ax : angular momentum component x
    ay : angular momentum component y
    az : angular momentum component z
    '''
    #f = np.sqrt(double_factorial(2*ax-1) * double_factorial(2*ay-1) * double_factorial(2*az-1))
    f = np.sqrt(odd_double_factorial(2*ax-1) * odd_double_factorial(2*ay-1) * odd_double_factorial(2*az-1))
    N = (2*aa/np.pi)**(3/4) * (4 * aa)**((ax+ay+az)/2) / f
    return N

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

@jax.jit
def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * np.dot(A-C, A-C)) / (aa + bb)))
    return ss

@jax.jit
def kinetic_ss(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    P = (aa * bb) / (aa + bb)
    ab = -1.0 * np.dot(A-C, A-C)
    K = overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, aa, bb) * (3 * P + 2 * P * P * ab)
    return K

@jax.jit
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

def overlap_factory(args, start_am, target_am, current, old=None, dim=6):
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
    Ax, Ay, Az, Bx, By, Bz, aa, bb = args
    for idx in range(dim): 
        if start_am[idx] != target_am[idx]:
            # Define coefficients of terms
            ai = start_am[idx]
            # Build one-increment higher angular momentum function
            if ai == 0:
                if idx<=2:
                    def new(Ax, Ay, Az, Bx, By, Bz, aa, bb): 
                        return (1/(2*aa)) * (jax.grad(current,idx)(Ax, Ay, Az, Bx, By, Bz, aa, bb))
                else:
                    def new(Ax, Ay, Az, Bx, By, Bz, aa, bb): 
                        return (1/(2*bb)) * (jax.grad(current,idx)(Ax, Ay, Az, Bx, By, Bz, aa, bb))
            else:
                if idx<=2:
                    def new(Ax, Ay, Az, Bx, By, Bz, aa, bb): 
                        return (1/(2*aa)) * (jax.grad(current,idx)(Ax, Ay, Az, Bx, By, Bz, aa, bb) + ai * old(Ax, Ay, Az, Bx, By, Bz, aa, bb))
                else:
                    def new(Ax, Ay, Az, Bx, By, Bz, aa, bb): 
                        return (1/(2*bb)) * (jax.grad(current,idx)(Ax, Ay, Az, Bx, By, Bz, aa, bb) + ai * old(Ax, Ay, Az, Bx, By, Bz, aa, bb))
            promotion = onp.zeros(dim)
            promotion[idx] += 1
            return overlap_factory(args, start_am + promotion, target_am, new, current, dim=dim)
        else:
            continue
    return current

def angular_momentum_factory(args, start_am, target_am, current, old=None, dim=6):
    ''' Produces integral functions of higher angular momentum from functions of lower angular momentum '''
    for idx in range(dim): 
        if start_am[idx] != target_am[idx]:
            ai = start_am[idx]
            if ai == 0:
                if idx<=2:
                    def new(*args):
                        return (1/(2 * args[-2])) * (jax.grad(current,idx)(*args))
                else:
                    def new(*args):
                        return (1/(2 * args[-1])) * (jax.grad(current,idx)(*args))
            else:
                if idx<=2:
                    def new(*args):
                        return (1 / (2 * args[-2])) * (jax.grad(current,idx)(*args) + ai * old(*args))
                else:
                    def new(*args):
                        return (1 / (2 * args[-1])) * (jax.grad(current,idx)(*args) + ai * old(*args))
            promotion = onp.zeros(dim)
            promotion[idx] += 1
            return angular_momentum_factory(args, start_am + promotion, target_am, new, current, dim=dim)
        else:
            continue
    return current


geom = np.array([[0.0,0.0,-0.849220457955],
                 [0.0,0.0, 0.849220457955]])
charge = np.array([1.0,1.0])

## S P P P basis set 
#exponents = np.repeat(0.5, 10)
#nbf_per_atom = np.array([1,9])
#angular_momentum = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1]])

# S P basis set 
exponents = np.repeat(0.5, 4)
nbf_per_atom = np.array([1,3])
angular_momentum = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])

# P P basis set 
#exponents = np.repeat(0.5, 6)
#nbf_per_atom = np.array([3,3])
#angular_momentum = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1]])

centers = np.repeat(geom, nbf_per_atom, axis=0)
nbf = exponents.shape[0]

# Only unique overlaps of lower triangle
args = (1.,1.,1.,1.,1.,1.,1.,1.)

#func_dict['000000'] = jax.jit(overlap_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,0,0,0,0,0]), overlap_ss, old=None, dim=6))  
#func_dict['100000'] = jax.jit(overlap_factory(args, onp.array([0,0,0,0,0,0]), onp.array([1,0,0,0,0,0]), overlap_ss, old=None, dim=6)) 
#func_dict['100100'] = jax.jit(overlap_factory(args, onp.array([0,0,0,0,0,0]), onp.array([1,0,0,1,0,0]), overlap_ss, old=None, dim=6)) 
#func_dict['010000'] = jax.jit(overlap_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,1,0,0,0,0]), overlap_ss, old=None, dim=6)) 
#func_dict['010100'] = jax.jit(overlap_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,1,0,1,0,0]), overlap_ss, old=None, dim=6)) 
#func_dict['010010'] = jax.jit(overlap_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,1,0,0,1,0]), overlap_ss, old=None, dim=6)) 
#func_dict['001000'] = jax.jit(overlap_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,0,1,0,0,0]), overlap_ss, old=None, dim=6)) 
#func_dict['001100'] = jax.jit(overlap_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,0,1,1,0,0]), overlap_ss, old=None, dim=6)) 
#func_dict['001010'] = jax.jit(overlap_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,0,1,0,1,0]), overlap_ss, old=None, dim=6)) 
#func_dict['001001'] = jax.jit(overlap_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,0,1,0,0,1]), overlap_ss, old=None, dim=6)) 
#func_dict['100010'] = jax.jit(overlap_factory(args, onp.array([0,0,0,0,0,0]), onp.array([1,0,0,0,1,0]), overlap_ss, old=None, dim=6)) 
#func_dict['100001'] = jax.jit(overlap_factory(args, onp.array([0,0,0,0,0,0]), onp.array([1,0,0,0,0,1]), overlap_ss, old=None, dim=6)) 
#func_dict['010001'] = jax.jit(overlap_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,1,0,0,0,1]), overlap_ss, old=None, dim=6)) 

overlap_dict = {}
kinetic_dict = {}
potential_dict = {}
overlap_dict['000000'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,0,0,0,0,0]), overlap_ss, old=None, dim=6))  
overlap_dict['000001'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,0,0,0,0,1]), overlap_ss, old=None, dim=6)) 
overlap_dict['000100'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,0,0,1,0,0]), overlap_ss, old=None, dim=6)) 
overlap_dict['001000'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,0,1,0,0,0]), overlap_ss, old=None, dim=6)) 

overlap_dict['100100'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([1,0,0,1,0,0]), overlap_ss, old=None, dim=6)) 
overlap_dict['010010'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,1,0,0,1,0]), overlap_ss, old=None, dim=6)) 
overlap_dict['001001'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,0,1,0,0,1]), overlap_ss, old=None, dim=6)) 

aa = exponents[0]
bb = exponents[1]
#Ax, Ay, Az = centers[0]
#Bx, By, Bz = centers[1]

Ax, Ay, Az = np.array([0.1,0.3,-0.849220457955])
Bx, By, Bz = np.array([0.4,-0.1, 0.849220457955])

#pi, pj, pk = angular_momentum[i]
#qi, qj, qk = angular_momentum[j]

# REAL px px
overlap = overlap_dict['100100'](Ax,Ay,Az,Bx,By,Bz,aa,bb)
print("REAL px px")
print(overlap)

# REAL py py
overlap = overlap_dict['010010'](Ax,Ay,Az,Bx,By,Bz,aa,bb)
print("REAL py py")
print(overlap)

# REAL pz pz
overlap = overlap_dict['001001'](Ax,Ay,Az,Bx,By,Bz,aa,bb)
print("REAL pz pz")
print(overlap)


# use px px function to get pz pz
overlap = overlap_dict['100100'](Az,Ay,Ax,Bz,By,Bx,aa,bb)
print(overlap)
overlap = overlap_dict['100100'](Az,Ax,Ay,Bz,Bx,By,aa,bb)
print(overlap)

# use pz pz function to get px px
overlap = overlap_dict['001001'](Az,Ay,Ax,Bz,By,Bx,aa,bb)
print(overlap)


# use pz pz function to get py py
overlap = overlap_dict['001001'](Az,Ax,Ay,Bz,Bx,By,aa,bb)
print(overlap)

# use px px function to get py py
overlap = overlap_dict['100100'](Ay,Ax,Az,By,Bx,Bz,aa,bb)
print(overlap)



new_overlap_dict = {}
# s s
new_overlap_dict['000000'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,0,0,0,0,0]), overlap_ss, old=None, dim=6))  
# p s
new_overlap_dict['100000'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([1,0,0,0,0,0]), overlap_ss, old=None, dim=6)) 
# p p
new_overlap_dict['100100'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([1,0,0,1,0,0]), overlap_ss, old=None, dim=6)) 
# dii dii
new_overlap_dict['200200'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([2,0,0,2,0,0]), overlap_ss, old=None, dim=6)) 
# dii dij
new_overlap_dict['200110'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([2,0,0,1,1,0]), overlap_ss, old=None, dim=6)) 
# dij dij
new_overlap_dict['110110'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([1,1,0,1,1,0]), overlap_ss, old=None, dim=6)) 
# dii s
new_overlap_dict['200000'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([2,0,0,0,0,0]), overlap_ss, old=None, dim=6)) 
# dij s
new_overlap_dict['110000'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([1,1,0,0,0,0]), overlap_ss, old=None, dim=6)) 
# dii p
new_overlap_dict['200100'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([2,0,0,1,0,0]), overlap_ss, old=None, dim=6)) 
# dij p
new_overlap_dict['110100'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([1,1,0,1,0,0]), overlap_ss, old=None, dim=6)) 


# use dxx dxx to get dzz dzz
new_overlap_dict['002002'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,0,2,0,0,2]), overlap_ss, old=None, dim=6)) 
overlap = new_overlap_dict['002002'](Ax,Ay,Az,Bx,By,Bz,aa,bb)
print('real dz dz')
print(overlap)


overlap = new_overlap_dict['200200'](Az,Ax,Ay,Bz,Bx,By,aa,bb)
print(overlap)

# use dxy dxy to get dyz dxz
new_overlap_dict['011101'] = jax.jit(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,1,1,1,0,1]), overlap_ss, old=None, dim=6)) 
overlap = new_overlap_dict['011101'](Ax,Ay,Az,Bx,By,Bz,aa,bb)
print('real dyz dxz')
print(overlap)

overlap = new_overlap_dict['110110'](Ay,Az,Ax,Bx,Bz,By,aa,bb)
print(overlap)


#for i in range(nbf):
#    for j in range(i+1):
#        aa = exponents[i]
#        bb = exponents[j]
#        Ax, Ay, Az = centers[i]
#        Bx, By, Bz = centers[j]
#        pi, pj, pk = angular_momentum[i]
#        qi, qj, qk = angular_momentum[j]
#        Na = normalize(aa, pi, pj, pk)
#        Nb = normalize(bb, qi, qj, qk)
#        ang_mom_vec = np.hstack((angular_momentum[i], angular_momentum[j]))
#        lookup = "".join(str(_) for _ in ang_mom_vec)
#        if lookup not in unique_lookups:
#            print(lookup)
#            unique_lookups.append(lookup)
#        
        #overlap = Na * Nb * overlap_dict[lookup](Ax,Ay,Az,Bx,By,Bz,aa,bb)
        #kinetic = Na * Nb * kinetic_dict[lookup](Ax,Ay,Az,Bx,By,Bz,aa,bb)
        #potential = Na * Nb * potential_dict[lookup](Ax,Ay,Az,Bx,By,Bz,geom,charge,aa,bb)

        #print(overlap,kinetic, potential)




