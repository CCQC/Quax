import jax
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.experimental import loops

def boys(m,x,eps=1e-12):
    return 0.5 * (x + eps)**(-(m + 0.5)) * jax.lax.igamma(m + 0.5, x + eps) \
           * np.exp(jax.lax.lgamma(m + 0.5))

def binomial_prefactor(k, l1, l2, PAx, PBx):
    """
    Function to binomial prefactor, commonly denoted f_k()
    Fermann, Valeev 2.46
    Similar equivalent form in eqn 15 Augsberger Dykstra 1989 J Comp Chem 11 105-111
    PAx, PBx are all vectors of components Pi-Ai, Pi-Bi raised to a power of angluar momentum.
    PAx = [PAx^0, PAx^1,...,PAx^max_am
    """
    q = jax.lax.max(-k, k-2*l2)
    q_final = jax.lax.min(k, 2*l1-k)
    with loops.Scope() as L:
      L.total = 0.
      L.q = q
      for _ in L.while_range(lambda: L.q <= q_final):
        i = (k+L.q)//2
        j = (k-L.q)//2
        L.total += PAx[l1-i] * PBx[l2-j] * binomials[l1,i] * binomials[l2,j]
        L.q += 2
    return L.total

#def factorial(n):
#  '''Note: switch to float for high values (n>20) for stability'''
#  with loops.Scope() as s:
#    s.result = 1
#    s.k = 1
#    for _ in s.while_range(lambda: s.k < n + 1):
#      s.result *= s.k
#      s.k += 1
#    return s.result

def gaussian_product(alpha1,A,alpha2,B):
    '''Gaussian product theorem. Returns center.'''
    return (alpha1*A+alpha2*B)/(alpha1+alpha2)
 
def find_unique_shells(nshells):
    '''Find shell quartets which correspond to corresponding to unique two-electron integrals, i>=j, k>=l, IJ>=KL'''
    v = onp.arange(nshells,dtype=np.int16) 
    indices = old_cartesian_product(v,v,v,v)
    cond1 = (indices[:,0] >= indices[:,1]) & (indices[:,2] >= indices[:,3]) 
    cond2 = indices[:,0] * (indices[:,0] + 1)/2 + indices[:,1] >= indices[:,2] * (indices[:,2] + 1)/2 + indices[:,3]
    mask = cond1 & cond2 
    return np.asarray(indices[mask,:])

def cartesian_product(*arrays):
    '''JAX-friendly version of cartesian product. Same order as other function, more memory requirements though.'''
    tmp = np.asarray(np.meshgrid(*arrays, indexing='ij')).reshape(len(arrays),-1).T
    return np.asarray(tmp)

def am_vectors(am, length=3):
    '''
    Builds up all possible angular momentum component vectors of with total angular momentum 'am'
    am = 2 ---> [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    Returns a generator which must be converted to an iterable,
    for example, call the following: [list(i) for i in am_vectors(2)]

    Works by building up each possibility :
    For a given value in reversed(range(am+1)), find all other possible values for other entries in length 3 vector
     value     am_vectors(am-value,length-1)    (value,) + permutation
       2 --->         [0,0]                 ---> [2,0,0] ---> dxx
       1 --->         [1,0]                 ---> [1,1,0] ---> dxy
         --->         [0,1]                 ---> [1,0,1] ---> dxz
       0 --->         [2,0]                 ---> [0,2,0] ---> dyy
         --->         [1,1]                 ---> [0,1,1] ---> dyz
         --->         [0,2]                 ---> [0,0,2] ---> dzz
    '''
    if length == 1:
        yield (am,)
    else:
        # reverse so angular momentum order is canonical, e.g., dxx dxy dxz dyy dyz dzz
        for value in reversed(range(am + 1)):
            for permutation in am_vectors(am - value,length - 1):
                yield (value,) + permutation

# Need to store factorials up to l1 + l2 + l3 + l4 + 1
# support for h functions requires up to 21!, we add a one more to be safe 
factorials = np.array([1.0000000000000000e0, 1.0000000000000000e0, 2.0000000000000000e0,
                       6.0000000000000000e0, 2.4000000000000000e1, 1.2000000000000000e2,
                       7.2000000000000000e2, 5.0400000000000000e3, 4.0320000000000000e4,
                       3.6288000000000000e5, 3.6288000000000000e6, 3.9916800000000000e7,
                       4.7900160000000000e8, 6.2270208000000000e9, 8.7178291200000000e10,
                       1.3076743680000000e12,2.0922789888000000e13,3.5568742809600000e14,
                       6.4023737057280000e15,1.2164510040883200e17,2.4329020081766400e18],dtype=int)
                       #6.4023737057280000e15,1.2164510040883200e17,2.4329020081766400e18,
                       #5.1090942171709440e19,1.1240007277776077e21,2.5852016738884978e22,
                       #6.2044840173323941e23,1.5511210043330986e25,4.0329146112660565e26],dtype=int)


# Double factorials for overlap/kinetic. 
# We need 0!! to (l1+l2+1+2)!! (the plus 2 is for kinetic components) 
# but sometimes we index -1, so put a 1 at the end.
double_factorials = np.array([1,1,2,3,8,15,48,105,384,945,3840,10395,46080,135135,645120,2027025,10321920,1],dtype=int)
 
# All elements for a,b in which satisfy a! / (b! (a-2b)!)
# factorial(a) / factorial(b) / factorial(a-2*b)
# Must support up to L = l1 + l2 + l3 + l4 on row dimension, L/2 col dimension 
fact_ratio2 = np.array([[1,  0,     0,       0,         0,          0,            0,             0,               0,               0,                0,                0,                 0,0],
                        [1,  0,     0,       0,         0,          0,            0,             0,               0,               0,                0,                0,                 0,0],
                        [1,  2,     0,       0,         0,          0,            0,             0,               0,               0,                0,                0,                 0,0],
                        [1,  6,     0,       0,         0,          0,            0,             0,               0,               0,                0,                0,                 0,0],
                        [1, 12,    12,       0,         0,          0,            0,             0,               0,               0,                0,                0,                 0,0],
                        [1, 20,    60,       0,         0,          0,            0,             0,               0,               0,                0,                0,                 0,0],
                        [1, 30,   180,     120,         0,          0,            0,             0,               0,               0,                0,                0,                 0,0],
                        [1, 42,   420,     840,         0,          0,            0,             0,               0,               0,                0,                0,                 0,0],
                        [1, 56,   840,    3360,      1680,          0,            0,             0,               0,               0,                0,                0,                 0,0],
                        [1, 72,  1512,   10080,     15120,          0,            0,             0,               0,               0,                0,                0,                 0,0],
                        [1, 90,  2520,   25200,     75600,      30240,            0,             0,               0,               0,                0,                0,                 0,0],
                        [1,110,  3960,   55440,    277200,     332640,            0,             0,               0,               0,                0,                0,                 0,0],
                        [1,132,  5940,  110880,    831600,    1995840,       665280,             0,               0,               0,                0,                0,                 0,0],
                        [1,156,  8580,  205920,   2162160,    8648640,      8648640,             0,               0,               0,                0,                0,                 0,0],
                        [1,182, 12012,  360360,   5045040,   30270240,     60540480,      17297280,               0,               0,                0,                0,                 0,0],
                        [1,210, 16380,  600600,  10810800,   90810720,    302702400,     259459200,               0,               0,                0,                0,                 0,0],
                        [1,240, 21840,  960960,  21621600,  242161920,   1210809600,    2075673600,       518918400,               0,                0,                0,                 0,0],
                        [1,272, 28560, 1485120,  40840800,  588107520,   4116752640,   11762150400,      8821612800,               0,                0,                0,                 0,0],
                        [1,306, 36720, 2227680,  73513440, 1323241920,  12350257920,   52929676800,     79394515200,     17643225600,                0,                0,                 0,0],
                        [1,342, 46512, 3255840, 126977760, 2793510720,  33522128640,  201132771840,    502831929600,    335221286400,                0,                0,                 0,0],
                        [1,380, 58140, 4651200, 211629600, 5587021440,  83805321600,  670442572800,   2514159648000,   3352212864000,     670442572800,                0,                 0,0],
                        [1,420, 71820, 6511680, 341863200,10666131840, 195545750400, 2011327718400,  10559470521600,  23465490048000,   14079294028800,                0,                 0,0],
                        [1,462, 87780, 8953560, 537213600,19554575040, 430200650880, 5531151225600,  38718058579200, 129060195264000,  154872234316800,   28158588057600,                 0,0],
                        [1,506,106260,12113640, 823727520,34596555840, 899510451840,14135164243200, 127216478188800, 593676898214400, 1187353796428800,  647647525324800,                 0,0],
                        [1,552,127512,16151519,1235591279,59308381439,1799020903680,33924394183680, 381649434566400,2374707592857600, 7124122778572800, 7771770303897600,  1295295050649600,0],
                        [1,600,151800,21252000,1817046000,98847302400,3459655584000,77100895872000,1060137318240000,8481098545920000,35620613892864000,64764752532480000, 32382376266240000,0]],dtype=int)

# Binomial Coefficients
# C = factorial(n) // (factorial(k) * factorial(n-k))
# Minimum required dimension is (max_am * 2, max_am)
binomials = np.array([[1, 1,  0,  0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0], 
                      [1, 1,  0,  0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 2,  1,  0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 3,  3,  1,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 4,  6,  4,   1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 5, 10, 10,   5,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 6, 15, 20,  15,    6,    1,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 7, 21, 35,  35,   21,    7,    1,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 8, 28, 56,  70,   56,   28,    8,    1,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 9, 36, 84, 126,  126,   84,   36,    9,    1,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1,10, 45,120, 210,  252,  210,  120,   45,   10,    1,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1,11, 55,165, 330,  462,  462,  330,  165,   55,   11,    1,    0,    0,    0,   0,  0,  0, 0,0],
                      [1,12, 66,220, 495,  792,  924,  792,  495,  220,   66,   12,    1,    0,    0,   0,  0,  0, 0,0],
                      [1,13, 78,286, 715, 1287, 1716, 1716, 1287,  715,  286,   78,   13,    1,    0,   0,  0,  0, 0,0],
                      [1,14, 91,364,1001, 2002, 3003, 3432, 3003, 2002, 1001,  364,   91,   14,    1,   0,  0,  0, 0,0],
                      [1,15,105,455,1365, 3003, 5005, 6435, 6435, 5005, 3003, 1365,  455,  105,   15,   1,  0,  0, 0,0],
                      [1,16,120,560,1820, 4368, 8008,11440,12870,11440, 8008, 4368, 1820,  560,  120,  16,  1,  0, 0,0],
                      [1,17,136,680,2380, 6188,12376,19448,24310,24310,19448,12376, 6188, 2380,  680, 136, 17,  1, 0,0],
                      [1,18,153,816,3060, 8568,18564,31824,43758,48620,43758,31824,18564, 8568, 3060, 816,153, 18, 1,0],
                      [1,19,171,969,3876,11628,27132,50388,75582,92378,92378,75582,50388,27132,11628,3876,969,171,19,1]], dtype=int)

# Angular momentum distribution combinations, up to max_am=5, (h functions)
angular_momentum_combinations = np.array([
[0,0,0], 
[1,0,0],[0,1,0],[0,0,1],
[2,0,0],[1,1,0],[1,0,1],[0,2,0],[0,1,1],[0,0,2], 
[3,0,0],[2,1,0],[2,0,1],[1,2,0],[1,1,1],[1,0,2],[0,3,0],[0,2,1],[0,1,2],[0,0,3], 
[4,0,0],[3,1,0],[3,0,1],[2,2,0],[2,1,1],[2,0,2],[1,3,0],[1,2,1],[1,1,2],[1,0,3],[0,4,0],[0,3,1],[0,2,2],[0,1,3],[0,0,4], 
[5,0,0],[4,1,0],[4,0,1],[3,2,0],[3,1,1],[3,0,2],[2,3,0],[2,2,1],[2,1,2],[2,0,3],[1,4,0],[1,3,1],[1,2,2],[1,1,3],[1,0,4],[0,5,0],[0,4,1],[0,3,2],[0,2,3],[0,1,4],[0,0,5]], dtype=int)

# The first index of angular_momentum_combinations which corresponds to beginning of s-class, p-class, d-class, f-class, g-class, h-class
am_leading_indices = np.array([0,1,4,10,20,35,56], dtype=int)

# Powers of negative one, need indices up to l1 + l2 + l3 + l4 = 20 for h functions
neg_one_pow = np.array([1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1])

