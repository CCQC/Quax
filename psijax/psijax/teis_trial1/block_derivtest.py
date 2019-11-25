import jax
import jax.numpy as np
from jax import jacfwd as fwd
from integrals_utils import lower_take_mask, boys0

A = np.array([0.0,0.0,-0.849220457955])
B = A
C = np.array([0.0,0.0,0.849220457955])
D = C
alpha = 0.5
beta = 0.5
gamma = 0.4
delta = 0.4
c1,c2,c3,c4 = 1.0,1.0,1.0,1.0

args = (A, B, C, D, alpha, beta, gamma, delta, c1, c2, c3, c4)

def eri_ssss(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4):
    coeff = c1 * c2 * c3 * c4
    zeta = aa + bb
    eta = cc + dd
   
    factor = np.sqrt(2)*np.pi**(5/4)
    K_ab = (factor / zeta) * np.exp((-aa * bb / zeta) * np.dot(A-B,A-B))
    K_cd = (factor / eta) * np.exp((-cc * dd / eta) * np.dot(C-D,C-D))

    P = (aa * A + bb * B) / zeta
    Q = (cc * C + dd * D) / eta

    boys_arg = (zeta * eta / (zeta + eta)) * np.dot(P-Q,P-Q)
    ssss = coeff * (zeta + eta)**(-1/2) * K_ab * K_cd * boys0(boys_arg)
    return ssss

def eri_psss(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4):
    '''
    [a+1i b | cd] = 1/2a ( d/dA [ab|cd] + ai [a-1i b |cd] )
    px , py , pz
    '''
    factor = 1 / (2 * aa)
    dA = fwd(eri_ssss, 0)(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4)
    return factor * dA  

def eri_dsss(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4):
    '''
    [a+1i b | cd] = 1/2a ( d/dA [ab|cd] + ai [a-1i b |cd] )
    dxx , dxy , dxz
    dyx , dyy , dyz
    dzx , dzy , dzz
    '''
    factor = 1 / (2 * aa)
    dA = fwd(eri_psss, 0)(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4)
    #TODO add second term
    return factor * dA  

def eri_fsss(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4):
    factor = 1 / (2 * aa)
    dA = fwd(eri_dsss, 0)(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4)
    #TODO add second term
    return factor * dA

def eri_gsss(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4):
    factor = 1 / (2 * aa)
    dA = fwd(eri_fsss, 0)(A, B, C, D, aa, bb, cc, dd, c1, c2, c3, c4)
    #TODO add second term
    return factor * dA

def transfer_bra(upper, lower, args):
    '''
    Use horizontal recurrence relation to transfer angular momentum 
    't' times from left side of bra (center A) to right side of bra (center B)
    
    upper : array 
        The higher angular momentum term [(a+1i)b|cd] 
    lower : array
        The lower angular momentum term [ab|cd] 
    args: tuple
        A tuple of ERI arguments, defined elsewhere: (A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)

    '''
    AmB = args[0] - args[1]
    result = upper + np.broadcast_to(AmB, upper.shape) * np.broadcast_to(lower, upper.shape)
    return result

def transfer_ket(upper, lower, args):
    '''
    Use horizontal recurrence relation to transfer angular momentum 
    't' times from left side of ket (center C) to right side of ket (center D)
    
    upper : array 
        The higher angular momentum term [ab|(c+1i)d] 
    lower : array
        The lower angular momentum term [ab|cd] 
    args: tuple
        A tuple of ERI arguments, defined elsewhere: (A,B,C,D,aa,bb,cc,dd,c1,c2,c3,c4)

    '''
    AmB = args[2] - args[3]
    result = upper + np.broadcast_to(AmB, upper.shape) * np.broadcast_to(lower, upper.shape)
    return result

# To make (pp|pp), should just need
# (pp|ds) = transfer_bra(dsds, psds)
# (pp|pp) = transfer_ket(ppds, psps)


ppss = transfer_bra( eri_dsss(*args), eri_psss(*args), args)
dpss = transfer_bra( eri_fsss(*args), eri_dsss(*args), args)
print(ppss.shape)
print(dpss.shape)



px = fwd(eri_ssss, argnums=0) 
py = fwd(eri_ssss, argnums=1) 
pz = fwd(eri_ssss, argnums=2) 

dxx = fwd(px, argnums=(0))
dxy = fwd(px, argnums=(1))
dxz = fwd(px, argnums=(2))
dyy = fwd(py, argnums=(1))
dyz = fwd(py, argnums=(2))
dzz = fwd(pz, argnums=(2))

fxxx = fwd(dxx, argnums=(0))
fxxy = fwd(dxx, argnums=(1))
fxxz = fwd(dxx, argnums=(2))

fxyy = fwd(dxy, argnums=(1))
fxyz = fwd(dxy, argnums=(2))

fxzz = fwd(dxz, argnums=(2))

fyyy = fwd(dyy, argnums=(1))
fyyz = fwd(dyy, argnums=(2))

fyzz = fwd(dyz, argnums=(2))

fzzz = fwd(dzz, argnums=(2))






#def f(x,y,z):
#    return x**2, y**2, z**2



#
#print(f(np.array([0.0]),np.array([1.0]),np.array([2.0])))
#print(grad(np.array([0.0]),np.array([1.0]),np.array([2.0])))
#print(hess(np.array([0.0]),np.array([1.0]),np.array([2.0])))
#print('grad')
#for i in grad(np.array([0.0]),np.array([1.0]),np.array([2.0])):
#    print(i)
#print('hess')
#for i in hess(np.array([0.0]),np.array([1.0]),np.array([2.0])):
#    print(i)
#
#
#
#print(f(0.0,1.0,2.0))
#print('grad')
#for i in grad(0.0,1.0,2.0):
#    print(i)
#print('hess')
#for i in hess(0.0,1.0,2.0):
#    print(i)
#
