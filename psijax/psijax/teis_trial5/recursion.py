import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
#from integrals_utils import boys0, boys1, boys2, boys3, boys_old, taylor

'''
Sketch of Obara-Saika in lax.scan
centers = Ax,Ay,Az
          Bx,By,Bz
          Cx,Cy,Cz
          Dx,Dy,Dz

ang_mom = ax,ay,az
          bx,by,bz
          cx,cy,cz
          dx,dy,dz

exps = [aa,bb,cc,dd]

coeff

'''
xgrid_array = np.asarray(onp.arange(0, 30, 1e-5))
# Load boys function values, defined in the range x=0,30, at 1e-5 intervals
# NOTE: The factorial pre-factors and minus signs are appropriately fused into the boys function values
boys = np.asarray(onp.load('boys/boys_F0_F10_grid_0_30_1e5.npy'))

def boysn(x,n):
    interval = 1e-5 
    i = jax.lax.convert_element_type(np.round(x / interval), np.int64) # index of gridpoint nearest to x
    xgrid = xgrid_array[i] # grid x-value
    xx = x - xgrid
    # NOTE this scheme requires factorial terms/minus signs to NOT be built in, since they CHANGE based on value of n
    # Offset which F_n values to take based on value of n
    tmp = boys[:,i]
    f = tmp[np.arange(6) + n]
    #np.array([1,-1,0.5,-1/6,1/24,-1/120,1/720,1/5040,1/40320,1/362880,1/3628800])
    F = f[0] - xx * f[1] + 0.5*xx**2 * f[2] - (1/6) * xx**3 * f[3] + (1/24) * xx**4 * f[4] - (1/120) * xx**5 * f[5]
    return F

    #tmp = np.where(n == 0, np.ones(F0.shape[0]), np.pad(np.arange(F0.shape[0]-n) / (xx**n + 1e-12), (n,0), constant_values=0))
    #print(F0)
    #print(tmp)
    #tmp = np.pad(np.arange(F0.shape[0]-n) / (xx**n + 1e-12), (n,0), constant_values=0)
    #tmp = np.pad(np.arange(F0
    #tmp = np.pad(np.arange(F0.shape[0]-n) / (xx**n + 1e-12), (F0.shape[0] - n), constant_values=0)
    #tmp1 = np.arange(6-n) / (xx + 1e-12)
    #tmp = np.pad(tmp1, (F0.shape[0] - tmp1.shape[0], 0))
    #return np.sum(tmp * F0)
    
print(boysn(0.5, 0))
print(boysn(0.5, 1))
print(boysn(0.5, 2))


def general_braleft_tei(A,B,C,D,aa,bb,cc,dd,coeff, promotions):
    '''
    For now, just figure out how to promote only the x angular momentum of center A.
    assuming only the first term in VRR is valid
    '''
    # Preprocessing
    zeta = aa + bb
    eta = cc + dd
    K_ab = (1/zeta) * jax.lax.exp((-aa * bb * (1/zeta)) * jax.lax.dot(A-B,A-B))
    K_cd = (1/eta) * jax.lax.exp((-cc * dd * (1/eta)) * jax.lax.dot(C-D,C-D))
    P = (aa * A + bb * B) * (1/zeta) 
    Q = (cc * C + dd * D) * (1/eta)
    boys_arg = (zeta * eta / (zeta + eta)) * jax.lax.dot(P-Q,P-Q)
    ssss_0 = 2 * np.pi**(10/4) * coeff * (zeta + eta)**(-1/2) * K_ab * K_cd 

    # Scan setup
    init_integral_val = 0.
    first_int = ssss_0
    Px = P[0]
    Ax = A[0]
    init_carry = (init_integral_val, first_int, Px, Ax, boys_arg)

    def body(carry, i):
        integral, first_int, Px, Ax, boys_arg = carry
                                                      # would need to be jittable boys_n function with n arg, while carrying along m of first term, etc
        integral = integral + (Px - Ax) * first_int * boysn(boys_arg, 0)
        new_carry = (integral, first_int, Px, Ax, boys_arg)
        return new_carry, 0

    data, gunk = jax.lax.scan(body, init_carry, np.arange(promotions))
    return data[0] 

A = np.array([-0.4939594255,-0.2251760374, 0.3240754142])
B = np.array([ 0.4211401526, 1.8106751596,-0.1734137286])
C = np.array([-0.5304044183, 1.5987236612, 2.0935583523])
D = np.array([ 1.9190079941, 0.0838367286, 1.4064021040])

alpha = 0.2
beta = 0.3
gamma = 0.4
delta = 0.5
coeff = 1.0

args = (A, B, C, D, alpha, beta, gamma, delta, coeff, 5)

print(general_braleft_tei(*args))






