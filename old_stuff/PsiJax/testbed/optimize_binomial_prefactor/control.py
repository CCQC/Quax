import jax
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.experimental import loops


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

@jax.jit
def new_binomial_prefactor(s,l1,l2,PAx,PBx):
    """
    Eqn 15 Augsberger Dykstra 1989 J Comp Chem 11 105-111
    TODO also close to eqn 2.46 Ferman Valeev
    PAx, PBx are all vectors of components Pi-Ai, Pi-Bi raised to a power of angluar momentum.
    PAx = [PAx^0, PAx^1,...,PAx^max_am

    What sort of values does s take on?
    Values from l1 + l2 to zero, but it varies based on what stage of the loop you're in.


    f_k = 

    Can you make an array of indices such that youcan call address [s, l1, l2] and obtain possible values of t?

    Alternatively, if you can just collect the valid indices [t], [s-t], [s+t]

    """
    with loops.Scope() as L:
        L.total = 0.
        L.t = 0
        for _ in L.while_range(lambda: L.t < s + 1):
          #TEMP TODO rewrite this. The cond_range causes a huge overhead.
          # Try Valeev implementation
          for _ in L.cond_range(((s - l1) <= L.t) & (L.t <= l2)):
            L.total += binomials[l1,s-L.t] * binomials[l2,L.t] * PAx[l1-s + L.t] * PBx[l2 - L.t]
          L.t += 1
        return L.total

def test(s, l1, l2, PAx, PBx):
    total = 0.
    t = 0
    #print('s',s, 'l1', l1, 'l2', l2)
    while t < s + 1:
        if (s-l1 <= t) and (t <= l2):
            total += binomials[l1,s-t] * binomials[l2,t] * PAx[l1-s + t] * PBx[l2 - t]
            #print(t, s - t, l1-s+t, l2-t)
            #print('t',t)
        t += 1
    return total

def valeev1(k, l1, l2, PAx, PBx):
    # ?????
    total = 0.
    for i in range(k+1):
      for li in range(l1,k+1):
        for j in range(k+1):
          for lj in range(l2,k+1):
            total += PAx[li-i] * binomials[li,i] * PBx[lj-j] * binomials[lj,j]
    return total

def valeev2(k, l1, l2, PAx, PBx):
    # THIS WORKS! And no boolean checks!
    total = 0.
    # You could definitely preconstruct arr[k,l1,l2]--> q, q_final
    # Or better yet, arr[k,l1,l2] -> all [i,j]
    q = max(-k, k-2*l2)
    q_final = min(k, 2*l1 - k)
    while q <= q_final:
        i = (k+q)//2
        j = (k-q)//2
        total += PAx[l1-i] * binomials[l1,i] * PBx[l2-j] * binomials[l2,j]
        q += 2
    return total

@jax.jit
def jax_valeev2(k, l1, l2, PAx, PBx):
    """
    Fermann, Valeev 2.46
    """
    q = jax.lax.max(-k, k-2*l2)
    q_final = jax.lax.min(k, 2*l1-k)
    with loops.Scope() as L:
      L.total = 0.
      L.q = q
      for _ in L.while_range(lambda: L.q <= q_final):
        i = (k+L.q)//2
        j = (k-L.q)//2
        L.total += PAx[l1-i] * binomials[l1,i] * PBx[l2-j] * binomials[l2,j]
        L.q += 2
    return L.total
        
    #while q <= q_final:


PAx = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
PBx = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7])

#print(valeev1(2,3,3,PAx,PBx))
for i in range(100000):
  jax_valeev2(3,3,3,PAx,PBx)

#for i in range(100000):
#  new_binomial_prefactor(3,3,3,PAx,PBx)

#print(test(2,3,3,PAx,PBx))
#print(valeev2(2,3,3,PAx,PBx))


#print("---------")
#test(0,3,0)
#print("---------")
#test(1,3,0)
#print("---------")
#test(2,3,0)
#print("---------")
#test(3,3,0)
#print("---------")
#
#print("---------")
#test(0,3,1)
#print("---------")
#test(1,3,1)
#print("---------")
#test(2,3,1)
#print("---------")
#test(3,3,1)
#print("---------")
#test(4,3,1)
#print("---------")
#
#print("---------")
#test(0,3,2)
#print("---------")
#test(1,3,2)
#print("---------")
#test(2,3,2)
#print("---------")
#test(3,3,2)
#print("---------")
#test(4,3,2)
#print("---------")
#test(5,3,2)
#print("---------")
#
#print("---------")
#test(0,3,3)
#print("---------")
#test(1,3,3)
#print("---------")
#test(2,3,3)
#print("---------")
#test(3,3,3)
#print("---------")
#test(4,3,3)
#print("---------")
#test(5,3,3)
#print("---------")
#test(6,3,3)


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
