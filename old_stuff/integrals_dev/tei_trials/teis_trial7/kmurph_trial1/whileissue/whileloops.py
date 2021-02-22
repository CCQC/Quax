import jax 
from jax import lax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
from jax.experimental import loops
np.set_printoptions(linewidth=300)

@jax.jit
def test(lA,lB,lC,lD):
  with loops.Scope() as s:
    s.quantity = 0.0
    s.lA, s.lB, s.lC, s.lD = lA, lB, lC, lD
    s.l = s.lA + s.lB 
    s.r = np.floor(s.l/2) 
    s.lp = s.lC + s.lD 
    s.rp = np.floor(s.lp/2) 
    s.i =  np.floor((s.l - 2*s.r + s.lp - 2*s.rp) / 2)
    
    #s.i = s.lA + s.lB + s.lC + s.lD 
    for _ in s.while_range(lambda: s.l > -1):
      s.r = np.floor(s.l/2)
      for _ in s.while_range(lambda: s.r > -1):
        s.lp = s.lC + s.lD 
        for _ in s.while_range(lambda: s.lp > -1):
          s.rp = np.floor(s.lp/2)
          for _ in s.while_range(lambda: s.rp > -1):
            #s.i = np.floor((s.l + s.lp - 2 * s.r - 2 * s.rp) / 2)  # This has a slow compile
            s.i =  np.floor((s.l - 2*s.r + s.lp - 2*s.rp) / 2) # This compiles fine
            for _ in s.while_range(lambda: s.i > -1):
              s.quantity += s.l + s.r + s.lp + s.rp + s.i
              s.i -= 1
            s.rp -= 1
          s.lp -= 1
        s.r -= 1
      s.l -= 1

    return s.quantity

#print(test(3,3,3,3))
print(test(1.,1.,1.,1.))
#print(test(0,0,0,0))


@jax.jit
def test(A,B,C,D):
  with loops.Scope() as s:
    s.quantity = 0.0
    s.A, s.B, s.C, s.D = A, B, C, D
    s.l = s.lA + s.lB 
    s.r = np.floor(s.l/2) 
    s.lp = s.lC + s.lD 
    s.rp = np.floor(s.lp/2) 
    s.i =  np.floor((s.l - 2*s.r + s.lp - 2*s.rp) / 2)
    
    #s.i = s.lA + s.lB + s.lC + s.lD 
    for _ in s.while_range(lambda: s.l > -1):
      s.r = np.floor(s.l/2)
      for _ in s.while_range(lambda: s.r > -1):
        s.lp = s.lC + s.lD 
        for _ in s.while_range(lambda: s.lp > -1):
          s.rp = np.floor(s.lp/2)
          for _ in s.while_range(lambda: s.rp > -1):
            #s.i = np.floor((s.l + s.lp - 2 * s.r - 2 * s.rp) / 2)  # This has a slow compile
            s.i =  np.floor((s.l - 2*s.r + s.lp - 2*s.rp) / 2) # This compiles fine
            for _ in s.while_range(lambda: s.i > -1):
              s.quantity += s.l + s.r + s.lp + s.rp + s.i
              s.i -= 1
            s.rp -= 1
          s.lp -= 1
        s.r -= 1
      s.l -= 1

