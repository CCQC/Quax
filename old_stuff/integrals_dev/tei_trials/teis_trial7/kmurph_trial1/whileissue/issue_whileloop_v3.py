import jax 
from jax import lax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
from jax.experimental import loops
np.set_printoptions(linewidth=300)

#Two loops with coupled variables: no issue

@jax.jit
def test(A,B,C,D):
  with loops.Scope() as s:
    s.quantity = 0.0
    s.A, s.B, s.C, s.D = A, B, C, D
    s.i = s.A + s.B
    s.j = 2 * s.i + 1

    for _ in s.while_range(lambda: s.i > 0):
      s.j = 2 * s.i + 1
      for _ in s.while_range(lambda: s.j > 0):
        s.quantity += s.i + s.j  
        #s.quantity += s.j + s.i  
        s.j -= 1
      s.i -= 1
    return s.quantity


print(test(1.,1.,1.,1.))


#def control(A,B,C,D):
#  quantity = 0
#  i = A + B
#  while i > 0:
#    j = 2 * i + 1
#    while j > 0:
#      k = C + D 
#      while k > 0:
#        #quantity += i + j + k
#        quantity += i + k + j
#        k -= 1
#      j -= 1
#    i -= 1
#  return quantity
#
#print(control(1.,1.,1.,1.))
