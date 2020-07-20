import jax 
import jax.numpy as np
import numpy as onp
from jax.experimental import loops

def test3(a,b):
  with loops.Scope() as s:
    s.quantity = 0.0

    s.dummy = 1. # define a scope variable that may or may not participate in the loop

    s.i = a + b
    s.j = 2 * s.i + 1
    s.k = 2 * s.j + 1

    for _ in s.while_range(lambda: s.i > 0):
      s.j = 2 * s.i + 1
      for _ in s.while_range(lambda: s.j > 0):
        s.k = 2 * s.j + 1
        for _ in s.while_range(lambda: s.k > 0):
          #s.quantity += s.i + s.j + s.k            # This works!
          #s.quantity += s.i + s.k + s.j            # Runs indefinitely
          s.quantity += s.i + s.k + s.j + s.dummy   # This works!
          s.k -= 1
        s.j -= 1
      s.i -= 1
    return s.quantity

print(test3(3.,2.))
