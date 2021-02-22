import jax                        
import jax.numpy as np            
import numpy as onp               
from jax.experimental import loops

def test4(a,b):
  with loops.Scope() as s:
    s.quantity = 0.0
    s.i = a + b
    s.j = 2 * s.i + 1
    s.k = b + 2

    for _ in s.while_range(lambda: s.i > 0):
      s.j = 2 * s.i + 1
      for _ in s.while_range(lambda: s.j > 0):
        s.k = b + 2 # NOTE this changed! 
        for _ in s.while_range(lambda: s.k > 0):
          s.quantity += s.i + s.j + s.k # This works!
          #s.quantity += s.i + s.k + s.j  # Runs indefinitely only when jit-compiled
          s.k -= 1
        s.j -= 1
      s.i -= 1
    return s.quantity

print(test4(3.,2.))
print(jax.jit(test4)(3.,2.))  # never halts for s.i + s.k + s.j
