import jax 
import jax.numpy as np
import numpy as onp
from jax.experimental import loops

# Question: defining s.i in terms of function args?
def test(a,b):
  with loops.Scope() as s:
    s.quantity = 0.0
    s.i = a + b
    s.j = 2 * s.i + 1
    s.k = 2 * s.j + 1

    for _ in s.while_range(lambda: s.i > 0):
      s.j = 2 * s.i + 1
      for _ in s.while_range(lambda: s.j > 0):
        s.k = 2 * s.j + 1 
        for _ in s.while_range(lambda: s.k > 0):
          #s.quantity += s.i + s.j + s.k  # This works!
          s.quantity += s.i + s.k + s.j   # This works! 
          s.k -= 1
        s.j -= 1
      s.i -= 1
    return s.quantity

print(test(3.,2.))

# The act of defining an s.a or s.b which shares the same name as the function arguments appears to be the issue
# it does not matter if they are even being used.

# Question: defining s.i in terms of s.a s.b 
def test(a,b):
  with loops.Scope() as s:
    s.quantity = 0.0

    s.dummy = 1. # define a scope variable that may or may not participate in the loop

    s.i = a + b
    s.j = 0.
    s.k = 0.

    for _ in s.while_range(lambda: s.i > 0):
      s.j = 2 * s.i + 1
      for _ in s.while_range(lambda: s.j > 0):
        s.k = 2 * s.j + 1 
        for _ in s.while_range(lambda: s.k > 0):
          #s.quantity += s.i + s.j + s.k            # This works!
          #s.quantity += s.i + s.k + s.j            # Runs indefinitely
          s.quantity += s.i + s.k + s.j + s.dummy   # This works 
          s.k -= 1
        s.j -= 1
      s.i -= 1
    return s.quantity


# Issue #1 : while-loops never terminate if you 1. define a loop scope variable which is UNUSED in the loop, and 2. your operations in the loop are not performed in a specific order (the nested loop order). In the example above, 
# (i + j + k works, i + k + j results in infinite loop). 

# HYPOTHESIS Issue #2 : nested while-loops never terminate if you...
    # 1. jit compile the function 
    # 2. define a nested loop scope variable in terms of values outside the scope (s.k = a + b), and 
    # 3. Sum quantities in certain orders (i + j + k works, but i + k + j runs indefinitely).

def test4(a,b):
  with loops.Scope() as s:
    s.quantity = 0.0
    s.i = a + b
    s.j = 2 * s.i + 1
    s.k = b + 2 
    #s.k = 2 * s.j + 1

    for _ in s.while_range(lambda: s.i > 0):
      s.j = 2 * s.i + 1
      for _ in s.while_range(lambda: s.j > 0):
        #s.k = b + 2 # NOTE this changed! 
        #s.k = 2 # NOTE this changed! 
        s.k = 2 * s.j + 1
        for _ in s.while_range(lambda: s.k > 0):
          #s.quantity += s.i + s.j + s.k # This works!
          s.quantity += s.i + s.k + s.j  # Runs indefinitely only when jit-compiled
          s.k -= 1
        s.j -= 1
      s.i -= 1
    return s.quantity

print(test4(3.,2.))
print(jax.jit(test4)(3.,2.))


