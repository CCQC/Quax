import jax 
import jax.numpy as np
from jax.experimental import loops
'''
Possible titles:
 An odd bug with nested while loops 
 Changing operation order in nested while loop causes infinite loop 
'''

# For my use case, I have a function containing nested for-loops, and the bounds of each loop are dependent on function arguments.
# As-is, this function is not jit-compilable, since I cannot pass abstract arguments to Python's `range`. 
# However, I can jit-compile (and therefore use vmap/scan/other awesome JAX stuff) if I use `lax.while_loop`   
# with the aid of the experimental.loops module for simplicity (thanks, @gnecula !)

# The followin


#@jax.jit
#def test(A,B,C,D):
#  with loops.Scope() as s:
#    s.quantity = 0.0
#    s.A, s.B, s.C, s.D = A, B, C, D
#    s.i = s.A + s.B
#    s.j = 2 * s.i + 1
#    s.k = s.C + s.D
#    #s.k = 2 * s.j + 1 
#    for _ in s.while_range(lambda: s.i > 0):
#      s.j = 2 * s.i + 1
#      for _ in s.while_range(lambda: s.j > 0):
#        s.k = s.C + s.D 
#        #s.k = 2 * s.j + 1 
#        for _ in s.while_range(lambda: s.k > 0):
#          s.quantity += s.i + s.j + s.k   # Works fine
#          #s.quantity += s.i + s.k + s.j   # Runs indefinitely, with gradual memory increase until out of memory
#          #s.quantity += s.j + s.i + s.k   # Works fine 
#          #s.quantity += s.k + s.i + s.j   # Runs indefinitely, with gradual memory increase until out of memory
#          #s.quantity += s.j + s.k + s.i    # Runs indefinitely, with gradual memory increase until out of memory
#          #s.quantity += s.k + s.i   
#          s.k -= 1
#        s.j -= 1
#      s.i -= 1
#    return s.quantity


def test(a,b):
  with loops.Scope() as s:
    s.quantity = 0.0
    #s.a, s.b = a, b
    #s.i = s.a + s.b

    #s.a, s.b = a, b
    s.i = a + b



    s.j = 2 * s.i + 1
    s.k = 2 * s.j + 1

    for _ in s.while_range(lambda: s.i > 0):
      s.j = 2 * s.i + 1
      for _ in s.while_range(lambda: s.j > 0):
        s.k = 2 * s.j + 1 
        for _ in s.while_range(lambda: s.k > 0):
          #s.quantity += s.i + s.j + s.k   # Works fine
          s.quantity += s.i + s.k + s.j   # Runs indefinitely, with gradual memory increase until out of memory
          #s.quantity += s.j + s.i + s.k   # Works fine 
          #s.quantity += s.k + s.i + s.j   # Runs indefinitely, with gradual memory increase until out of memory
          #s.quantity += s.j + s.k + s.i    # Runs indefinitely, with gradual memory increase until out of memory
          #s.quantity += s.k + s.i   
          s.k -= 1
        s.j -= 1
      s.i -= 1
    return s.quantity

print(test(1.,1.))
#print(test(1.,1.,1.,1.))
#print(test(2.,1.,2.,1.))
#print(test(2.,3.,3.,2.))

def control(a,b):
  quantity = 0
  i = a + b
  while i > 0:
    j = 2 * i + 1
    while j > 0:
      k = 2 * j + 1 
      while k > 0:
        quantity += i + j + k
        k -= 1
      j -= 1
    i -= 1
  return quantity

#for i in range(100000):
#for i in range(10000):
#    #control(1.,1.,1.,1.)
#    #control(2.,1.,2.,1.)
#    #control(2.,3.,3.,2.)
#    test(1.,1.,1.,1.)
#    test(2.,1.,2.,1.)
#    test(2.,3.,3.,2.)
#    #print(control(1.,1.,1.,1.))
#    #print(control(2.,1.,2.,1.))
#    #print(control(2.,3.,3.,2.))

#A = np.repeat(np.arange(10), 10000)
#B = np.repeat(np.arange(10), 10000)
#print(A.shape)

#result = jax.vmap(test, (0,0))(A,B)
#result = jax.vmap(test, (0,0))(A,B)


#for i in range(1



