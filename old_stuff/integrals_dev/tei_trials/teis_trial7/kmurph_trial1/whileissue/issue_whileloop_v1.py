import jax 
from jax import lax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
from jax.experimental import loops
np.set_printoptions(linewidth=300)

@jax.jit
def test(A,B,C,D):
  with loops.Scope() as s:
    s.quantity = 0.0
    s.A, s.B, s.C, s.D = A, B, C, D
    s.i = s.A + s.B
    s.j = 2 * s.i + 1
    s.k = s.C + s.D
    s.l = 2 * s.k + 1
    s.m =  s.i + s.j + s.k + s.l

    s.i = s.A + s.B
    for _ in s.while_range(lambda: s.i > 0):
      s.j = 2 * s.i + 1
      for _ in s.while_range(lambda: s.j > 0):
        s.k = s.C + s.D 
        for _ in s.while_range(lambda: s.k > 0):
          s.l = 2 * s.k + 1 
          for _ in s.while_range(lambda: s.l > 0):
            s.m =  s.i + s.j + s.k + s.l  # Compiles and runs fine
            #s.m =  s.k + s.i + s.j + s.l  # Never finishes compiling, leaks memory until out of memory 
            for _ in s.while_range(lambda: s.m > 0):
              s.quantity += s.i + s.j + s.k + s.l + s.m
              s.m -= 1
            s.l -= 1
          s.k -= 1
        s.j -= 1
      s.i -= 1
    return s.quantity

print(test(1.,1.,1.,1.))

@jax.jit
def test(A,B,C,D):
  with loops.Scope() as s:
    #s.A, s.B, s.C, s.D = A, B, C, D
    s.i = A + B
    #s.j = 2 * s.i + 1
    #s.k = C + D
    #s.l = 2 * s.k + 1
    s.j = 0.
    s.k = 0.
    s.l = 0.
    s.quantity = 0.0

    s.i = A + B
    for _ in s.while_range(lambda: s.i > 0):
      s.j = 2 * s.i + 1
      for _ in s.while_range(lambda: s.j > 0):
        s.k = C + D 
        for _ in s.while_range(lambda: s.k > 0):
          s.l = 2 * s.k + 1 
          for _ in s.while_range(lambda: s.l > 0):
            #s.quantity =  s.i + s.j + s.k + s.l  # Compiles and runs fine
            s.quantity +=  s.k + s.i + s.j + s.l  # Never finishes compiling, leaks memory until out of memory 
            s.l -= 1
          s.k -= 1
        s.j -= 1
      s.i -= 1
    return s.quantity

print(test(1.,1.,1.,1.))
