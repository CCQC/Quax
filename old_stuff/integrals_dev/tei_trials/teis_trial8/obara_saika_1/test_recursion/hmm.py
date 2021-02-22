import jax
from jax.experimental import loops
import jax.numpy as np

#def factorial(x):
#    return jax.lax.cond(x < 1, x, lambda x: x * factorial(x - 1), x, lambda x: 1)


#@jax.jit
def while_factorial(x):
    with loops.Scope() as s:
        s.x = x
        s.result = 1
        for _ in s.while_range(lambda: s.x > 0):
            s.result *= s.x
            s.x -= 1
        return s.result

#print(factorial(3))
print(while_factorial(3))

#@jax.jit
def factorial(n):
  n = n.astype(float)
  return jax.lax.exp(jax.lax.lgamma(n + 1))


for i in range(10):
    for j in range(100):
        factorial(i)
        #while_factorial(i)
