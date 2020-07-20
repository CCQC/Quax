from functools import partial
from math import factorial

import jax.numpy as np
import matplotlib.pyplot as plt
from jax import jvp, vmap


def f(x):
  return 1./5 * x**3 + 3 * x**2 - x + 1.

x0 = 1.
t = np.linspace(-10, 10, 100)


# to make a first-order approx of f at x0, we can use `jvp` and compute both
# term0 = f(x) and term1 = f'(x) * v
def approx1(x0, v):
  term0, term1 = jvp(f, (x0,), (v,))
  return term0 + term1

plt.figure()
plt.plot(t, f(t), 'b-')
plt.plot(t, vmap(partial(approx1, x0))(t - x0), '--', color='orange')


# to make a second-order approx of f at x0, we can use `jvp` twice, though
# there's some redundant work being done that could be shared
def approx2(x0, v):
  term0, term1 = jvp(f, (x0,), (v,))
  term2 = jvp(lambda x: jvp(f, (x,), (v,))[1], (x0,), (v,))[1]
  return term0 + term1 + term2 / 2.

plt.figure()
plt.plot(t, f(t), 'b-')
plt.plot(t, vmap(partial(approx1, x0))(t - x0), '--', color='orange')
plt.plot(t, vmap(partial(approx2, x0))(t - x0), '--', color='green')


# a recursive definition shares some work
def taylor(f, order):
  def improve_approx(g, k):
    return lambda x, v: jvp_first(g, (x, v), v)[1] + f(x) / factorial(k)
  approx = lambda x, v: f(x) / factorial(order)
  for n in range(order):
    approx = improve_approx(approx, order - n - 1)
  return approx

def jvp_first(f, primals, tangent):
  x, xs = primals[0], primals[1:]
  return jvp(lambda x: f(x, *xs), (x,), (tangent,))


x0 = np.ones_like(t)
approx = taylor(f, 3)

plt.figure()
plt.plot(t, f(t), 'b-')
plt.plot(t, approx(x0, t - x0), '--', color='orange')
plt.show()
