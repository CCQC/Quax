
import jax.numpy as np
import jax


def func1(a,b):
    return a**2 + b**2

def func2(a,b):
    return np.array([2*a + 2*b, 2*a + 2*b])

def func3(a,b):
    return np.array([10*a + 10*b, 10*a + 10*b, 10*a + 10*b])

def mapper(a,b,which):
    val = np.where(which == 1, np.pad(func1(a,b), (0,2)),
          np.where(which == 2, np.pad(func2(a,b), (0,1)),
          np.where(which == 3, func3(a,b), np.zeros(3))))
    return val

a = np.arange(9)
b = np.arange(9)
c = np.array([1,2,3,1,2,3,1,2,3])

func = jax.vmap(mapper, (0,0,0))

print(func(a,b,c))

