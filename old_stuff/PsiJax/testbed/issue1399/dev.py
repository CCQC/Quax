import jax
from jax import jit
from jax import jacrev
import jax.numpy as np
import numpy as onp
import time
from jax import lax

NUM_VAR = 10**4
NUM_ARG = 200

rng = onp.random.RandomState(0)  # NOTE: control RNG for reproducibility

var_onp = rng.rand(NUM_VAR)
var = np.asarray(var_onp) # allocating the variables to GRAM

def func(arg):
    divider = 0. # NOTE: I made these floats
    numerator = 0.

    def body_fun(carry, x):
        divider, numerator = carry
        temp = np.dot(x, var)
        temp1 = np.sin(temp)
        temp2 = np.cos(temp)

        divid = np.add(temp1, temp2)
        divid = np.power(divid, 2)
        divid = np.sum(divid)

        numer = np.add(temp1, temp2)
        numer = np.sum(numer)
        numer = np.power(numer, 2)
        numerator = np.add(numer, numerator)

        divider = np.add(divider, divid)

        new_carry = divider, numerator
        return new_carry, ()
    (divider, numerator), _ = lax.scan(body_fun, (divider, numerator), arg)
    divider = np.power(divider, 1/2)

    return np.log(np.divide(numerator, divider))

res = jax.jit(jax.grad(func))

# arg = jax.random.normal(jax.random.PRNGKey(1), (NUM_ARG, 0))
arg = rng.rand(NUM_ARG)
real_args = np.asarray(arg)
print("... ... ...Start calculating Gradient... ... ...")
start = time.time()
print(res(real_args))
end = time.time()
print("First Execution time: ", end-start, " s ... ... ...")


for step in range(10):
    arg = rng.rand(NUM_ARG)
    real_args = np.asarray(arg)
    # print("... ... ...Start calculating Gradient... ... ...")
    start = time.time()
    res(real_args).block_until_ready()  # NOTE: added!
    end = time.time()
    print(str(step) + "th Execution time: ", end-start, " s ... ... ...")
