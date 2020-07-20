import jax
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)

@jax.jit
def boys(x,n):
    return 0.5 * (x + 1e-12)**(-(n + 0.5)) * jax.lax.igamma(n + 0.5, x + 1e-12) * np.exp(jax.lax.lgamma(n + 0.5))

# One million evaluations of the boys function.
N = 1000000
seed = jax.random.PRNGKey(0)
n_vals = jax.random.randint(seed, (N,), 0, 10)
x_vals = jax.random.uniform(seed, (N,), minval=0.0, maxval=35.0)

# 1. naive for-loop no jit 
# Too slow, take 10,000 and multiply by 100 to get 1 million. 
#for i in range(10000):
#    x = boys(x_vals[i], n_vals[i])
# result: 2,400 seconds

# 2. naive for-loop with jit  
#for i in range(10000):
#    x = boys(x_vals[i], n_vals[i])
# result: 1,400 seconds

# 3. vmap and jit
#vmap_boysn = jax.vmap(boys, (0,0)) 
#result = vmap_boysn(x_vals, n_vals)
# result: 1.89 seconds

# 4. just send the arrays through with jit you dummy
result = boys(x_vals, n_vals)
# result: 1.89 seconds




# can you use loops with variable input sizes, and do they benefit from jit compiles?
#@jax.jit
#def test(aa,bb,cc,dd):
#    val = 0
#    for i in range(aa):
#        for j in range(bb):
#            for k in range(cc):
#                for l in range(dd):
#                    val = val + np.pi * i + j - k * l
#    return val



