from jax import vmap,lax
import jax.numpy as np

## works
#vmap_igamma = vmap(lax.igamma, (0,0))
#vmap_igamma(np.array([1.0,2.0]), np.array([0.5,0.6]))
#
## works
#vmap_betainc = vmap(lax.betainc, (0, 0, 0))
#vmap_betainc(np.array([1.0,2.0]), np.array([1.0,2.0]), np.array([1.0,2.0]))

# fails
#vmap_igamma = vmap(lax.igamma, (None,0))
#vmap_igamma(1.0, np.array([0.5,0.6]))

## fails
#vmap_igammac = vmap(lax.igammac, (None,0))
#vmap_igammac(1.0, np.array([0.5,0.6]))
#
## fails
#vmap_betainc = jax.vmap(lax.betainc, (None, None, 0))
#vmap_betainc(1.0, 2.0, np.array([1.0,2.0]))

# This behavior is not seen for other `naryop`'s in `jax/lax/lax.py`.
# works
vmap_pow = vmap(lax.pow, (None,0))
print(vmap_pow(1.0, np.array([0.5,0.6])))

vmap_atan2 = vmap(lax.atan2, (None,0))
print(vmap_atan2(1.0, np.array([0.5,0.6])))

# fails
#vmapped_igamma = jax.vmap(jax.lax.igamma, (0,None))

#vmap_igamma = jax.vmap(igamma, (None,0))
#print(vmap_igamma(1.0, np.array([1.0,2.0])))

#vmap_igamma = jax.vmap(igamma, (0,None))
#print(vmap_igamma(np.array([1.0,2.0]), 1.0))
