import jax
import jax.numpy as np


seed = jax.random.PRNGKey(0)
tmp = jax.random.uniform(seed, (10000,10000))
a = np.dot(tmp, tmp.T) 


vv = lambda x, y: np.vdot(x,y)
mv = jax.vmap(vv, (0, None), 0)
mm = jax.vmap(mv, (None, 1), 1)


#t1 = mm(a,a)
t2 = np.einsum('ij,jk->ik', a,a)
#print(t1)
#print(t2)

