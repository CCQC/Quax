import jax
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=500, edgeitems=10)

# Create random vector of size of lower triangle for NxN matrix
N = 20
tril_size = int((N**2 - N) / 2)
seed = jax.random.PRNGKey(0)
tril = jax.random.uniform(seed, (tril_size,))

def func(tril):
    y1 = np.sin(tril)
    y2 = y1**2
    y3 = y2 * 3
    # Populate a matrix with symmetric lower and upper triangle
    M = np.zeros((N,N))

    # METHOD 1: Many 'simple' calls to index_update
    idx_u = np.triu_indices(N,k=1)
    idx_l = np.tril_indices(N,k=-1)
    for i in range(y3.shape[0]):
        M = jax.ops.index_update(M, (idx_u[0][i], idx_u[1][i]), y3[i])
        M = jax.ops.index_update(M, (idx_l[0][i], idx_l[1][i]), y3[i])

    # METHOD 2: Two complex calls to index_update
    #M = jax.ops.index_update(M, np.triu_indices(N,k=1), y3)
    #M = jax.ops.index_update(M, np.tril_indices(N,k=-1), y3)

    # METHOD 3: One very complex call to index_update
    #triu_idx = np.triu_indices(N,k=1)
    #tril_idx = np.tril_indices(N,k=-1)
    #idx0 = np.hstack((triu_idx[0], tril_idx[0]))
    #idx1 = np.hstack((triu_idx[1], tril_idx[1]))
    #M = jax.ops.index_update(M, (idx0, idx1), np.repeat(y3,2))
    return np.sum(M)

#val = func(tril)
#grad = jax.jacfwd(func)(tril)
hess = jax.jacfwd(jax.jacfwd(func))(tril)
#cube = jax.jacfwd(jax.jacfwd(jax.jacfwd(func)))(tril)
#quar = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(func))))(tril)

def func2(tril):
    y1 = np.sin(tril)
    y2 = y1**2
    y3 = y2 * 3
    return np.sum(np.concatenate((y3, np.zeros((N)), y3)))
    #return np.concatenate((y3, np.zeros((N)), y3))

#val = func2(tril)
#print(val)
#grad = jax.jacrev(func2)(tril)
#hess = jax.jacfwd(jax.jacrev(func2))(tril)
#cube = jax.jacfwd(jax.jacfwd(jax.jacrev(func2)))(tril)
#quar = jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(func2))))(tril)


#    # Populate a matrix with symmetric lower and upper triangle
#    M = np.zeros((N,N))
#    M = jax.ops.index_update(M, np.triu_indices(N,k=1),  y3)
#    M = jax.ops.index_update(M, np.tril_indices(N,k=-1),  y3)
#    #return np.sum(np.dot(np.dot(M, M), M))
#    return np.sum(M)



