import numpy as np

dim = 100
G = np.random.rand(dim,dim,dim,dim)
C = np.random.rand(dim,dim)
#tmp = np.random.rand(dim,dim)
# Make C symmetric
#C = np.dot(tmp, tmp.T) 

# correct!
def vanilla(G, C):                                                      
    """                                                                                
    Transform TEI's to MO basis.                                                       
    This algorithm is worse than below, since it creates intermediate arrays in memory.
    """                                                                                
    newG = np.einsum('pqrs, pP, qQ, rR, sS -> PQRS', G, C, C, C, C, optimize='optimal')  
    return newG                                                                           

# correct!
def new1(G,C):
    newG = np.zeros((dim,dim,dim,dim))
    for i in range(dim):
      for j in range(dim):
        newG[i,:,:,:] += C[j,i] * np.einsum('qrs, qQ, rR, sS -> QRS', G[j,:,:,:], C, C, C, optimize='optimal')  
    return newG


import jax.numpy as jnp
import jax
from jax.experimental import loops

def new2(G,C):
    with loops.Scope() as s:
      s.newG = jnp.zeros((dim,dim,dim,dim))
      for i in s.range(dim): 
        for j in s.range(dim): 
          val = C[j,i] * jnp.einsum('qrs, qQ, rR, sS -> QRS', G[j,:,:,:], C, C, C, optimize='optimal')  
          s.newG = jax.ops.index_add(s.newG, jax.ops.index[i,:,:,:], val)
      return s.newG

def new3(G, C):                                                      
    """                                                                                
    Transform TEI's to MO basis.                                                       
    This algorithm is worse than below, since it creates intermediate arrays in memory.
    """                                                                                
    newG = jnp.einsum('pqrs, pP, qQ, rR, sS -> PQRS', G, C, C, C, C, optimize='optimal')  
    return newG                                                                           

def new4(G, C):                                                      
    with loops.Scope() as s:
      s.newG = jnp.zeros((dim,dim,dim,dim))
      s.temp = jnp.zeros((dim,dim,dim,dim))  
      s.temp2 = jnp.zeros((dim,dim,dim,dim))  
      s.temp3 = jnp.zeros((dim,dim,dim,dim))  
      for i in s.range(0,dim):
          for m in s.range(0,dim):
              s.temp= jax.ops.index_add(s.temp, jax.ops.index[i,:,:,:], C[m,i]*G[m,:,:,:])
          for j in s.range(0,dim):
              for n in s.range(0,dim):
                  s.temp2 = jax.ops.index_add(s.temp2, jax.ops.index[i,j,:,:], C[n,j]*s.temp[i,n,:,:])
              for k in s.range(0,dim):
                  for o in s.range(0,dim):
                      s.temp3 = jax.ops.index_add(s.temp3, jax.ops.index[i,j,k,:], C[o,k]*s.temp2[i,j,o,:])
                  for l in s.range(0,dim):
                      for p in s.range(0,dim):
                          s.newG = jax.ops.index_add(s.newG, jax.ops.index[i,j,k,l], C[p,l]*s.temp3[i,j,k,p] )
      return s.newG

def new5(G,C):
    with loops.Scope() as s:
      s.newG = jnp.zeros((dim,dim,dim,dim))
      for i in s.range(0,dim):  
        for j in s.range(0,dim):  
          for k in s.range(0,dim):  
            for l in s.range(0,dim):  
              for m in s.range(0,dim):  
                for n in s.range(0,dim):  
                  for o in s.range(0,dim):  
                    for p in s.range(0,dim):  
                      s.newG = jax.ops.index_add(s.newG, jax.ops.index[i,j,k,l], C[m,i]*C[n,j]*C[o,k]*C[p,l]*G[m,n,o,p])
      return s.newG


#def new2(G,C):
#    newG = np.zeros((dim,dim,dim,dim))
#    temp = np.zeros((dim,dim,dim,dim))  
#    temp2 = np.zeros((dim,dim,dim,dim))  
#    temp3 = np.zeros((dim,dim,dim,dim))  
#    for i in range(0,dim):
#        for m in range(0,dim):
#            temp[i,:,:,:] += C[i,m]*G[m,:,:,:]
#        for j in range(0,dim):
#            for n in range(0,dim):
#                temp2[i,j,:,:] += C[j,n]*temp[i,n,:,:]
#            for k in range(0,dim):
#                for o in range(0,dim):
#                    temp3[i,j,k,:] += C[k,o]*temp2[i,j,o,:] 
#                for l in range(0,dim):
#                    for p in range(0,dim):
#                        newG[i,j,k,l] += C[l,p]*temp3[i,j,k,p]  
#    return newG


#vanilla(G,C)
#new1(G,C)
#new2(jnp.asarray(G),jnp.asarray(C))
new3(jnp.asarray(G),jnp.asarray(C))

#new5(jnp.asarray(G),jnp.asarray(C))

#vanilla(G,C)
#new4(jnp.asarray(G),jnp.asarray(C))

#print(np.allclose(vanilla(G,C), new1(G,C)))
#print(np.allclose(vanilla(G,C), new2(jnp.asarray(G),jnp.asarray(C))))
#print(np.allclose(vanilla(G,C), new3(jnp.asarray(G),jnp.asarray(C))))
#print(np.allclose(vanilla(G,C), new4(jnp.asarray(G),jnp.asarray(C))))
#print(np.allclose(vanilla(G,C), new5(jnp.asarray(G),jnp.asarray(C))))

#print(np.allclose(vanilla(G,C), new1(G,C)))
#print(np.allclose(vanilla(G,C), new2(G,C)))
