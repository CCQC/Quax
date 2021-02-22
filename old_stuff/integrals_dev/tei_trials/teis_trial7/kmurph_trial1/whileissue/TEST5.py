def control(a,b):                    
  quantity = 0                       
  i = a + b                          
  while i > 0:                       
    j = 2 * i + 1                    
    while j > 0:                     
      k = 2 * j + 1                  
      while k > 0:                   
        quantity += i + j + k        
        k -= 1                       
      j -= 1                         
    i -= 1                           
  return quantity                    


import jax                        
import jax.numpy as np            
import numpy as onp               
from jax.experimental import loops

def test(a,b):                                                
  with loops.Scope() as s:                                    
    s.quantity = 0.0
    s.a = a
    s.b = b                                          
    s.i = s.a + s.b                                               
    s.j = 2 * s.i + 1                                         
    s.k = 2 * s.j + 1                                         
                                                              
    for _ in s.while_range(lambda: s.i > 0):                  
      s.j = 2 * s.i + 1                                       
      for _ in s.while_range(lambda: s.j > 0):                
        s.k = 2 * s.j + 1                                     
        for _ in s.while_range(lambda: s.k > 0):              
          s.quantity += s.i + s.j + s.k                       
          s.k -= 1                                            
        s.j -= 1                                              
      s.i -= 1                                                
    return s.quantity                


#A = np.repeat(np.arange(10), 1000)
#B = np.repeat(np.arange(10), 1000)
#v_test = jax.vmap(test, (0,0))
#result1 = v_test(A,B).block_until_ready()
#A = onp.repeat(onp.arange(10), 1000)
#B = onp.repeat(onp.arange(10), 1000)
#result2 = [control(A[i],B[i]) for i in range(A.shape[0])]
#print(np.allclose(result1, np.asarray(result2)))


def test2(a,b):                                                
  with loops.Scope() as s:                                    
    s.quantity = 0.0
    s.a = a
    s.b = b                                          
    s.i = s.a + s.b                                               
    s.j = 2 * s.i + 1                                         
    s.k = 2 * s.j + 1                                         
                                                              
    for _ in s.while_range(lambda: s.i > 0):                  
      s.j = 2 * s.i + 1                                       
      for _ in s.while_range(lambda: s.j > 0):                
        s.k = 2 * s.j + 1                                     
        for _ in s.while_range(lambda: s.k > 0): 
          #s.quantity += s.i + s.j + s.k  # This works!
          s.quantity += s.i + s.k + s.j   # This runs indefinitely
          s.k -= 1                                            
        s.j -= 1                                              
      s.i -= 1                                                
    return s.quantity

#print(test2(3.,2.))                                       


def test3(a,b):
  with loops.Scope() as s:
    s.quantity = 0.0

    s.dummy = 0. # define a scope variable that may or may not participate in the loop

    s.i = a + b
    s.j = 2 * s.i + 1
    s.k = 2 * s.j + 1

    for _ in s.while_range(lambda: s.i > 0):
      s.j = 2 * s.i + 1
      for _ in s.while_range(lambda: s.j > 0):
        s.k = 2 * s.j + 1
        for _ in s.while_range(lambda: s.k > 0):
          #s.quantity += s.i + s.j + s.k            # This works!
          #s.quantity += s.i + s.k + s.j            # Runs indefinitely
          s.quantity += s.i + s.k + s.j + s.dummy   # This works!
          s.k -= 1
        s.j -= 1
      s.i -= 1
    return s.quantity

#print(test3(3.,2.))

def test4(a,b):
  with loops.Scope() as s:
    s.quantity = 0.0
    s.i = a + b
    s.j = 2 * s.i + 1
    s.k = b + 2

    for _ in s.while_range(lambda: s.i > 0):
      s.j = 2 * s.i + 1
      for _ in s.while_range(lambda: s.j > 0):
        s.k = b + 2 # NOTE this changed! 
        for _ in s.while_range(lambda: s.k > 0):
          s.quantity += s.i + s.j + s.k # This works!
          #s.quantity += s.i + s.k + s.j  # Runs indefinitely only when jit-compiled
          s.k -= 1
        s.j -= 1
      s.i -= 1
    return s.quantity

print(test4(3.,2.))
print(jax.jit(test4)(3.,2.))  # never halts for s.i + s.k + s.j


