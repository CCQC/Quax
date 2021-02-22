
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
          s.quantity += s.i + s.j
          #s.quantity += s.i 
          #s.quantity += s.k
          #s.quantity += s.j            # Runs indefinitely
          #s.quantity += s.i + s.k + s.j + s.dummy   # This works!
          s.k -= 1
        s.j -= 1
      s.i -= 1
    return s.quantity

print(test3(3.,2.))

#def test4(a,b):
#  with loops.Scope() as s:
#    s.quantity = 0.0
#    s.i = a + b
#    s.j = 2 * s.i + 1
#    s.k = b + 2
#
#    for _ in s.while_range(lambda: s.i > 0):
#      s.j = 2 * s.i + 1
#      for _ in s.while_range(lambda: s.j > 0):
#        s.k = b + 2 # NOTE this changed! 
#        for _ in s.while_range(lambda: s.k > 0):
#          s.quantity += s.i + s.j + s.k # This works!
#          #s.quantity += s.i + s.k + s.j  # Runs indefinitely only when jit-compiled
#          s.k -= 1
#        s.j -= 1
#      s.i -= 1
#    return s.quantity
#
#print(test4(3.,2.))
#print(jax.jit(test4)(3.,2.))  # never halts for s.i + s.k + s.j
#
#
#def test_laxwhile(a,b):
#    quantity = 0.
#    i = a + b
#    j = 2 * i + 1
#    k = 2 * j + 1
#
#    condfun_1 = lambda inp: inp[1] > 0. # i > 0.
#    condfun_2 = lambda inp: inp[2] > 0. # j > 0.
#    condfun_3 = lambda inp: inp[3] > 0. # k > 0.
#
#    def bodyfun_1(inp):
#        quantity, i, j, k = inp
#        j = 2 * i + 1
#        def bodyfun_2(inp):
#            quantity, i, j, k = inp
#            k = 2 * j + 1
#            def bodyfun_3(inp):
#                quantity, i, j, k = inp
#                #quantity += i + j + k  # This works!
#                quantity += i + k + j   # This works!
#                k -= 1.
#                return (quantity, i, j, k)
#            result = jax.lax.while_loop(condfun_3, bodyfun_3, (quantity,i,j,k))
#            quantity = result[0]
#            j -= 1.
#            return (quantity, i, j, k)
#        result = jax.lax.while_loop(condfun_2, bodyfun_2, (quantity,i,j,k))
#        quantity = result[0]
#        i -= 1.
#        return (quantity, i, j, k)
#
#    result = jax.lax.while_loop(condfun_1, bodyfun_1, (quantity,i,j,k))
#    return result[0]
#
#print(test_laxwhile(3.,2.))
#print(control(3.,2.))
#print(test(3.,2.))
#
