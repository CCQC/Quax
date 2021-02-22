

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
