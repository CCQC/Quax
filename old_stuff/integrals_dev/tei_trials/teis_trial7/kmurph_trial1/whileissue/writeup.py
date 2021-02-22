'''
Possible titles:
 An odd bug with nested while loops 
 Changing operation order in nested while loop causes infinite loop 
'''

# For my use case, I have a function containing nested for-loops, and the bounds of each loop are dependent on function arguments.
# As-is, this function is not jit-compilable, since I cannot pass abstract arguments to Python's `range`. 
# However, I can jit-compile (and therefore use vmap/scan/other awesome JAX stuff) if I use `lax.while_loop`   
# with the aid of the experimental.loops module for simplicity (thanks, @gnecula !)

# The following simple Python loop, 
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

# Can be converted to a compilable JAX function as:

#def test(a,b):
#  with loops.Scope() as s:
#    s.quantity = 0.0
#    s.i = a + b
#    s.j = 2 * s.i + 1
#    s.k = 2 * s.j + 1
#
#    for _ in s.while_range(lambda: s.i > 0):
#      s.j = 2 * s.i + 1
#      for _ in s.while_range(lambda: s.j > 0):
#        s.k = 2 * s.j + 1 
#        for _ in s.while_range(lambda: s.k > 0):
#          s.quantity += s.i + s.j + s.k 
#          s.k -= 1
#        s.j -= 1
#      s.i -= 1
#    return s.quantity

#  Since it is now jit-compilable, I can vectorize this function `test` across many inputs and get really good performance relative to naive python control flow: 

# 1.28 seconds
#A = np.repeat(np.arange(10), 1000)
#B = np.repeat(np.arange(10), 1000)
#result1 = jax.vmap(test, (0,0))(A,B)

# 16.5 seconds
#A = onp.repeat(onp.arange(10), 1000)
#B = onp.repeat(onp.arange(10), 1000)
#result2 = []
#for i in range(A.shape[0]):
#    result.append(control(A[i], B[i]))


# This is great! However, a very strange bug occurs for the function using JAX while-loops: 
#  if the order of addition in the innermost loop changes such that `s.k` comes before either of the other two values `s.i` and `s.j`,
#  the loop never halts, and the function continuously uses more and more memory.
