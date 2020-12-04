
import math

n = 4 # order of diff
k = 2 # number of centers
l = 6 # number of atoms

val = 1
for i in range(n): 
    val *= (3 * (k + l) + i)
    
val /= math.factorial(n)
print(int(val))


