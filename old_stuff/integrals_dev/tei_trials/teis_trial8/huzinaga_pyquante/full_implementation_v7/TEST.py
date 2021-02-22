from tei import binom 
from reference import binomial 
import numpy as onp

from tei import factorial


print(factorial(11))
print(factorial(12))
print(factorial(13))
print(factorial(14))
print(factorial(15))
print(factorial(16))
print(factorial(20))
print(factorial(21))
print(factorial(22))
print(factorial(23))
print(factorial(24))
print(factorial(25))
print(factorial(26))

from math import factorial

print(factorial(11))
print(factorial(12))
print(factorial(13))
print(factorial(14))
print(factorial(15))
print(factorial(16))
print(factorial(20))
print(factorial(21))
print(factorial(22))
print(factorial(23))
print(factorial(24))
print(factorial(25))
print(factorial(26))




# Test binomial coefficients
i1 = [8,3,4,26,9,15]
i2 = [1,2,3,4,5,6]
for i in range(6):
    one = binom(i1[i],i2[i])
    two = binomial(i1[i],i2[i])
    print(one,two)
    print(onp.allclose(one,two))
    
