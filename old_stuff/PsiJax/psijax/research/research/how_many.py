import math

def how_many_derivs(k, n):
    """k is number centers, n is deriv order, no potential integrals"""
    val = 1
    for i in range(n):
        val *= (3 * k + i)
    return (1 / math.factorial(n)) * val

#print(how_many_derivs(4, 1))
#print(how_many_derivs(4, 2))
#print(how_many_derivs(4, 3))
#print(how_many_derivs(4, 4))


print(how_many_derivs(4, 1))
print(how_many_derivs(4, 2))
print(how_many_derivs(4, 3))
print(how_many_derivs(4, 4))

# Number of unique nuclear derivatives 
print(how_many_derivs(6, 4))


