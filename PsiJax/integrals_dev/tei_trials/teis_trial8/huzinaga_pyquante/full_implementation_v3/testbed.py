# We know where in the integral array
# a shell quartet dimesion starts 
# and how many places it fills (3 for p, 6 for d...)

# We want to loop over different possible angluar momentum
# distributions

import jax.numpy as np

angular momentum = np.array([0,0,0], 
                            [1,0,0],                             
                            [0,1,0],
                            [0,0,1],
                            [2,0,0],
                            [1,1,0],
                            [1,0,1],
                            [0,2,0],
                            [0,1,1],
                            [0,0,2]])

# Given an angular mometum, access the starting point in above array for angular momentum distribution
am = 0 
leading = 0
for i in range(am):
    leading += (i+1)*(i+2) // 2

print(leading)




