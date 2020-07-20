import numpy as np
from scipy import special

def boys_general(n, x):
    denom = 2 * n + 1
    num = special.hyp1f1(n+0.5,n+1.5,-x)
    return num / denom

print(boys_general(0, 30),np.sqrt(np.pi) / (2 * np.sqrt(30)))
print(boys_general(0, 40),np.sqrt(np.pi) / (2 * np.sqrt(40)))
print(boys_general(0, 50),np.sqrt(np.pi) / (2 * np.sqrt(50)))
print(boys_general(0, 60),np.sqrt(np.pi) / (2 * np.sqrt(60)))
print(boys_general(0, 70),np.sqrt(np.pi) / (2 * np.sqrt(70)))
print(boys_general(0, 80),np.sqrt(np.pi) / (2 * np.sqrt(80)))
print(boys_general(0, 90),np.sqrt(np.pi) / (2 * np.sqrt(90)))
print(boys_general(0, 20000),np.sqrt(np.pi) / (2 * np.sqrt(1000)))

#grid = np.arange(0, 30, 1e-5)
#
#F0 = boys_general(0, grid)
#F1 = boys_general(1, grid)
#F2 = boys_general(2, grid)
#F3 = boys_general(3, grid)
#F4 = boys_general(4, grid)
#F5 = boys_general(5, grid)
#F6 = boys_general(6, grid)
#F7 = boys_general(7, grid)
#F8 = boys_general(8, grid)
#F9 = boys_general(9, grid)
#F10 = boys_general(10, grid)
#
#F = np.vstack([F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10])
#
## Fuse factorial pre-factors into the boys function arguments
#F = np.einsum('ij,i->ij', F, np.array([1,-1,0.5,-1/6,1/24,-1/120,1/720,1/5040,1/40320,1/362880,1/3628800]))
#
#np.save('boys/boys_F0_F10_grid_0_30_1e5', F)



