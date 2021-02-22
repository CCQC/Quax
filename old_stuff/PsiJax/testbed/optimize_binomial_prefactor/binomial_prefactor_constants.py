import numpy as onp
import jax.numpy as np

binomials = onp.array([[1, 1,  0,  0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0], 
                      [1, 1,  0,  0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 2,  1,  0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 3,  3,  1,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 4,  6,  4,   1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 5, 10, 10,   5,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 6, 15, 20,  15,    6,    1,    0,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 7, 21, 35,  35,   21,    7,    1,    0,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 8, 28, 56,  70,   56,   28,    8,    1,    0,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1, 9, 36, 84, 126,  126,   84,   36,    9,    1,    0,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1,10, 45,120, 210,  252,  210,  120,   45,   10,    1,    0,    0,    0,    0,   0,  0,  0, 0,0],
                      [1,11, 55,165, 330,  462,  462,  330,  165,   55,   11,    1,    0,    0,    0,   0,  0,  0, 0,0],
                      [1,12, 66,220, 495,  792,  924,  792,  495,  220,   66,   12,    1,    0,    0,   0,  0,  0, 0,0],
                      [1,13, 78,286, 715, 1287, 1716, 1716, 1287,  715,  286,   78,   13,    1,    0,   0,  0,  0, 0,0],
                      [1,14, 91,364,1001, 2002, 3003, 3432, 3003, 2002, 1001,  364,   91,   14,    1,   0,  0,  0, 0,0],
                      [1,15,105,455,1365, 3003, 5005, 6435, 6435, 5005, 3003, 1365,  455,  105,   15,   1,  0,  0, 0,0],
                      [1,16,120,560,1820, 4368, 8008,11440,12870,11440, 8008, 4368, 1820,  560,  120,  16,  1,  0, 0,0],
                      [1,17,136,680,2380, 6188,12376,19448,24310,24310,19448,12376, 6188, 2380,  680, 136, 17,  1, 0,0],
                      [1,18,153,816,3060, 8568,18564,31824,43758,48620,43758,31824,18564, 8568, 3060, 816,153, 18, 1,0],
                      [1,19,171,969,3876,11628,27132,50388,75582,92378,92378,75582,50388,27132,11628,3876,969,171,19,1]], dtype=int)


def valeev(k, l1, l2, PAx, PBx):
    # THIS WORKS! And no boolean checks!
    total = 0.
    q = max(-k, k-2*l2)
    q_final = min(k, 2*l1 - k)
    while q <= q_final:
        i = (k+q)//2
        j = (k-q)//2
        total += PAx[l1-i] * binomials[l1,i] * PBx[l2-j] * binomials[l2,j]
        q += 2
    return total

def new_binomial_prefactor(k, l1, l2, PAx, PBx):
    total = 0.
    length = binomial_prefactor_iterlength[k,l1,l2]
    q = 0  
    while q < length:
        i = binomial_prefactor_i[k,l1,l2,q]
        j = binomial_prefactor_j[k,l1,l2,q]
        total += PAx[l1-i] * binomials[l1,i] * PBx[l2-j] * binomials[l2,j]
        q += 1
    return total

PAx = onp.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
PBx = onp.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7])

# Binomial prefactor indices cache.
# For binomial prefactor 
# f_k = sum over q={max(-k,k-l2),min(k,2*l1-k), increment 2} [binom(l1,i) * binom(l2,j) * PAx^(l1-i) * PBx^(l2-j)]
# where i = (k + q) // 2 and j = (k - q) // 2
# We pre-save the number of iterations the sum runs over, and the valid values of i and j
# Assuming a maximum angular momentum of 5 (h functions) in the basis set.
# We index the arrays by index k, l1, and l2, which completely determine q, and therefore i and j. 
# Then, when the function binomial_prefactor is called, 
# it only has to pull out the appropriate indices rather than compute it on-the-fly.
# Since binomial prefactor is called 2-20x more than the number of primitive integrals, this reduction is massive.
def binomial_index_builder(k, l1, l2):
    """
    Function for collecting all possible values of i,j, and total binomial prefactor function iteration length
    given k (subscript in f_k in eqn 2.46 in Fermann and Valeev) and angular momenta l1 and l2, where k can be any value between 0 and l1+l2
    """
    q = max(-k, k-2*l2)
    q_final = min(k, 2*l1 - k)
    LENGTH = 0
    i_list = []
    j_list = []
    while q <= q_final:
        i = (k+q)//2
        j = (k-q)//2

        i_list.append(i)
        j_list.append(j)
        LENGTH += 1
        q += 2
    return LENGTH, i_list, j_list

max_am = 5
binomial_prefactor_iterlength = onp.zeros((2*max_am+1,max_am+1,max_am+1,1), dtype=int)
binomial_prefactor_i = onp.zeros((2*max_am+1, max_am+1, max_am+1, 2*max_am+1), dtype=int)
binomial_prefactor_j = onp.zeros((2*max_am+1, max_am+1, max_am+1, 2*max_am+1), dtype=int)

for l1 in range(max_am+1):
    for l2 in range(max_am+1):
        for k in range(l1 + l2 + 1):
            q, i_list, j_list = binomial_index_builder(k,l1,l2)
            binomial_prefactor_iterlength[k,l1,l2,0] = q 
            for i in range(len(i_list)):
                binomial_prefactor_i[k,l1,l2,i] = i_list[i]
            for j in range(len(j_list)):
                binomial_prefactor_j[k,l1,l2,j] = j_list[j]


#for l1 in range(max_am+1):
#    for l2 in range(max_am+1):
#        for k in range(l1 + l2 + 1):
#            a = valeev(k,l1,l2, PAx, PBx)
#            b = new_binomial_prefactor(k,l1,l2, PAx, PBx)
##            print(np.allclose(a,b))



binomial_prefactor_iterlength = np.array(binomial_prefactor_iterlength)
binomial_prefactor_i = np.array(binomial_prefactor_i)
binomial_prefactor_j = np.array(binomial_prefactor_j)



#for i in range(100000):
#  new_binomial_prefactor(3,3,3,PAx,PBx)

#print(test(2,3,3,PAx,PBx))
#print(valeev2(2,3,3,PAx,PBx))


#print("---------")
#test(0,3,0)
#print("---------")
#test(1,3,0)
#print("---------")
#test(2,3,0)
#print("---------")
#test(3,3,0)
#print("---------")
#
#print("---------")
#test(0,3,1)
#print("---------")
#test(1,3,1)
#print("---------")
#test(2,3,1)
#print("---------")
#test(3,3,1)
#print("---------")
#test(4,3,1)
#print("---------")
#
#print("---------")
#test(0,3,2)
#print("---------")
#test(1,3,2)
#print("---------")
#test(2,3,2)
#print("---------")
#test(3,3,2)
#print("---------")
#test(4,3,2)
#print("---------")
#test(5,3,2)
#print("---------")
#
#print("---------")
#test(0,3,3)
#print("---------")
#test(1,3,3)
#print("---------")
#test(2,3,3)
#print("---------")
#test(3,3,3)
#print("---------")
#test(4,3,3)
#print("---------")
#test(5,3,3)
#print("---------")
#test(6,3,3)

