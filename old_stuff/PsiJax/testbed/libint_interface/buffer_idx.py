import numpy as np
from itertools import combinations_with_replacement as cwr

class Shell(object):
    """ Dummy shell object. not complete"""
    def __init__(self, am, atom_idx):
        self.am = am 
        self.atom_idx = atom_idx  

# Create 4 p functions which are on atoms 3, 0, 2 and 1
s1 = Shell(1, 0)
s2 = Shell(1, 0)
s3 = Shell(1, 1)
s4 = Shell(1, 1)
ShellObjectList = [s1, s2, s3, s4]

def convert_buffer_idx(d_i, n, shells, k=4):
    """
    d_i : first index of the LibInt ERI Shell quartet derivative buffer
    n : derivative order
    shells : a list of Shell objects with atom_idx attributes
    k : number of centers in integral 
    """
    triu_indices = np.asarray(list(cwr(np.arange(3 * k), n)))
    # Can skip this and just save these two arrays for case of second derivatives of ERI's, k=4, n=2.
    shell_center_indices, cart_component_indices = np.divmod(triu_indices,3)

    cart_idx = cart_component_indices[d_i]
    shell_deriv_idx = shell_center_indices[d_i]
    
    atom_idx_1 = shells[shell_deriv_idx[0]].atom_idx
    atom_idx_2 = shells[shell_deriv_idx[1]].atom_idx

    cart_idx_1 = cart_idx[0]
    cart_idx_2 = cart_idx[1]
    print(cart_idx_1)
    print(cart_idx_2)

    g1 = 3 * atom_idx_1 + cart_idx_1
    g2 = 3 * atom_idx_2 + cart_idx_2
    return g1, g2

#To get the last two indices in (n,n,n,n,g1,g2) for buffer[27][:], run
g1, g2 = convert_buffer_idx(27, 2, ShellObjectList) 
print(g1, g2)


# Psuedocode for using the above function
# result = empty_vec_of_size_n^4g^2
# for shell M
#   for shell N
#     for shell P
#       for shell Q
#         
#          buffer = compute_eri_deriv2(M, N, P, Q)
#          Shells = [M, N, P, Q] 
#
#          # Loop over first dimension of buffer 
#          for i in range(len(buffer)):
#            g1, g2 = convert_buffer_idx(i, 2, Shells)   
#             
#            # Loop over angular momentum distributions
#            for s1 in range(shellsize1):
#              for s2 in range(shellsize2):
#                for s3 in range(shellsize3):
#                  for s4 in range(shellsize4):
#                    # Get first 4 indices n1, n2, n3, n4 
#                    n1 = start_idx1 + s1
#                    n2 = start_idx2 + s2
#                    n3 = start_idx3 + s3
#                    n4 = start_idx4 + s4
#                    #Find second dimension of buffer index 
#                    buffer_shell_idx = flatten_idx(n1,n2,n3,n4)
#                    if g1 == g2:
#                      idx1 = flatten_idx(n1,n2,n3,n4,g1,g2)
#                      result[idx1] = buffer[i, buffer_shell_index]
#                    elif g1 != g2:
#                      idx1 = flatten_idx(n1,n2,n3,n4,g1,g2)
#                      idx2 = flatten_idx(n1,n2,n3,n4,g2,g1)
#                      result[idx1] = buffer[i, buffer_shell_index]
#                      result[idx2] = buffer[i, buffer_shell_index]


