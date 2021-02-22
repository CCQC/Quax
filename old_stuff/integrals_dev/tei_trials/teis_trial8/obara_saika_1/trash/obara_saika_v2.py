# Figure out how to compute a single primitive with OS, not really worrying about JAX
# inspired by https://github.com/ChiCheng45/Gaussium/blob/master/src/integrals/twoelectronrepulsion/obara_saika_scheme.py
import jax
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
import numpy as onp
from integrals_utils import gaussian_product, delta, create_primitive, boys, ssss_0

def primitive_eri(g1, g2, g3, g4):
    a_1 = g1[2]
    a_2 = g2[2]
    a_3 = g3[2]
    a_4 = g4[2]
    a_5 = a_1 + a_2
    a_6 = a_3 + a_4
    a_7 = (a_5 * a_6) / (a_5 + a_6)

    r_1, r_2, r_3, r_4 = g1[1], g2[1], g3[1], g4[1]

    r_5 = gaussian_product(a_1, r_1, a_2, r_2)
    r_6 = gaussian_product(a_3, r_3, a_4, r_4)
    r_7 = gaussian_product(a_5, r_5, a_6, r_6)

    r_12 = np.dot(r_1-r_2,r_1-r_2)
    r_34 = np.dot(r_3-r_4,r_3-r_4)
    r_56 = np.dot(r_5-r_6,r_5-r_6)

    boys_x = (a_5 * a_6 * r_56**2) / (a_5 + a_6)
    ssss1 = (2 * np.pi**(5/2)) / (a_5 * a_6 * np.sqrt(a_5 + a_6))
    ssss2 = np.exp(((- a_1 * a_2 * r_12**2) / a_5) - ((a_3 * a_4 * r_34**2) / a_6))

    l_total = np.sum(g1[0]) + np.sum(g2[0]) + np.sum(g3[0]) + np.sum(g4[0])
    nu = np.arange(l_total + 1)
    boys_evals = boys(nu, np.full_like(nu,boys_x))
    boys_evals = boys_evals * ssss1 * ssss2 
    #boys_out3 = boys_function(l_total, boys_x)
    #self.end_dict = {l_total: boys_out1 * boys_out2 * boys_out3}
    return os_begin(0,g1,g2,g3,g4, boys_evals)

def os_begin(m, g1, g2, g3, g4, boys_evals):
    '''Starts recursion
    '''
    l_1 = g1[0]
    l_2 = g2[0]
    l_3 = g3[0]
    l_4 = g4[0]

    if l_1[0] > 0:
        return os_recursion(0, m, *os_gaussian_factory(0, g1, g2, g3, g4), boys_evals)
    elif l_1[1] > 0:
        return os_recursion(1, m, *os_gaussian_factory(1, g1, g2, g3, g4), boys_evals)
    elif l_1[2] > 0:
        return os_recursion(2, m, *os_gaussian_factory(2, g1, g2, g3, g4), boys_evals)
    elif l_2[0] > 0:
        return os_recursion(0, m, *os_gaussian_factory(0, g2, g1, g4, g3), boys_evals)
    elif l_2[1] > 0:
        return os_recursion(1, m, *os_gaussian_factory(1, g2, g1, g4, g3), boys_evals)
    elif l_2[2] > 0:
        return os_recursion(2, m, *os_gaussian_factory(2, g2, g1, g4, g3), boys_evals)
    elif l_3[0] > 0:
        return os_recursion(0, m, *os_gaussian_factory(0, g3, g4, g1, g2), boys_evals)
    elif l_3[1] > 0:
        return os_recursion(1, m, *os_gaussian_factory(1, g3, g4, g1, g2), boys_evals)
    elif l_3[2] > 0:
        return os_recursion(2, m, *os_gaussian_factory(2, g3, g4, g1, g2), boys_evals)
    elif l_4[0] > 0:
        return os_recursion(0, m, *os_gaussian_factory(0, g4, g3, g2, g1), boys_evals)
    elif l_4[1] > 0:
        return os_recursion(1, m, *os_gaussian_factory(1, g4, g3, g2, g1), boys_evals)
    elif l_4[2] > 0:
        return os_recursion(2, m, *os_gaussian_factory(2, g4, g3, g2, g1), boys_evals)
    else:
        return boys_evals[m]

def os_recursion(i, m, g1, g2, g3, g4, g5, g6, g7, g8, boys_evals):
    '''
    Parameters
    ----------
    i: int
        Which component of angular momentum (0,1,2) we are promoting. As to which center, this will be controlled by
        which primitive Guassians are being passed in.
    m: int
        Current auxilliary index
    g1,g2,g3,g4,g5,g6,g7,g8:
        The eight distinct Gaussians which appear in the OS equation, modulated downward to read as
        [ab|cd] = f([(a-1i)b|cd], [(a-2i)b|cd], [(a-1i)(b-1i)|cd], [(a-1i)b|(c-1i)d], [(a-1i)b|c(d-1i)])
        There are 8 distinct primitive Gaussians in this eqaution, 
        (a-1i), b, c, d, (a-2i), (b-1i), (c-1i), (d-1i)
    '''
    out1, out2, out3, out4, out5, out6, out7, out8 = 0,0,0,0,0,0,0,0

    a_1 = g1[2]
    a_2 = g2[2]
    a_3 = g3[2]
    a_4 = g4[2]
    a_5 = a_1 + a_2  # zeta
    a_6 = a_3 + a_4  # eta 
    a_7 = (a_5 * a_6) / (a_5 + a_6) # self variable before recursion  
    r_1 = g1[1]
    r_2 = g2[1]
    r_5 = gaussian_product(a_1, r_1, a_2, r_2) # P: done in recursion function
    r_6 = gaussian_product(a_1, r_1, a_2, r_2) # Q: stored inside self.r_7 (W) before recursion
    r_7 = gaussian_product(a_5, r_5, a_6, r_6) # W: stored as self before recursion

    if r_5[i] != r_1[i]:
        out1 = (r_5[i] - r_1[i]) * os_begin(m, g1, g2, g3, g4, boys_evals)
    if r_7[i] != r_5[i]:
        out2 = (r_7[i] - r_5[i]) * os_begin(m + 1, g1, g2, g3, g4, boys_evals)
    if g5[0][i] >= 0:
        out3 = delta(g1[0][i]) * (1 / (2 * a_5)) * os_begin(m, g5, g2, g3, g4, boys_evals)
        out4 = delta(g1[0][i]) * (a_7 / (2 * a_5 ** 2)) * os_begin(m+1, g5, g2, g3, g4, boys_evals)
    if g6[0][i] >= 0:
        out5 = delta(g2[0][i]) * (1 / (2 * a_5)) * os_begin(m, g1, g6, g3, g4, boys_evals)
        out6 = delta(g2[0][i]) * (a_7 / (2 * a_5 ** 2)) * os_begin(m+1, g1, g6, g3, g4, boys_evals)
    if g7[0][i] >= 0:
        out7 = delta(g3[0][i]) * (1 / (2 * (a_5 + a_6))) * os_begin(m+1, g1, g2, g7, g4, boys_evals)
    if g8[0][i] >= 0:
        out8 = delta(g4[0][i]) * (1 / (2 * (a_5 + a_6))) * os_begin(m+1, g1, g2, g3, g8, boys_evals)

    return out1 + out2 + out3 - out4 + out5 - out6 + out7 + out8


def os_gaussian_factory(i, g1, g2, g3, g4):
    a_1 = g1[2]
    a_2 = g2[2]
    a_3 = g3[2]
    a_4 = g4[2]

    r_1 = g1[1]
    r_2 = g2[1]
    r_3 = g3[1]
    r_4 = g4[1]
        
    l_1 = g1[0]
    l_2 = g2[0]
    l_3 = g3[0]
    l_4 = g4[0]

    if i == 0:
        g1x1 = create_primitive(l_1[0] - 1, l_1[1], l_1[2], r_1, a_1)
        g1x2 = create_primitive(l_1[0] - 2, l_1[1], l_1[2], r_1, a_1)
        g2x1 = create_primitive(l_2[0] - 1, l_2[1], l_2[2], r_2, a_2)
        g3x1 = create_primitive(l_3[0] - 1, l_3[1], l_3[2], r_3, a_3)
        g4x1 = create_primitive(l_4[0] - 1, l_4[1], l_4[2], r_4, a_4)
        return g1x1, g2, g3, g4, g1x2, g2x1, g3x1, g4x1
    elif i == 1:
        g1y1 = create_primitive(l_1[0], l_1[1] - 1, l_1[2], r_1, a_1)
        g1y2 = create_primitive(l_1[0], l_1[1] - 2, l_1[2], r_1, a_1)
        g2y1 = create_primitive(l_2[0], l_2[1] - 1, l_2[2], r_2, a_2)
        g3y1 = create_primitive(l_3[0], l_3[1] - 1, l_3[2], r_3, a_3)
        g4y1 = create_primitive(l_4[0], l_4[1] - 1, l_4[2], r_4, a_4)
        return g1y1, g2, g3, g4, g1y2, g2y1, g3y1, g4y1
    elif i == 2:
        g1z1 = create_primitive(l_1[0], l_1[1], l_1[2] - 1, r_1, a_1)
        g1z2 = create_primitive(l_1[0], l_1[1], l_1[2] - 2, r_1, a_1)
        g2z1 = create_primitive(l_2[0], l_2[1], l_2[2] - 1, r_2, a_2)
        g3z1 = create_primitive(l_3[0], l_3[1], l_3[2] - 1, r_3, a_3)
        g4z1 = create_primitive(l_4[0], l_4[1], l_4[2] - 1, r_4, a_4)
        return g1z1, g2, g3, g4, g1z2, g2z1, g3z1, g4z1

    
#g1 = create_primitive(1,0,0, np.array([0.0,0.0, 0.9]), 0.5) 
#g2 = create_primitive(0,0,0, np.array([0.0,0.0,-0.9]), 0.5) 
#g3 = create_primitive(0,0,0, np.array([0.0,0.0, 0.9]), 0.5) 
#g4 = create_primitive(0,0,0, np.array([0.0,0.0,-0.9]), 0.5) 
#
#result = primitive_eri(g1, g2, g3, g4)
#print(result)

