import jax
import jax.numpy as np
from jax.config import config; config.update("jax_enable_x64", True)

def double_factorial(n):
    '''The double factorial function for small Python integer `n`.'''
    return np.prod(np.arange(n, 1, -2))

def normalize_old(aa):
    '''Normalization constant for s primitive basis functions. Argument is orbital exponent coefficient'''
    N = (2*aa/np.pi)**(3/4)
    return N

def normalize(aa,ax,ay,az):
    '''
    Normalization constant for gaussian basis function. 
    aa : orbital exponent
    ax : angular momentum component x
    ay : angular momentum component y
    az : angular momentum component z
    '''
#    f = np.sqrt(double_factorial(2*ax-1) * double_factorial(2*ay-1) * double_factorial(2*az-1))
#    N = (2*aa/np.pi)**(3/4) * (4 * aa)**((ax+ay+az)/2) / f

    f = double_factorial(2*ax-1) * double_factorial(2*ay-1) * double_factorial(2*az-1) * np.pi ** (3/2)
    N = ( (2**(2*(ax + ay + az) + 3/2) * aa**(ax + ay + az + (3/2)) ) / (f) ) ** 0.5
    return N

def overlap(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    Na = normalize_old(aa)
    Nb = normalize_old(bb)
    ss = Na * Nb * ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * np.dot(A-C, A-C)) / (aa + bb)))
    #ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * np.dot(A-C, A-C)) / (aa + bb)))
    return ss

# This works, compute derivative and divide by exponent... odd
px_sx = jax.grad(overlap, 2)
print(px_sx(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955, 0.5, 0.5))
print("Here")
print(px_sx(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955, 0.5, 0.4)/ np.sqrt(0.5))
print("Here")

px_px = jax.grad(jax.grad(overlap, 2), 5)
print(px_px(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955, 0.5, 0.4)/ (np.sqrt(0.5) *np.sqrt(0.4)))

print("DDDD")
dz_dz = jax.grad(jax.grad(jax.grad(jax.grad(overlap, 2), 2), 5), 5)
#print(dz_dz(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955, 0.5, 0.5) / 0.5**2)
print(dz_dz(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955, 0.5, 0.5))


# method 2: Do not normalize until the end.
# First compute (s|s), differentiate, and include factor

def overlap(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * np.dot(A-C, A-C)) / (aa + bb)))
    return ss


pz_s = jax.grad(overlap, 2)
val = pz_s(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955, 0.4, 0.4) / (2 * 0.4)
Npz = normalize(0.4,0,0,1)
Ns = normalize(0.4,0,0,0)
print("New")
print(val * Npz * Ns)

pz_pz = jax.grad(jax.grad(overlap, 2), 5)
val = pz_pz(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955, 0.5, 0.4) / ((2 * 0.5) * (2 * 0.4)) 
N1 = normalize(0.5,0,0,1)
N2 = normalize(0.4,0,0,1)
print("New")
print(val * N1 * N2)

dxy_dxy = jax.grad(jax.grad(jax.grad(jax.grad(overlap, 0), 1), 3), 4)
val = dxy_dxy(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955, 0.5, 0.5) / ((2 * 0.5) * (2 * 0.5))**2
N1 = normalize(0.5,1,1,0)
N2 = normalize(0.5,1,1,0)
print(val * N1 * N2)



#px_px = jax.grad(jax.grad(overlap, 0), 3)
#print(px_px(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955, 0.5, 0.5))



#px_px = jax.grad(jax.grad(overlap, 0), 3)
#print(px_px(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955, 0.4, 0.4)/0.4)
#print(px_px(0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955, 0.4, 0.4)/0.4)
#print(px_px(0.0,0.0, 0.849220457955,0.0,0.0, 0.849220457955, 0.4, 0.4)/0.4)
#
#print(px_px(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955, 0.5, 0.5)/0.5)
#print(px_px(0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955, 0.5, 0.5)/0.5)
#print(px_px(0.0,0.0, 0.849220457955,0.0,0.0, 0.849220457955, 0.5, 0.5)/0.5)
#
#pz_pz = jax.grad(jax.grad(overlap, 2), 5)
#print(pz_pz(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955, 0.4, 0.4)/0.4)
#print(pz_pz(0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955, 0.4, 0.4)/0.4)
#print(pz_pz(0.0,0.0, 0.849220457955,0.0,0.0, 0.849220457955, 0.4, 0.4)/0.4)
#
#print(pz_pz(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955, 0.5, 0.5)/0.5)
#print(pz_pz(0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955, 0.5, 0.5)/0.5)
#print(pz_pz(0.0,0.0, 0.849220457955,0.0,0.0, 0.849220457955, 0.5, 0.5)/0.5)


#def overlap2(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
#    A = np.array([Ax, Ay, Az])
#    C = np.array([Cx, Cy, Cz])
#    ss = ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * np.dot(A-C, A-C)) / (aa + bb)))
#    return ss
#
#print("METHOD 2")
#px_px = jax.grad(jax.grad(overlap2, 0), 3)
#print(px_px(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955, 0.4, 0.4))
#print(px_px(0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955, 0.4, 0.4))
#print(px_px(0.0,0.0, 0.849220457955,0.0,0.0, 0.849220457955, 0.4, 0.4))
#
#N11, N12 = normalize(0.4, 1,0,0), normalize(0.4, 1,0,0)
#N01, N02 = normalize(0.4, 0,0,0), normalize(0.4, 0,0,0)
#
#
#print("HERE")
#print(px_px(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955, 0.4, 0.4) * ((N11 * N12) - (N01 * N02)))
#print(px_px(0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955, 0.4, 0.4) * N1 * N2)
#print(px_px(0.0,0.0, 0.849220457955,0.0,0.0, 0.849220457955, 0.4, 0.4) * N1 * N2)
#
#t1 = px_px(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955, 0.4, 0.4) * N1 * N2
#t2 = px_px(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955, 0.4, 0.4) * N1 * N2
#
#N1, N2 = normalize(0.5, 1,0,0), normalize(0.5, 1,0,0)
#print(px_px(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955, 0.5, 0.5) * N1 * N2)
#print(px_px(0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955, 0.5, 0.5) * N1 * N2)
#print(px_px(0.0,0.0, 0.849220457955,0.0,0.0, 0.849220457955, 0.5, 0.5) * N1 * N2)





def overlap_pxpx(aa, cc, ax, ay, az, cx, cy, cz, Ax, Ay, Az, Cx, Cy, Cz):
    '''Computes a single overlap integral over two primitive s-orbital basis functions'''

    def fundamental(Ax, Ay, Az, Cx, Cy, Cz, aa, cc):
        '''Computes 'fundamental' (s|s) overlap integral (unnormalized)'''
        A = np.array([Ax, Ay, Az])
        C = np.array([Cx, Cy, Cz])
        Na = normalize(aa, ax,ay,az)
        Nc = normalize(cc, cx,cy,cz)
        ss = Na * Nc * ((np.pi / (aa + cc))**(3/2) * np.exp((-aa * cc * np.dot(A-C, A-C)) / (aa + cc)))
        #ss = ((np.pi / (aa + cc))**(3/2) * np.exp((-aa * cc * np.dot(A-C, A-C)) / (aa + cc)))
        return ss

    #Na = normalize(aa, ax,ay,az)
    #Nc = normalize(cc, cx,cy,cz)
    func = jax.grad(jax.grad(fundamental,0),3)
    return func(Ax, Ay, Az, Cx, Cy, Cz, aa, cc)

    #return Na * Nb * fundamental(Ax, Ay, Az, Cx, Cy, Cz, aa, bb)

def fundamental_overlap(Ax, Ay, Az, Cx, Cy, Cz, aa, bb):
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    Na = normalize(aa,0,0,0)
    Nb = normalize(bb,0,0,0)
    ss = Na * Nb * ((np.pi / (aa + bb))**(3/2) * np.exp((-aa * bb * np.dot(A-C, A-C)) / (aa + bb)))
    return ss


#Na1, Nc1 = normalize_old(0.4), normalize_old(0.4)
#Na2, Nc2 = normalize(0.4, 1,0,0), normalize(0.4, 1,0,0)
#
#N1 = Na1 * Nc1
#N2 = Na2 * Nc2
#print(N1)
#print(N2)
#print(N1 * N2)
#print(N1 / N2)
#print(N2 / N1)
#print("DONE")

#
#pxpx = jax.grad(jax.grad(fundamental_overlap, 0), 3)
#print(pxpx(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955,0.4, 0.4) / 0.4)
#print(pxpx(0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955,0.4, 0.4) / 0.4)
#
##s1s1 = fundamental_overlap(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955,0.4, 0.4)
##s1s2 = fundamental_overlap(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955 ,0.4, 0.4)
##s2s2 = fundamental_overlap(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955 ,0.4, 0.4)
##print(s1s1)
##print(s1s2)
##print(s2s2)
#
#px1px1 = overlap_pxpx(0.4, 0.4, 1, 0, 0, 1, 0, 0, 0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955)
#print(px1px1)
#px1px1 = overlap_pxpx(0.4, 0.4, 1, 0, 0, 1, 0, 0, 0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955)
#print(px1px1)
#
#
##s1s1 = fundamental_overlap(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955,0.4, 0.4)
##s1s2 = fundamental_overlap(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955,0.4, 0.4)
##s2s2 = fundamental_overlap(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955,0.4, 0.4)
##print(s1s1)
##print(s1s2)
##print(s2s2)
##
#s1s1 = overlap(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955,0.4, 0.4)
#s1s2 = overlap(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955,0.4, 0.4)
#s2s2 = overlap(0.0,0.0,-0.849220457955,0.0,0.0,0.849220457955,0.4, 0.4)
#print(s1s1)
#print(s1s2)
#print(s2s2)
#
#
##
#tmpgrad = jax.grad(overlap, argnums=0)
#gradpx = jax.grad(tmpgrad, argnums=3)
#print(gradpx(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955,0.4, 0.4))
#print(gradpx(0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955,0.4, 0.4))
#print(gradpx(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955,0.4, 0.4)/ 0.4)
#print(gradpx(0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955,0.4, 0.4) / 0.4)
#
#tmpgrad = jax.grad(overlap, argnums=2)
#gradpx = jax.grad(tmpgrad, argnums=5)
#print(gradpx(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955,0.4, 0.4))
#print(gradpx(0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955,0.4, 0.4))
#print(gradpx(0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955,0.4, 0.4)/ 0.4)
#print(gradpx(0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955,0.4, 0.4) / 0.4)
#
##
##pz1pz2 = gradpz(0.4, 0.4, 0.0,0.0,-0.849220457955,0.0,0.0,-0.849220457955)
##print(pz1pz2)
##pz1pz2 = gradpz(0.4, 0.4, 0.0,0.0,-0.849220457955,0.0,0.0, 0.849220457955)
#print(pz1pz2)

