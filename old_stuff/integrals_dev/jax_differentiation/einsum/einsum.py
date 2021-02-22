import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)

def intermediate(A,B,aa,bb):
    zeta = aa + bb
    P = ((aa * A + bb * B) / zeta)

def overlap_ss(A, B, aa, bb):
    ss = ((np.pi / (aa + bb ))**(3/2) * np.exp((-aa * bb * np.dot(A-B,A-B)) / (aa + bb)))
    return ss

def overlap_ps(A, B, aa, bb):
    zeta = aa + bb
    P = ((aa * A + bb * B) / zeta)
    PA = P - A
    return PA * overlap_ss(A,B,aa,bb)

def overlap_pp(A,B,aa,bb):
    zeta = aa + bb
    P = ((aa * A + bb * B) / zeta)
    PB = P - B
    identity = np.eye(3)
    ps = overlap_ps(A,B,aa,bb)
    return np.einsum('j,i->ij', PB, ps) + 1/(2*zeta) * identity * overlap_ss(A,B,aa,bb)


def overlap_ds(A,B,aa,bb):
    zeta = aa + bb
    P = ((aa * A + bb * B) / zeta)
    PA = P - A
    ps = overlap_ps(A,B,aa,bb)
    identity = np.eye(3)
    return np.einsum('j,i->ij', PA, ps) + 1/(2*zeta) * identity * overlap_ss(A,B,aa,bb)

def overlap_dp(A,B,aa,bb):
    zeta = aa + bb
    P = ((aa * A + bb * B) / zeta)
    PB = P - B
    ds = overlap_ds(A,B,aa,bb)
    ps = overlap_ps(A,B,aa,bb)
    identity = np.eye(3)
    ootzeta = 1 / (2 * zeta)
    return  np.einsum('k,ij->ijk',PB,ds) + ootzeta * np.einsum('ik,j->ijk',identity, ps) + ootzeta * np.einsum('jk,i->ijk', identity, ps)

@jax.jit
def overlap_dd(A,B,aa,bb):
    zeta = aa + bb
    P = ((aa * A + bb * B) / zeta)
    PB = P - B
    dp = overlap_dp(A,B,aa,bb)
    pp = overlap_pp(A,B,aa,bb)
    ds = overlap_ds(A,B,aa,bb)
    identity = np.eye(3)
    ootzeta = 1 / (2 * zeta)
    return np.einsum('l,ijk->ijkl',PB,dp) + ootzeta * np.einsum('il,jk->ijkl',identity,pp) + ootzeta * np.einsum('jl,ik->ijkl',identity,pp) + ootzeta * np.einsum('kl,ij->ijkl',identity,ds)

A = np.array([0.0,0.0,-0.849220457955])
#B = np.array([0.0,0.0,-0.849220457955])
B = np.array([0.0,0.0, 0.849220457955])
a = 0.5
b = 0.5
#print(overlap_pp(A,B,a,b))
args = (A,B,a,b)
#jaxpr = jax.make_jaxpr(overlap_dd)(*args)
#print('dd', len(str(jaxpr).splitlines()))
#
#jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(overlap_dd)))))(*args)
#print('dd quartic', len(str(jaxpr).splitlines()))

for i in range(10):
    jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(overlap_dd))))(*args)

