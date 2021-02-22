import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)

def overlap_ss(A, B, alpha_bra, alpha_ket):
    ss = ((np.pi / (alpha_bra + alpha_ket))**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(A-B,A-B)) / (alpha_bra + alpha_ket)))
    return ss

def overlap_ps_block(A, B, alpha_bra, alpha_ket):
    oot_alpha_bra = 1 / (2 * alpha_bra)
    return oot_alpha_bra * jax.jacrev(overlap_ss,0)(A,B,alpha_bra,alpha_ket)

def overlap_sp_block(A, B, alpha_bra, alpha_ket): # not really needed is it?
    oot_alpha_bra = 1 / (2 * alpha_bra)
    return oot_alpha_bra * jax.jacrev(overlap_ss,1)(A,B,alpha_bra,alpha_ket)

def overlap_pp_block(A, B, alpha_bra, alpha_ket):
    oot_alpha_ket = 1 / (2 * alpha_ket)
    return oot_alpha_ket * (jax.jacfwd(overlap_ps_block, 1)(A,B,alpha_bra,alpha_ket))

def overlap_ds_block(A,B,alpha_bra,alpha_ket):
    oot_alpha_bra = 1 / (2 * alpha_bra)
    result = oot_alpha_bra * (jax.jacfwd(overlap_ps_block, 0)(A,B,alpha_bra,alpha_ket) + np.eye(3) * overlap_ss(A,B,alpha_bra,alpha_ket))
    iu = np.triu_indices(3)
    return result[iu]

def overlap_dp_block(A,B,alpha_bra,alpha_ket):
    oot_alpha_ket = 1 / (2 * alpha_ket) # use ket, since we are promoting ket from s-->p
    return np.ravel(oot_alpha_ket * jax.jacfwd(overlap_ds_block, 1)(A,B,alpha_bra,alpha_ket))

@jax.jit
def overlap_dd_block(A,B,alpha_bra,alpha_ket):
    oot_alpha_ket = 1 / (2 * alpha_ket) # use ket, since we are promoting ket from p-->d
    first_term = jax.jacfwd(overlap_dp_block, 1)(A,B,alpha_bra,alpha_ket)
    factor = np.tile(np.eye(3),(6,1))
    tmp_second_term = overlap_ds_block(A,B,alpha_bra,alpha_ket)
    second_term = factor * np.repeat(tmp_second_term, 9).reshape(18,3)
    result = oot_alpha_ket * (first_term + second_term)
    iu1,iu2 = np.triu_indices(3)
    result = result.reshape(6,3,3)[:,iu1,iu2].reshape(6,6)
    return result

A = np.array([0.0,0.0,-0.849220457955])
#B = np.array([0.0,0.0,-0.849220457955])
B = np.array([0.0,0.0, 0.849220457955])
a = 0.5
b = 0.5

args = (A,B,a,b)
#jaxpr = jax.make_jaxpr(overlap_dd_block)(*args)
#print('dd', len(str(jaxpr).splitlines()))
#
#jaxpr = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(overlap_dd_block)))))(*args)
#print('dd quartic', len(str(jaxpr).splitlines()))

#for i in range(100):
#    overlap_dd_block(*args)

for i in range(10):
    jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(overlap_dd_block))))(*args)


