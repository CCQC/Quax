
import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=500)
from integrals_utils import boys, gaussian_product


def overlap_ss(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns a (s|s) overlap integral
    """
    A = np.array([Ax, Ay, Az])
    C = np.array([Cx, Cy, Cz])
    alpha_sum = alpha_bra + alpha_ket
    return c1 * c2 * (np.pi / alpha_sum)**(3/2) * np.exp((-alpha_bra * alpha_ket * np.dot(A-C, A-C)) / alpha_sum)

def angular_momentum_factory(args, start_am, target_am, current, old=None, dim=6):
    ''' Produces integral functions of higher angular momentum from functions of lower angular momentum '''
    for idx in range(dim):
        if start_am[idx] != target_am[idx]:
            ai = start_am[idx]
            if ai == 0:
                if idx<=2:
                    def new(*args):
                        return (1/(2 * args[-4])) * (jax.grad(current,idx)(*args))
                else:
                    def new(*args):
                        return (1/(2 * args[-3])) * (jax.grad(current,idx)(*args))
            else:
                if idx<=2:
                    def new(*args):
                        return (1 / (2 * args[-4])) * (jax.grad(current,idx)(*args) + ai * old(*args))
                else:
                    def new(*args):
                        return (1 / (2 * args[-3])) * (jax.grad(current,idx)(*args) + ai * old(*args))
            promotion = onp.zeros(dim)
            promotion[idx] += 1
            return angular_momentum_factory(args, start_am + promotion, target_am, new, current, dim=dim)
        else:
            continue
    #return current
    #TODO cancel jit compiling if its bad
    return jax.jit(current)

args = (1.,1.,1.,1.,1.,1.,1.,1.,0.5,0.5)

jaxpr = jax.make_jaxpr(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([1,0,0,0,0,0]), overlap_ss, old=None, dim=6))(*args)
print('(px|s)', len(str(jaxpr).splitlines()))

jaxpr = jax.make_jaxpr(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([2,0,0,0,0,0]), overlap_ss, old=None, dim=6))(*args)
print('(dxx|s)', len(str(jaxpr).splitlines()))

jaxpr = jax.make_jaxpr(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([1,1,0,0,0,0]), overlap_ss, old=None, dim=6))(*args)
print('(dxy|s)', len(str(jaxpr).splitlines()))

jaxpr = jax.make_jaxpr(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([3,0,0,0,0,0]), overlap_ss, old=None, dim=6))(*args)
print('(fxxx|s)', len(str(jaxpr).splitlines()))

jaxpr = jax.make_jaxpr(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([1,1,1,0,0,0]), overlap_ss, old=None, dim=6))(*args)
print('(fxyz|s)', len(str(jaxpr).splitlines()))

jaxpr = jax.make_jaxpr(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([4,0,0,0,0,0]), overlap_ss, old=None, dim=6))(*args)
print('(gxxxx|s)', len(str(jaxpr).splitlines()))

jaxpr = jax.make_jaxpr(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([2,1,1,0,0,0]), overlap_ss, old=None, dim=6))(*args)
print('(gxxyz|s)', len(str(jaxpr).splitlines()))

jaxpr = jax.make_jaxpr(angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([2,0,0,1,0,0]), overlap_ss, old=None, dim=6))(*args)
print('(dxx|px)', len(str(jaxpr).splitlines()))



px_s = angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([1,0,0,0,0,0]), overlap_ss, old=None, dim=6)
py_s = angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,1,0,0,0,0]), overlap_ss, old=None, dim=6)
pz_s = angular_momentum_factory(args, onp.array([0,0,0,0,0,0]), onp.array([0,0,1,0,0,0]), overlap_ss, old=None, dim=6)

#@jax.jit
def overlap_ps(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2):
    i = px_s(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    j = py_s(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    k = pz_s(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    return np.array([i,j,k])

print(overlap_ps(*args))
jaxpr = jax.make_jaxpr(overlap_ps)(*args)
print('(p|s)', len(str(jaxpr).splitlines()))



