import jax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)

_float = {onp.floating, jax.dtypes.bfloat16}
def boys(x):
    return boys_p.bind(x)
boys_p = jax.core.Primitive('boys')

# evalutation rule of primitive
def boys_eval(x):
    x = x + 1e-12
    return 0.88622692545275798 * jax.lax.rsqrt(x) * jax.lax.erf(jax.lax.sqrt(x))

#def boys_jvp_rule(g, x):
#    tmp = boys(x)
#    result = jax.lax.select(x < 1e-8, (-0.3333333333333333333) + + (2 * x * 0.1) + -(3 * x**2 * 0.023809523809523808) + (4 * x**3 * 0.004629629629629629),
#                                        jax.lax.div(-jax.lax.sub(tmp, jax.lax.exp(-x)),  jax.lax.mul(jax.lax._const(x,2), x)))
#    return result 

#def boys_jvp_rule(g, x):
#    tmp = boys(x)
#    result = jax.lax.select(x < 1e-8, (-0.3333333333333333333) + (2 * x * 0.1) + -(3 * x**2 * 0.023809523809523808) + (4 * x**3 * 0.004629629629629629),
#                            jax.lax.div(-jax.lax.sub(tmp, jax.lax.exp(-x)),  jax.lax.mul(jax.lax._const(x,2), x)))
#    return result 

def boys_jvp_rule(g, ans, x):
    result = jax.lax.select(x < 1e-8, (-0.3333333333333333333) + (2 * x * 0.1) + -(3 * x**2 * 0.023809523809523808) + (4 * x**3 * 0.004629629629629629),
                            jax.lax.div(-jax.lax.sub(ans, jax.lax.exp(-x)),  jax.lax.mul(jax.lax._const(x,2), x)))
    return result 

def f_vjp(x):
  return boys(x), lambda g: (2 * g * x,)

jax.lax.lax.standard_unop(_float, 'boys') 
boys_p.def_impl(boys_eval)

#jax.interpreters.ad.defjvp(boys_p, boys_jvp_rule)
# okay, defjvp2 assumes 3 arguments: tangent, result of original function, function argument
jax.interpreters.ad.defjvp2(boys_p, boys_jvp_rule)

jax.interpreters.ad.defvjp(boys_p, f_vjp)

print(boys(0.5))
print(jax.jacfwd(boys)(0.5))

print(jax.jacrev(boys)(0.5))


