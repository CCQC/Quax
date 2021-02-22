
# I am writing a new primitive for a function in which I know how to compute exact derivatives
# to arbitrary order.  
# The new primitive takes in an vector as input, references an external library for 
# some expensive computations, and returns 4-dimensional tensor. 

# I want to be able to support nested differentiation of this function to several orders.
# I run into some trouble, however, since this new primitive requires a few extra arguments 
# which are needed for the semantics of calling the computation in the external library. 

# The details are complicated, so I did my best to prepare a simple, analogous example
# here.

import jax
import numpy as onp
import jax.numpy as np

# Just do it without any funny business so you can figure out shapes, etc

# Create primitive
new_primitive_p = jax.core.Primitive("new_primitive")

# Create function to call the primitive
def new_primitive(vec):
    return new_primitive_p.bind(vec)

# Create evaluation rule
# G = ones * (exp^2x + exp^2y + exp^2z)
# First derivative should be just 2x, second deriv 4x, third deriv 8x, etc
def new_primitive_impl(vec):
    G = onp.ones((2,2,2,2))
    G = G * onp.sum(onp.exp(2*vec))
    return G

new_primitive_p.def_impl(new_primitive_impl)

def new_primitive_jvp(primals, tangents):
    print("JVP called")
    vec, = primals
    primals_out = new_primitive(vec) 

    #print(tangents)
    derivatives = []
    for v_dot in tangents:
        print('vdot',v_dot)
        # derivative is the same as the eval x 2
        deriv = 2 * new_primitive(vec)
        derivatives.append(deriv)

    tangents_out = np.concatenate(derivatives, axis=-1)
    #tangents_out = derivatives
    return primals_out, tangents_out 

jax.ad.primitive_jvps[new_primitive_p] = new_primitive_jvp 

p, t = jax.jvp(new_primitive, (np.arange(3.),), (np.array([1.,0.,0.]),))
print(p.shape)
#print(t)

# Define Batching rule
#def new_primitive_batch(vector_arg_values, batch_axes): 

# YOLO try this
jax.interpreters.batching.defvectorized(new_primitive_p)


blah = jax.jacfwd(new_primitive)(np.arange(3.))
print(blah.shape)

blah = jax.jacfwd(jax.jacfwd(new_primitive))(np.arange(3.))
print(blah.shape)

#jax.interpreters.batching.primitive_batchers[new_primitive_p] = new_primitive_batch

