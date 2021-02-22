Perhaps you could take the basis set dictionary and convert it to a JAX pytree object, and just
manually pull out the information needed as you go. This would be much cleaner function for evaluating TEIs.

"Many JAX functions, including all function transformations, operate over pytrees of arrays (other leaf types are sometimes allowed as well). 
Transformations are only applied to the leaf arrays while preserving the original pytree structure; for example, vmap and pmap only map over arrays, 
but automatically map over arrays inside of standard Python sequences, and can return mapped Python sequences."

It appears that just using basis dict directly and computing TEIs works 
jsut fine.. not sure what changed in JAX since i last tried it. 

Okay here is the problem, you can use the basis dict, but you can not use JAX functions
in place of for loops, since youre looping over basis dictionary information, which 
cannot be queried with Jaxpr tracer objects, so you get the error, unhashable type
when you try to index. 

Other random note: jit has a new donate argnums command which removes 
args from memory once they are done being used. cool. 



# Libint pseudo code 
```c
// loop over (contracted) Gaussian shells    **This is a loop over basis functions**
for(int s0=0; s0<nshell; ++s0) {
    for(int s1=0; s1<nshell; ++s1) {
        for(int s2=0; s2<nshell; ++s2) {
            for(int s3=0; s3<nshell; ++s3) {
                // decide whether to evaluate the integral (uniqueness, magnitude, etc.)
                // ..
                // loop over each primitive combination for this shell set
                int p0123 = 0;
                for(int p0=0; p0<nprim[s0]; ++p0) {                  * this is likely a typo s1 s2 s3
                    for(int p1=0; p1<nprim[s0]; ++p1) {
                        for(int p2=0; p2<nprim[s0]; ++p2) {
                            for(int p3=0; p3<nprim[s0]; ++p3) {
```

