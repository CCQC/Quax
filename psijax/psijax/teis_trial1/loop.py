




    with loops.Scope() as S:
    # NOTE TODO this is shell code needs to be modified
    # Computes a primitive ss overlap
    def primitive(A, B, C, D, aa, bb, cc, dd, coeff, am):
        '''Geometry parameters, exponents, coefficients, angular momentum identifier'''
        args = (Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2)
        sgra = (Cx, Cy, Cz, Ax, Ay, Az, e2, e1, c2, c1)
        primitive =  np.where(e1 ==  0, 0.0,
                     np.where(am ==  0, overlap_ss(*args), 0.0))
        return primitive
    # Computes multiple primitive ss overlaps with same center, angular momentum 
    vectorized_primitive = jax.vmap(primitive, (None,None,None,None,None,None,0,0,0,0,None))

    # Computes a contracted ss overlap 
    @jax.jit
    def contraction(Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2):
        primitives = vectorized_primitive(Ax, Ay, Az, Cx, Cy, Cz, e1, e2, c1, c2, 0)
        return np.sum(primitives)



        s.G = np.zeros(nbf,nbf,nbf,nbf)
        for i in s.range(nshells):
            for j in s.range(nshells):
                for k in s.range(nshells):
                    for l in s.range(nshells):


