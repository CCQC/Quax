
* PyQuante VRR for (a 0 | c 0) integrals
    * This works, but is very slow. 10,000 primitives integrals, vmapped, 
      with total angular momentum of 7 takes 1m33s, 4.4 GB. 
       (obara_saika_pyquante/os_v4.py)
    * Note this isnt even the full integrals, and its still really slow with massive memory overhead. 
    * It could be that explicit contraction loops will perform better than vmap, like observed in hunzinaga.
    * NEw jit arg `donate_argnums` could save on memory if used correctly 
    * An interesting tidbit here is if you instead of using jax index update, just dump everything
      into a dummy variable 

* PyQuante Taketa/Hunzinaga/Oohata compiles quickly and runs quickly until you implement contractions
    A straightfoward vmap of 1,000,000 primitives with total angular momentum of 7
    is merely 29s, 2.3 GB.

    If I do H2 with cc-pvdz, wayy less than 1 million primitives, this computation takes 
    2m52s, 1.4 GB, difference being the nested vmaps and contraction sum, I assume.      
    If we do this same computation twice, it takes 5m44s, implying the compilation overhead is minimal. 
    Adding JIT on the vmap nest and running twice 5m29s
    Adding JIt to the inner vmap nest and running twice 5m25s 
    In principle, this means you could specialize the functions a bit more, and using 
    static_argnums in JIT without too much issue.

    Here the main issue appears to be the padding to 81 contraction lenght. Using
    np.where to only compute non redundant primitives doesnt appear to help much.
    Worst case, we are computing 810,000 primitives, which is less than 1 million,
    but the computation cost is 5x somehow due to contractions.
    

* Hamilton Schaefer VRR
    * Idea here is to first create (a 0 | 0 0), then transfer using HSRR to (a 0 | c 0),
    then use the horizontal relation. I have been unable to derive the algorithm for the loop structure
    for performing the HSRR, which means I also do not know how to eventually do the HRR.

# New proposals

1. Refactor Hunzinaga approach to collect all primitive indices, and primitive data,
    and jax.ops.index_add to G each primitive, one at a time. Downside: probably huge memory overhead
    for holding all the primitive data together. Upside: no contraction nonsense.
    Just big data flow. Could perhaps perform it in batches so G is built up incrementally.
    Implementation difficulty: easy

2. Refactor Hunzinaga approach to directly generate contracted shell quartet classes and insert
    into G at appropriate indices.
    This would involve passing a function 4 basis functions, each with a given
    angular momentum (S, P, D, F..) (l1,l2,l3,l4) set of exponents and 
    contraction coefficients (aa,bb,cc,dd,c1,c2,c3,c4),
    A set of cartesian coordinate centers (A,B,C,D)
    which are arrays of variable length, and then producing all angular momentum combinations
    and assigning to the appropriate place in G.
    Implementation difficulty: medium

    Algorithm structure:

    # function which computes all contracted integrals in a class
    @partial(jax.jit, static_argnums(...))
    def fill_quartet(G, 
    # UPDATE
    I have refactored Hunzinaga to do the contraction manually, it is somewhat slow but low memory.
    Almost all the time is compile time, it finishes in like 4 s after compiling.
    10,000 (ff|ff) contracted integrals, made up of 3x3x3x3 contractions each, 3m16s, 0.6 GB. 
    if (pp|pp), same otherwise, result is  2m51s, 0.5 GB.

    100,000 (pp|pp), 3x3x3x3 contractions, 3m37s, 0.8 GB. 
    
    One thing to do here is to refactor it so that each angular momentum case is covered
    within the function for each shell quartet, would need to scan and fill G manually.
    

3. Try to implement one-center obara saika promotion,  Hamilton Schaefer transfer, and Head-Gordon Pople
    horizontal recursion and hope that it is somehow quick in JAX.
    Implementation difficulty: hard

