The goal here is to experiment with alternatives to a niave
quadruple outer loop over primitive quartets.

1. What if its a single loop over quartets, enabled by cartesian product? fast?

2. What if you create a function, vmap it and collect primitives, indices, and call one big jax.ops.index_add?
(memory will be larger certainly)

3. What if intermeidates are shared better?


