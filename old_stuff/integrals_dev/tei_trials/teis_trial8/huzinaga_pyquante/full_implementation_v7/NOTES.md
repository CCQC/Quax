This implementation attempts a massive outer loop over primitive quartets rather than shell quartets which then perform contracrtion loops.
Idea here, if it even works, is to allow for constant size arrays which are not padded, which should enable more JAX flexibility
