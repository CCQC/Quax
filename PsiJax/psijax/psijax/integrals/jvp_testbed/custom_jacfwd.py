
def my_jacfwd(f):
    """A basic version of jax.jacfwd, assumes only one argument, no static args, etc"""
    def jacfun(x):
        # create little function that grabs tangents
        _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
        # evaluate tangents on standard basis
        Jt = jax.vmap(_jvp, in_axes=1)(np.eye(len(x)))
        return np.transpose(Jt)
    return jacfun

def my_jacfwd_novmap(f):
    """A basic version of jax.jacfwd, with no vmap. assumes only one argument, no static args, etc"""
    def jacfun(x):
        # create little function that grabs tangents (second arg returned, hence [1])
        _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
        # evaluate tangents on standard basis. Note we are only mapping over tangents arg of jvp
        #Jt = jax.vmap(_jvp, in_axes=1)(np.eye(len(x)))
        Jt = np.asarray([_jvp(i) for i in np.eye(len(x))])
        return np.transpose(Jt)
    return jacfun
