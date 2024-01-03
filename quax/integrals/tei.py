import jax 
import jax.numpy as jnp
import numpy as np
import h5py
import os
import psi4
from . import libint_interface
from ..utils import get_deriv_vec_idx, how_many_derivs

jax.config.update("jax_enable_x64", True)

class TEI(object):

    def __init__(self, basis1, basis2, basis3, basis4, xyz_path, max_deriv_order, options, mode):
        with open(xyz_path, 'r') as f:
            tmp = f.read()
        molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
        natoms = molecule.natom()

        nbf1 = basis1.nbf()
        nbf2 = basis2.nbf()
        nbf3 = basis3.nbf()
        nbf4 = basis4.nbf()

        if mode == 'core' and max_deriv_order > 0:
            # A list of ERI derivative tensors, containing only unique elements
            # corresponding to upper hypertriangle (since derivative tensors are symmetric)
            # Length of tuple is maximum deriv order, each array is (upper triangle derivatives,nbf,nbf,nbf,nbf)
            # Then when JAX calls JVP, read appropriate slice
            self.eri_derivatives = []
            for i in range(max_deriv_order):
                n_unique_derivs = how_many_derivs(natoms, i + 1)
                eri_deriv = libint_interface.eri_deriv_core(i + 1).reshape(n_unique_derivs, nbf1, nbf2, nbf3, nbf4)
                self.eri_derivatives.append(eri_deriv)

        self.mode = mode
        self.nbf1 = nbf1
        self.nbf2 = nbf2
        self.nbf3 = nbf3
        self.nbf4 = nbf4

        # Create new JAX primitive for TEI evaluation
        self.eri_p = jax.core.Primitive("eri")
        self.eri_deriv_p = jax.core.Primitive("eri_deriv")
        self.f12_p = jax.core.Primitive("f12")
        self.f12_deriv_p = jax.core.Primitive("f12_deriv")
        self.f12_squared_p = jax.core.Primitive("f12_squared")
        self.f12_squared_deriv_p = jax.core.Primitive("f12_squared_deriv")
        self.f12g12_p = jax.core.Primitive("f12g12")
        self.f12g12_deriv_p = jax.core.Primitive("f12g12_deriv")
        self.f12_double_commutator_p = jax.core.Primitive("f12_double_commutator")
        self.f12_double_commutator_deriv_p = jax.core.Primitive("f12_double_commutator_deriv")

        # Register primitive evaluation rules
        self.eri_p.def_impl(self.eri_impl)
        self.eri_deriv_p.def_impl(self.eri_deriv_impl)
        self.f12_p.def_impl(self.f12_impl)
        self.f12_deriv_p.def_impl(self.f12_deriv_impl)
        self.f12_squared_p.def_impl(self.f12_squared_impl)
        self.f12_squared_deriv_p.def_impl(self.f12_squared_deriv_impl)
        self.f12g12_p.def_impl(self.f12g12_impl)
        self.f12g12_deriv_p.def_impl(self.f12g12_deriv_impl)
        self.f12_double_commutator_p.def_impl(self.f12_double_commutator_impl)
        self.f12_double_commutator_deriv_p.def_impl(self.f12_double_commutator_deriv_impl)

        # Register the JVP rules with JAX
        jax.interpreters.ad.primitive_jvps[self.eri_p] = self.eri_jvp
        jax.interpreters.ad.primitive_jvps[self.eri_deriv_p] = self.eri_deriv_jvp
        jax.interpreters.ad.primitive_jvps[self.f12_p] = self.f12_jvp
        jax.interpreters.ad.primitive_jvps[self.f12_deriv_p] = self.f12_deriv_jvp
        jax.interpreters.ad.primitive_jvps[self.f12_squared_p] = self.f12_squared_jvp
        jax.interpreters.ad.primitive_jvps[self.f12_squared_deriv_p] = self.f12_squared_deriv_jvp
        jax.interpreters.ad.primitive_jvps[self.f12g12_p] = self.f12g12_jvp
        jax.interpreters.ad.primitive_jvps[self.f12g12_deriv_p] = self.f12g12_deriv_jvp
        jax.interpreters.ad.primitive_jvps[self.f12_double_commutator_p] = self.f12_double_commutator_jvp
        jax.interpreters.ad.primitive_jvps[self.f12_double_commutator_deriv_p] = self.f12_double_commutator_deriv_jvp

        # Register tei_deriv batching rule with JAX
        jax.interpreters.batching.primitive_batchers[self.eri_deriv_p] = self.eri_deriv_batch
        jax.interpreters.batching.primitive_batchers[self.f12_deriv_p] = self.f12_deriv_batch
        jax.interpreters.batching.primitive_batchers[self.f12_squared_deriv_p] = self.f12_squared_deriv_batch
        jax.interpreters.batching.primitive_batchers[self.f12g12_deriv_p] = self.f12g12_deriv_batch
        jax.interpreters.batching.primitive_batchers[self.f12_double_commutator_deriv_p] = self.f12_double_commutator_deriv_batch

    # Create functions to call primitives
    def eri(self, geom):
        return self.eri_p.bind(geom)

    def eri_deriv(self, geom, deriv_vec):
        return self.eri_deriv_p.bind(geom, deriv_vec)

    def f12(self, geom, beta):
        return self.f12_p.bind(geom, beta)

    def f12_deriv(self, geom, beta, deriv_vec):
        return self.f12_deriv_p.bind(geom, beta, deriv_vec)

    def f12_squared(self, geom, beta):
        return self.f12_squared_p.bind(geom, beta)

    def f12_squared_deriv(self, geom, beta, deriv_vec):
        return self.f12_squared_deriv_p.bind(geom, beta, deriv_vec)

    def f12g12(self, geom, beta):
        return self.f12g12_p.bind(geom, beta)

    def f12g12_deriv(self, geom, beta, deriv_vec):
        return self.f12g12_deriv_p.bind(geom, beta, deriv_vec)

    def f12_double_commutator(self, geom, beta):
        return self.f12_double_commutator_p.bind(geom, beta)

    def f12_double_commutator_deriv(self, geom, beta, deriv_vec):
        return self.f12_double_commutator_deriv_p.bind(geom, beta, deriv_vec)

    # Create primitive evaluation rules
    def eri_impl(self, geom):
        G = libint_interface.eri()
        G = G.reshape(self.nbf1, self.nbf2, self.nbf3, self.nbf4)
        return jnp.asarray(G)

    def f12_impl(self, geom, beta):
        F = libint_interface.f12(beta)
        F = F.reshape(self.nbf1, self.nbf2, self.nbf3, self.nbf4)
        return jnp.asarray(F)

    def f12_squared_impl(self, geom, beta):
        F = libint_interface.f12_squared(beta)
        F = F.reshape(self.nbf1, self.nbf2, self.nbf3, self.nbf4)
        return jnp.asarray(F)

    def f12g12_impl(self, geom, beta):
        F = libint_interface.f12g12(beta)
        F = F.reshape(self.nbf1, self.nbf2, self.nbf3, self.nbf4)
        return jnp.asarray(F)
    
    def f12_double_commutator_impl(self, geom, beta):
        F = libint_interface.f12_double_commutator(beta)
        F = F.reshape(self.nbf1, self.nbf2, self.nbf3, self.nbf4)
        return jnp.asarray(F)

    def eri_deriv_impl(self, geom, deriv_vec):
        deriv_vec = np.asarray(deriv_vec, int)
        deriv_order = np.sum(deriv_vec)
        idx = get_deriv_vec_idx(deriv_vec)

        # Use eri derivatives in memory
        if self.mode == 'core':
            G = self.eri_derivatives[deriv_order-1][idx,:,:,:,:]
            return jnp.asarray(G)

        if self.mode == 'f12':
            G = libint_interface.eri_deriv(deriv_vec)
            return jnp.asarray(G).reshape(self.nbf1, self.nbf2, self.nbf3, self.nbf4)

        # Read from disk
        elif self.mode == 'disk':
            # By default, look for full derivative tensor file with datasets named (type)_deriv(order)
            if os.path.exists("eri_derivs.h5"):
                file_name = "eri_derivs.h5"
                dataset_name = "eri_deriv" + str(deriv_order)
            # if not found, look for partial derivative tensor file with datasets named (type)_deriv(order)_(flattened_uppertri_idx)
            elif os.path.exists("eri_partials.h5"):
                file_name = "eri_partials.h5"
                dataset_name = "eri_deriv" + str(deriv_order) + "_" + str(idx)
            else:
                raise Exception("ERI derivatives not found on disk")

            with h5py.File(file_name, 'r') as f:
                data_set = f[dataset_name]
                if len(data_set.shape) == 5:
                    G = data_set[:,:,:,:,idx]
                elif len(data_set.shape) == 4:
                    G = data_set[:,:,:,:]
                else:
                    raise Exception("Something went wrong reading integral derivative file")
            return jnp.asarray(G)

    def f12_deriv_impl(self, geom, beta, deriv_vec):
        deriv_vec = np.asarray(deriv_vec, int)
        deriv_order = np.sum(deriv_vec)
        idx = get_deriv_vec_idx(deriv_vec)

        # Use f12 derivatives in memory
        if self.mode == 'f12':
            F = libint_interface.f12_deriv(beta, deriv_vec)
            return jnp.asarray(F).reshape(self.nbf1, self.nbf2, self.nbf3, self.nbf4)

        # Read from disk
        elif self.mode == 'disk':
            # By default, look for full derivative tensor file with datasets named (type)_deriv(order)
            if os.path.exists("f12_derivs.h5"):
                file_name = "f12_derivs.h5"
                dataset_name = "f12_deriv" + str(deriv_order)
            # if not found, look for partial derivative tensor file with datasets named (type)_deriv(order)_(flattened_uppertri_idx)
            elif os.path.exists("f12_partials.h5"):
                file_name = "f12_partials.h5"
                dataset_name = "f12_deriv" + str(deriv_order) + "_" + str(idx)
            else:
                raise Exception("F12 derivatives not found on disk")

            with h5py.File(file_name, 'r') as f:
                data_set = f[dataset_name]
                if len(data_set.shape) == 5:
                    F = data_set[:,:,:,:,idx]
                elif len(data_set.shape) == 4:
                    F = data_set[:,:,:,:]
                else:
                    raise Exception("Something went wrong reading integral derivative file")
            return jnp.asarray(F)

    def f12_squared_deriv_impl(self, geom, beta, deriv_vec):
        deriv_vec = np.asarray(deriv_vec, int)
        deriv_order = np.sum(deriv_vec)
        idx = get_deriv_vec_idx(deriv_vec)

        # Use f12 squared derivatives in memory
        if self.mode == 'f12':
            F = libint_interface.f12_squared_deriv(beta, deriv_vec)
            return jnp.asarray(F).reshape(self.nbf1, self.nbf2, self.nbf3, self.nbf4)

        # Read from disk
        elif self.mode == 'disk':
            # By default, look for full derivative tensor file with datasets named (type)_deriv(order)
            if os.path.exists("f12_squared_derivs.h5"):
                file_name = "f12_squared_derivs.h5"
                dataset_name = "f12_squared_deriv" + str(deriv_order)
            # if not found, look for partial derivative tensor file with datasets named (type)_deriv(order)_(flattened_uppertri_idx)
            elif os.path.exists("f12_squared_partials.h5"):
                file_name = "f12_squared_partials.h5"
                dataset_name = "f12_squared_deriv" + str(deriv_order) + "_" + str(idx)
            else:
                raise Exception("F12 Squared derivatives not found on disk")

            with h5py.File(file_name, 'r') as f:
                data_set = f[dataset_name]
                if len(data_set.shape) == 5:
                    F = data_set[:,:,:,:,idx]
                elif len(data_set.shape) == 4:
                    F = data_set[:,:,:,:]
                else:
                    raise Exception("Something went wrong reading integral derivative file")
            return jnp.asarray(F)

    def f12g12_deriv_impl(self, geom, beta, deriv_vec):
        deriv_vec = np.asarray(deriv_vec, int)
        deriv_order = np.sum(deriv_vec)
        idx = get_deriv_vec_idx(deriv_vec)

        # Use f12g12 derivatives in memory
        if self.mode == 'f12':
            F = libint_interface.f12g12_deriv(beta, deriv_vec)
            return jnp.asarray(F).reshape(self.nbf1, self.nbf2, self.nbf3, self.nbf4)

        # Read from disk
        elif self.mode == 'disk':
            # By default, look for full derivative tensor file with datasets named (type)_deriv(order)
            if os.path.exists("f12g12_derivs.h5"):
                file_name = "f12g12_derivs.h5"
                dataset_name = "f12g12_deriv" + str(deriv_order)
            # if not found, look for partial derivative tensor file with datasets named (type)_deriv(order)_(flattened_uppertri_idx)
            elif os.path.exists("f12g12_partials.h5"):
                file_name = "f12g12_partials.h5"
                dataset_name = "f12g12_deriv" + str(deriv_order) + "_" + str(idx)
            else:
                raise Exception("F12G12 derivatives not found on disk")

            with h5py.File(file_name, 'r') as f:
                data_set = f[dataset_name]
                if len(data_set.shape) == 5:
                    F = data_set[:,:,:,:,idx]
                elif len(data_set.shape) == 4:
                    F = data_set[:,:,:,:]
                else:
                    raise Exception("Something went wrong reading integral derivative file")
            return jnp.asarray(F)

    def f12_double_commutator_deriv_impl(self, geom, beta, deriv_vec):
        deriv_vec = np.asarray(deriv_vec, int)
        deriv_order = np.sum(deriv_vec)
        idx = get_deriv_vec_idx(deriv_vec)

        # Use f12 double commutator derivatives in memory
        if self.mode == 'f12':
            F = libint_interface.f12_double_commutator_deriv(beta, deriv_vec)
            return jnp.asarray(F).reshape(self.nbf1, self.nbf2, self.nbf3, self.nbf4)

        # Read from disk
        elif self.mode == 'disk':
            # By default, look for full derivative tensor file with datasets named (type)_deriv(order)
            if os.path.exists("f12_double_commutator_derivs.h5"):
                file_name = "f12_double_commutator_derivs.h5"
                dataset_name = "f12_double_commutator_deriv" + str(deriv_order)
            # if not found, look for partial derivative tensor file with datasets named (type)_deriv(order)_(flattened_uppertri_idx)
            elif os.path.exists("f12_double_commutator_partials.h5"):
                file_name = "f12_double_commutator_partials.h5"
                dataset_name = "f12_double_commutator_deriv" + str(deriv_order) + "_" + str(idx)
            else:
                raise Exception("F12 Double Commutator derivatives not found on disk")

            with h5py.File(file_name, 'r') as f:
                data_set = f[dataset_name]
                if len(data_set.shape) == 5:
                    F = data_set[:,:,:,:,idx]
                elif len(data_set.shape) == 4:
                    F = data_set[:,:,:,:]
                else:
                    raise Exception("Something went wrong reading integral derivative file")
            return jnp.asarray(F)

    # Create Jacobian-vector product rule, which given some input args (primals)
    # and a tangent std basis vector (tangent), returns the function evaluated at that point (primals_out)
    # and the slice of the Jacobian (tangents_out)
    # For high-order differentiation, we add the current value of deriv_vec to the incoming tangent vector

    def eri_jvp(self, primals, tangents):
        geom, = primals
        primals_out = self.eri(geom)
        tangents_out = self.eri_deriv(geom, tangents[0])
        return primals_out, tangents_out

    def eri_deriv_jvp(self, primals, tangents):
        geom, deriv_vec = primals
        primals_out = self.eri_deriv(geom, deriv_vec)
        tangents_out = self.eri_deriv(geom, deriv_vec + tangents[0])
        return primals_out, tangents_out

    def f12_jvp(self, primals, tangents):
        geom, beta = primals
        primals_out = self.f12(geom, beta)
        tangents_out = self.f12_deriv(geom, beta, tangents[0])
        return primals_out, tangents_out

    def f12_deriv_jvp(self, primals, tangents):
        geom, beta, deriv_vec = primals
        primals_out = self.f12_deriv(geom, beta, deriv_vec)
        tangents_out = self.f12_deriv(geom, beta, deriv_vec + tangents[0])
        return primals_out, tangents_out

    def f12_squared_jvp(self, primals, tangents):
        geom, beta = primals
        primals_out = self.f12_squared(geom, beta)
        tangents_out = self.f12_squared_deriv(geom, beta, tangents[0])
        return primals_out, tangents_out

    def f12_squared_deriv_jvp(self, primals, tangents):
        geom, beta, deriv_vec = primals
        primals_out = self.f12_squared_deriv(geom, beta, deriv_vec)
        tangents_out = self.f12_squared_deriv(geom, beta, deriv_vec + tangents[0])
        return primals_out, tangents_out

    def f12g12_jvp(self, primals, tangents):
        geom, beta = primals
        primals_out = self.f12g12(geom, beta)
        tangents_out = self.f12g12_deriv(geom, beta, tangents[0])
        return primals_out, tangents_out

    def f12g12_deriv_jvp(self, primals, tangents):
        geom, beta, deriv_vec = primals
        primals_out = self.f12g12_deriv(geom, beta, deriv_vec)
        tangents_out = self.f12g12_deriv(geom, beta, deriv_vec + tangents[0])
        return primals_out, tangents_out

    def f12_double_commutator_jvp(self, primals, tangents):
        geom, beta = primals
        primals_out = self.f12_double_commutator(geom, beta)
        tangents_out = self.f12_double_commutator_deriv(geom, beta, tangents[0])
        return primals_out, tangents_out

    def f12_double_commutator_deriv_jvp(self, primals, tangents):
        geom, beta, deriv_vec = primals
        primals_out = self.f12_double_commutator_deriv(geom, beta, deriv_vec)
        tangents_out = self.f12_double_commutator_deriv(geom, beta, deriv_vec + tangents[0])
        return primals_out, tangents_out

    # Define Batching rules, this is only needed since jax.jacfwd will call vmap on the JVP of tei
    # When the input argument of deriv_batch is batched along the 0'th axis
    # we want to evaluate every 4d slice, gather up a (ncart, n,n,n,n) array,
    # (expand dims at 0 and concatenate at 0)
    # and then return the results, indicating the out batch axis
    # is in the 0th position (return results, 0)

    def eri_deriv_batch(self, batched_args, batch_dims):
        geom_batch, deriv_batch = batched_args
        geom_dim, deriv_dim = batch_dims
        results = []
        for i in deriv_batch:
            tmp = self.eri_deriv(geom_batch, i)
            results.append(jnp.expand_dims(tmp, axis=0))
        results = jnp.concatenate(results, axis=0)
        return results, 0
    
    def f12_deriv_batch(self, batched_args, batch_dims):
        geom_batch, beta_batch, deriv_batch = batched_args
        geom_dim, beta_dim, deriv_dim = batch_dims
        results = []
        for i in deriv_batch:
            tmp = self.f12_deriv(geom_batch, beta_batch, i)
            results.append(jnp.expand_dims(tmp, axis=0))
        results = jnp.concatenate(results, axis=0)
        return results, 0

    def f12_squared_deriv_batch(self, batched_args, batch_dims):
        geom_batch, beta_batch, deriv_batch = batched_args
        geom_dim, beta_dim, deriv_dim = batch_dims
        results = []
        for i in deriv_batch:
            tmp = self.f12_squared_deriv(geom_batch, beta_batch, i)
            results.append(jnp.expand_dims(tmp, axis=0))
        results = jnp.concatenate(results, axis=0)
        return results, 0

    def f12g12_deriv_batch(self, batched_args, batch_dims):
        geom_batch, beta_batch, deriv_batch = batched_args
        geom_dim, beta_dim, deriv_dim = batch_dims
        results = []
        for i in deriv_batch:
            tmp = self.f12g12_deriv(geom_batch, beta_batch, i)
            results.append(jnp.expand_dims(tmp, axis=0))
        results = jnp.concatenate(results, axis=0)
        return results, 0

    def f12_double_commutator_deriv_batch(self, batched_args, batch_dims):
        geom_batch, beta_batch, deriv_batch = batched_args
        geom_dim, beta_dim, deriv_dim = batch_dims
        results = []
        for i in deriv_batch:
            tmp = self.f12_double_commutator_deriv(geom_batch, beta_batch, i)
            results.append(jnp.expand_dims(tmp, axis=0))
        results = jnp.concatenate(results, axis=0)
        return results, 0
