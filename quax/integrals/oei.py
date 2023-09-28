import jax
import jax.numpy as jnp
import numpy as np
import h5py
import os
import psi4
from . import libint_interface
from ..utils import get_deriv_vec_idx, how_many_derivs

jax.config.update("jax_enable_x64", True)

class OEI(object):

    def __init__(self, basis1, basis2, xyz_path, max_deriv_order, mode):
        with open(xyz_path, 'r') as f:
            tmp = f.read()
        molecule = psi4.core.Molecule.from_string(tmp, 'xyz+')
        natoms = molecule.natom()

        bs1 = psi4.core.BasisSet.build(molecule, 'BASIS', basis1, puream=0)
        bs2 = psi4.core.BasisSet.build(molecule, 'BASIS', basis2, puream=0)
        nbf1 = bs1.nbf()
        nbf2 = bs2.nbf()

        if 'core' in mode and max_deriv_order > 0:
            # A list of OEI derivative tensors, containing only unique elements
            # corresponding to upper hypertriangle (since derivative tensors are symmetric)
            # Length of tuple is maximum deriv order, each array is (upper triangle derivatives,nbf,nbf)
            # Then when JAX calls JVP, read appropriate slice
            self.overlap_derivatives = []
            self.kinetic_derivatives = []
            self.potential_derivatives = []
            for i in range(max_deriv_order):
                n_unique_derivs = how_many_derivs(natoms, i + 1)
                oei_deriv = libint_interface.oei_deriv_core(i + 1)
                self.overlap_derivatives.append(oei_deriv[0].reshape(n_unique_derivs, nbf1, nbf2))
                self.kinetic_derivatives.append(oei_deriv[1].reshape(n_unique_derivs, nbf1, nbf2))
                self.potential_derivatives.append(oei_deriv[2].reshape(n_unique_derivs, nbf1, nbf2))

        self.mode = mode
        self.nbf1 = nbf1
        self.nbf2 = nbf2

        # Create new JAX primitives for overlap, kinetic, potential evaluation and their derivatives
        self.overlap_p = jax.core.Primitive("overlap")
        self.overlap_deriv_p = jax.core.Primitive("overlap_deriv")
        self.kinetic_p = jax.core.Primitive("kinetic")
        self.kinetic_deriv_p = jax.core.Primitive("kinetic_deriv")
        self.potential_p = jax.core.Primitive("potential")
        self.potential_deriv_p = jax.core.Primitive("potential_deriv")

        # Register primitive evaluation rules
        self.overlap_p.def_impl(self.overlap_impl)
        self.overlap_deriv_p.def_impl(self.overlap_deriv_impl)
        self.kinetic_p.def_impl(self.kinetic_impl)
        self.kinetic_deriv_p.def_impl(self.kinetic_deriv_impl)
        self.potential_p.def_impl(self.potential_impl)
        self.potential_deriv_p.def_impl(self.potential_deriv_impl)

        # Register the JVP rules with JAX
        jax.interpreters.ad.primitive_jvps[self.overlap_p] = self.overlap_jvp
        jax.interpreters.ad.primitive_jvps[self.overlap_deriv_p] = self.overlap_deriv_jvp
        jax.interpreters.ad.primitive_jvps[self.kinetic_p] = self.kinetic_jvp
        jax.interpreters.ad.primitive_jvps[self.kinetic_deriv_p] = self.kinetic_deriv_jvp
        jax.interpreters.ad.primitive_jvps[self.potential_p] = self.potential_jvp
        jax.interpreters.ad.primitive_jvps[self.potential_deriv_p] = self.potential_deriv_jvp

        # Register the batching rules with JAX
        jax.interpreters.batching.primitive_batchers[self.overlap_deriv_p] = self.overlap_deriv_batch
        jax.interpreters.batching.primitive_batchers[self.kinetic_deriv_p] = self.kinetic_deriv_batch
        jax.interpreters.batching.primitive_batchers[self.potential_deriv_p] = self.potential_deriv_batch

    # Create functions to call primitives
    def overlap(self, geom):
        return self.overlap_p.bind(geom)

    def overlap_deriv(self, geom, deriv_vec):
        return self.overlap_deriv_p.bind(geom, deriv_vec)

    def kinetic(self, geom):
        return self.kinetic_p.bind(geom)

    def kinetic_deriv(self, geom, deriv_vec):
        return self.kinetic_deriv_p.bind(geom, deriv_vec)

    def potential(self, geom):
        return self.potential_p.bind(geom)

    def potential_deriv(self, geom, deriv_vec):
        return self.potential_deriv_p.bind(geom, deriv_vec)

    # Create primitive evaluation rules
    def overlap_impl(self, geom):
        S = libint_interface.overlap()
        S = S.reshape(self.nbf1, self.nbf2)
        return jnp.asarray(S)

    def kinetic_impl(self, geom):
        T = libint_interface.kinetic()
        T = T.reshape(self.nbf1, self.nbf2)
        return jnp.asarray(T)

    def potential_impl(self, geom):
        V = libint_interface.potential()
        V = V.reshape(self.nbf1, self.nbf2)
        return jnp.asarray(V)

    def overlap_deriv_impl(self, geom, deriv_vec):
        deriv_vec = np.asarray(deriv_vec, int)
        deriv_order = np.sum(deriv_vec)
        idx = get_deriv_vec_idx(deriv_vec)

        if 'core' in self.mode:
            S = self.overlap_derivatives[deriv_order-1][idx,:,:]
            return jnp.asarray(S)
        elif 'disk' in self.mode:
            if os.path.exists("oei_derivs.h5"):
                file_name = "oei_derivs.h5"
                dataset_name = "overlap_deriv" + str(deriv_order)
            elif os.path.exists("oei_partials.h5"):
                file_name = "oei_partials.h5"
                dataset_name = "overlap_deriv" + str(deriv_order) + "_" + str(idx)
            else:
                raise Exception("Something went wrong reading integral derivative file")
            with h5py.File(file_name, 'r') as f:
                data_set = f[dataset_name]
                if len(data_set.shape) == 3:
                    S = data_set[:,:,idx]
                elif len(data_set.shape) == 2:
                    S = data_set[:,:]
                else:
                    raise Exception("Something went wrong reading integral derivative file")
            return jnp.asarray(S)

    def kinetic_deriv_impl(self, geom, deriv_vec):
        deriv_vec = np.asarray(deriv_vec, int)
        deriv_order = np.sum(deriv_vec)
        idx = get_deriv_vec_idx(deriv_vec)

        if 'core' in self.mode:
            T = self.kinetic_derivatives[deriv_order-1][idx,:,:]
            return jnp.asarray(T)
        elif 'disk' in self.mode:
            if os.path.exists("oei_derivs.h5"):
                file_name = "oei_derivs.h5"
                dataset_name = "kinetic_deriv" + str(deriv_order)
            elif os.path.exists("oei_partials.h5"):
                file_name = "oei_partials.h5"
                dataset_name = "kinetic_deriv" + str(deriv_order) + "_" + str(idx)
            else:
                raise Exception("Something went wrong reading integral derivative file")
            with h5py.File(file_name, 'r') as f:
                data_set = f[dataset_name]
                if len(data_set.shape) == 3:
                    T = data_set[:,:,idx]
                elif len(data_set.shape) == 2:
                    T = data_set[:,:]
                else:
                    raise Exception("Something went wrong reading integral derivative file")
            return jnp.asarray(T)

    def potential_deriv_impl(self, geom, deriv_vec):
        deriv_vec = np.asarray(deriv_vec, int)
        deriv_order = np.sum(deriv_vec)
        idx = get_deriv_vec_idx(deriv_vec)

        if 'core' in self.mode:
            V = self.potential_derivatives[deriv_order-1][idx,:,:]
            return jnp.asarray(V)
        elif 'disk' in self.mode:
            if os.path.exists("oei_derivs.h5"):
                file_name = "oei_derivs.h5"
                dataset_name = "potential_deriv" + str(deriv_order)
            elif os.path.exists("oei_partials.h5"):
                file_name = "oei_partials.h5"
                dataset_name = "potential_deriv" + str(deriv_order) + "_" + str(idx)
            else:
                raise Exception("Something went wrong reading integral derivative file")
            with h5py.File(file_name, 'r') as f:
                data_set = f[dataset_name]
                if len(data_set.shape) == 3:
                    V = data_set[:,:,idx]
                elif len(data_set.shape) == 2:
                    V = data_set[:,:]
                else:
                    raise Exception("Something went wrong reading integral derivative file")
            return jnp.asarray(V)

    def overlap_jvp(self, primals, tangents):
        geom, = primals
        primals_out = self.overlap(geom)
        tangents_out = self.overlap_deriv(geom, tangents[0])
        return primals_out, tangents_out

    def overlap_deriv_jvp(self, primals, tangents):
        geom, deriv_vec = primals
        primals_out = self.overlap_deriv(geom, deriv_vec)
        tangents_out = self.overlap_deriv(geom, deriv_vec + tangents[0])
        return primals_out, tangents_out

    def kinetic_jvp(self, primals, tangents):
        geom, = primals
        primals_out = self.kinetic(geom)
        tangents_out = self.kinetic_deriv(geom, tangents[0])
        return primals_out, tangents_out

    def kinetic_deriv_jvp(self, primals, tangents):
        geom, deriv_vec = primals
        primals_out = self.kinetic_deriv(geom, deriv_vec)
        tangents_out = self.kinetic_deriv(geom, deriv_vec + tangents[0])
        return primals_out, tangents_out

    def potential_jvp(self, primals, tangents):
        geom, = primals
        primals_out = self.potential(geom)
        tangents_out = self.potential_deriv(geom, tangents[0])
        return primals_out, tangents_out

    def potential_deriv_jvp(self, primals, tangents):
        geom, deriv_vec = primals
        primals_out = self.potential_deriv(geom, deriv_vec)
        tangents_out = self.potential_deriv(geom, deriv_vec + tangents[0])
        return primals_out, tangents_out

    # Define Batching rules, this is only needed since jax.jacfwd will call vmap on the JVP's
    # of each oei function
    def overlap_deriv_batch(self, batched_args, batch_dims):
        # When the input argument of deriv_batch is batched along the 0'th axis
        # we want to evaluate every 2d slice, gather up a (ncart, n,n) array,
        # (expand dims at 0 and concatenate at 0)
        # and then return the results, indicating the out batch axis
        # is in the 0th position (return results, 0)
        geom_batch, deriv_batch = batched_args
        geom_dim, deriv_dim = batch_dims
        results = []
        for i in deriv_batch:
            tmp = self.overlap_deriv(geom_batch, i)
            results.append(jnp.expand_dims(tmp, axis=0))
        results = jnp.concatenate(results, axis=0)
        return results, 0

    def kinetic_deriv_batch(self, batched_args, batch_dims):
        geom_batch, deriv_batch = batched_args
        geom_dim, deriv_dim = batch_dims
        results = []
        for i in deriv_batch:
            tmp = self.kinetic_deriv(geom_batch, i)
            results.append(jnp.expand_dims(tmp, axis=0))
        results = jnp.concatenate(results, axis=0)
        return results, 0

    def potential_deriv_batch(self, batched_args, batch_dims):
        geom_batch, deriv_batch = batched_args
        geom_dim, deriv_dim = batch_dims
        results = []
        for i in deriv_batch:
            tmp = self.potential_deriv(geom_batch, i)
            results.append(jnp.expand_dims(tmp, axis=0))
        results = jnp.concatenate(results, axis=0)
        return results, 0

