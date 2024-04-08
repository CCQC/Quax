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

        nbf1 = basis1.nbf()
        nbf2 = basis2.nbf()

        if mode == 'core' and max_deriv_order > 0:
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
        self.dipole_p = jax.core.Primitive("dipole")
        self.dipole_deriv_p = jax.core.Primitive("dipole_deriv")
        self.quadrupole_p = jax.core.Primitive("quadrupole")
        self.quadrupole_deriv_p = jax.core.Primitive("quadrupole_deriv")

        # Register primitive evaluation rules
        self.overlap_p.def_impl(self.overlap_impl)
        self.overlap_deriv_p.def_impl(self.overlap_deriv_impl)
        self.kinetic_p.def_impl(self.kinetic_impl)
        self.kinetic_deriv_p.def_impl(self.kinetic_deriv_impl)
        self.potential_p.def_impl(self.potential_impl)
        self.potential_deriv_p.def_impl(self.potential_deriv_impl)
        self.dipole_p.def_impl(self.dipole_impl)
        self.dipole_deriv_p.def_impl(self.dipole_deriv_impl)
        self.quadrupole_p.def_impl(self.quadrupole_impl)
        self.quadrupole_deriv_p.def_impl(self.quadrupole_deriv_impl)

        # Register the JVP rules with JAX
        jax.interpreters.ad.primitive_jvps[self.overlap_p] = self.overlap_jvp
        jax.interpreters.ad.primitive_jvps[self.overlap_deriv_p] = self.overlap_deriv_jvp
        jax.interpreters.ad.primitive_jvps[self.kinetic_p] = self.kinetic_jvp
        jax.interpreters.ad.primitive_jvps[self.kinetic_deriv_p] = self.kinetic_deriv_jvp
        jax.interpreters.ad.primitive_jvps[self.potential_p] = self.potential_jvp
        jax.interpreters.ad.primitive_jvps[self.potential_deriv_p] = self.potential_deriv_jvp
        jax.interpreters.ad.primitive_jvps[self.dipole_p] = self.dipole_jvp
        jax.interpreters.ad.primitive_jvps[self.dipole_deriv_p] = self.dipole_deriv_jvp
        jax.interpreters.ad.primitive_jvps[self.quadrupole_p] = self.quadrupole_jvp
        jax.interpreters.ad.primitive_jvps[self.quadrupole_deriv_p] = self.quadrupole_deriv_jvp

        # Register the batching rules with JAX
        jax.interpreters.batching.primitive_batchers[self.overlap_deriv_p] = self.overlap_deriv_batch
        jax.interpreters.batching.primitive_batchers[self.kinetic_deriv_p] = self.kinetic_deriv_batch
        jax.interpreters.batching.primitive_batchers[self.potential_deriv_p] = self.potential_deriv_batch
        jax.interpreters.batching.primitive_batchers[self.dipole_deriv_p] = self.dipole_deriv_batch
        jax.interpreters.batching.primitive_batchers[self.quadrupole_deriv_p] = self.quadrupole_deriv_batch

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

    def dipole(self, geom):
        return self.dipole_p.bind(geom)

    def dipole_deriv(self, geom, deriv_vec):
        return self.dipole_deriv_p.bind(geom, deriv_vec)
    
    def quadrupole(self, geom):
        return self.quadrupole_p.bind(geom)

    def quadrupole_deriv(self, geom, deriv_vec):
        return self.quadrupole_deriv_p.bind(geom, deriv_vec)

    # Create primitive evaluation rules
    def overlap_impl(self, geom):
        S = libint_interface.compute_1e_int("overlap")
        S = S.reshape(self.nbf1, self.nbf2)
        return jnp.asarray(S)

    def kinetic_impl(self, geom):
        T = libint_interface.compute_1e_int("kinetic")
        T = T.reshape(self.nbf1, self.nbf2)
        return jnp.asarray(T)

    def potential_impl(self, geom):
        V = libint_interface.compute_1e_int("potential")
        V = V.reshape(self.nbf1, self.nbf2)
        return jnp.asarray(V)

    def dipole_impl(self, geom):
        Mu_X, Mu_Y, Mu_Z = libint_interface.compute_dipole_ints()
        Mu_X = Mu_X.reshape(self.nbf1, self.nbf2)
        Mu_Y = Mu_Y.reshape(self.nbf1, self.nbf2)
        Mu_Z = Mu_Z.reshape(self.nbf1, self.nbf2)
        return jnp.stack([Mu_X, Mu_Y, Mu_Z])
    
    def quadrupole_impl(self, geom):
        Mu_X, Mu_Y, Mu_Z, Th_XX, Th_XY,\
            Th_XZ, Th_YY, Th_YZ, Th_ZZ = libint_interface.compute_quadrupole_ints()
        Mu_X = Mu_X.reshape(self.nbf1, self.nbf2)
        Mu_Y = Mu_Y.reshape(self.nbf1, self.nbf2)
        Mu_Z = Mu_Z.reshape(self.nbf1, self.nbf2)
        Th_XX = Th_XX.reshape(self.nbf1, self.nbf2)
        Th_XY = Th_XY.reshape(self.nbf1, self.nbf2)
        Th_XZ = Th_XZ.reshape(self.nbf1, self.nbf2)
        Th_YY = Th_YY.reshape(self.nbf1, self.nbf2)
        Th_YZ = Th_YZ.reshape(self.nbf1, self.nbf2)
        Th_ZZ = Th_ZZ.reshape(self.nbf1, self.nbf2)
        return jnp.stack([Mu_X, Mu_Y, Mu_Z, Th_XX, Th_XY, Th_XZ, Th_YY, Th_YZ, Th_ZZ])

    def overlap_deriv_impl(self, geom, deriv_vec):
        deriv_vec = np.asarray(deriv_vec, int)
        deriv_order = np.sum(deriv_vec)
        idx = get_deriv_vec_idx(deriv_vec)

        if self.mode == 'core':
            S = self.overlap_derivatives[deriv_order-1][idx,:,:]
            return jnp.asarray(S)
        if self.mode == 'f12':
            S = libint_interface.compute_1e_deriv("overlap", deriv_vec)
            return jnp.asarray(S).reshape(self.nbf1,self.nbf2)
        elif self.mode == 'disk':
            if os.path.exists("oei_derivs.h5"):
                file_name = "oei_derivs.h5"
                dataset_name = "overlap_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order)
            elif os.path.exists("oei_partials.h5"):
                file_name = "oei_partials.h5"
                dataset_name = "overlap_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order) + "_" + str(idx)
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

        if self.mode == 'core':
            T = self.kinetic_derivatives[deriv_order-1][idx,:,:]
            return jnp.asarray(T)
        if self.mode == 'f12':
            T = libint_interface.compute_1e_deriv("kinetic", deriv_vec)
            return jnp.asarray(T).reshape(self.nbf1,self.nbf2)
        elif self.mode == 'disk':
            if os.path.exists("oei_derivs.h5"):
                file_name = "oei_derivs.h5"
                dataset_name = "kinetic_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order)
            elif os.path.exists("oei_partials.h5"):
                file_name = "oei_partials.h5"
                dataset_name = "kinetic_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order) + "_" + str(idx)
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

        if self.mode == 'core':
            V = self.potential_derivatives[deriv_order-1][idx,:,:]
            return jnp.asarray(V)
        if self.mode == 'f12':
            V = libint_interface.compute_1e_deriv("potential", deriv_vec)
            return jnp.asarray(V).reshape(self.nbf1,self.nbf2)
        elif self.mode == 'disk':
            if os.path.exists("oei_derivs.h5"):
                file_name = "oei_derivs.h5"
                dataset_name = "potential_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order)
            elif os.path.exists("oei_partials.h5"):
                file_name = "oei_partials.h5"
                dataset_name = "potential_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order) + "_" + str(idx)
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

    def dipole_deriv_impl(self, geom, deriv_vec):
        deriv_vec = np.asarray(deriv_vec, int)
        deriv_order = np.sum(deriv_vec)
        idx = get_deriv_vec_idx(deriv_vec)

        if self.mode == 'dipole':
            Mu_X, Mu_Y, Mu_Z = libint_interface.compute_dipole_derivs(deriv_vec)
            Mu_X = Mu_X.reshape(self.nbf1, self.nbf2)
            Mu_Y = Mu_Y.reshape(self.nbf1, self.nbf2)
            Mu_Z = Mu_Z.reshape(self.nbf1, self.nbf2)
            return jnp.stack([Mu_X, Mu_Y, Mu_Z])
        elif self.mode == 'disk':
            if os.path.exists("dipole_derivs.h5"):
                file_name = "dipole_derivs.h5"
                dataset1_name = "mu_x_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order)
                dataset2_name = "mu_y_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order)
                dataset3_name = "mu_z_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order)
            elif os.path.exists("dipole_partials.h5"):
                file_name = "dipole_partials.h5"
                dataset1_name = "mu_x_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order) + "_" + str(idx)
                dataset2_name = "mu_y_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order) + "_" + str(idx)
                dataset3_name = "mu_z_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order) + "_" + str(idx)
            else:
                raise Exception("Something went wrong reading integral derivative file")
            with h5py.File(file_name, 'r') as f:
                mu_x_set = f[dataset1_name]
                mu_y_set = f[dataset2_name]
                mu_z_set = f[dataset3_name]
                if len(mu_x_set.shape) == 3:
                    Mu_X = mu_x_set[:,:,idx]
                    Mu_Y = mu_y_set[:,:,idx]
                    Mu_Z = mu_z_set[:,:,idx]
                elif len(mu_x_set.shape) == 2:
                    Mu_X = mu_x_set[:,:]
                    Mu_Y = mu_y_set[:,:]
                    Mu_Z = mu_z_set[:,:]
                else:
                    raise Exception("Something went wrong reading integral derivative file")
            return jnp.stack([Mu_X, Mu_Y, Mu_Z])

    def quadrupole_deriv_impl(self, geom, deriv_vec):
        deriv_vec = np.asarray(deriv_vec, int)
        deriv_order = np.sum(deriv_vec)
        idx = get_deriv_vec_idx(deriv_vec)

        if self.mode == 'quadrupole':
            Mu_X, Mu_Y, Mu_Z, Th_XX, Th_XY,\
                Th_XZ, Th_YY, Th_YZ, Th_ZZ = libint_interface.compute_quadrupole_derivs()
            Mu_X = Mu_X.reshape(self.nbf1, self.nbf2)
            Mu_Y = Mu_Y.reshape(self.nbf1, self.nbf2)
            Mu_Z = Mu_Z.reshape(self.nbf1, self.nbf2)
            Th_XX = Th_XX.reshape(self.nbf1, self.nbf2)
            Th_XY = Th_XY.reshape(self.nbf1, self.nbf2)
            Th_XZ = Th_XZ.reshape(self.nbf1, self.nbf2)
            Th_YY = Th_YY.reshape(self.nbf1, self.nbf2)
            Th_YZ = Th_YZ.reshape(self.nbf1, self.nbf2)
            Th_ZZ = Th_ZZ.reshape(self.nbf1, self.nbf2)
            return jnp.stack([Mu_X, Mu_Y, Mu_Z, Th_XX, Th_XY, Th_XZ, Th_YY, Th_YZ, Th_ZZ])
        elif self.mode == 'disk':
            if os.path.exists("quadrupole_derivs.h5"):
                file_name = "quadrupole_derivs.h5"
                dataset1_name = "mu_x_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order)
                dataset2_name = "mu_y_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order)
                dataset3_name = "mu_z_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order)
                dataset4_name = "th_xx_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order)
                dataset5_name = "th_xy_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order)
                dataset6_name = "th_xz_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order)
                dataset7_name = "th_yy_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order)
                dataset8_name = "th_yz_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order)
                dataset9_name = "th_zz_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order)
            elif os.path.exists("quadrupole_partials.h5"):
                file_name = "quadrupole_partials.h5"
                dataset1_name = "mu_x_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order) + "_" + str(idx)
                dataset2_name = "mu_y_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order) + "_" + str(idx)
                dataset3_name = "mu_z_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order) + "_" + str(idx)
                dataset4_name = "th_xx_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order) + "_" + str(idx)
                dataset5_name = "th_xy_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order) + "_" + str(idx)
                dataset6_name = "th_xz_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order) + "_" + str(idx)
                dataset7_name = "th_yy_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order) + "_" + str(idx)
                dataset8_name = "th_yz_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order) + "_" + str(idx)
                dataset9_name = "th_zz_" + str(self.nbf1) + "_" + str(self.nbf2)\
                                          + "_deriv" + str(deriv_order) + "_" + str(idx)
            else:
                raise Exception("Something went wrong reading integral derivative file")
            with h5py.File(file_name, 'r') as f:
                mu_x_set = f[dataset1_name]
                mu_y_set = f[dataset2_name]
                mu_z_set = f[dataset3_name]
                th_xx_set = f[dataset1_name]
                th_xy_set = f[dataset2_name]
                th_xz_set = f[dataset3_name]
                th_yy_set = f[dataset1_name]
                th_yz_set = f[dataset2_name]
                th_zz_set = f[dataset3_name]
                if len(mu_x_set.shape) == 3:
                    Mu_X = mu_x_set[:,:,idx]
                    Mu_Y = mu_y_set[:,:,idx]
                    Mu_Z = mu_z_set[:,:,idx]
                    Th_XX = th_xx_set[:,:,idx]
                    Th_XY = th_xy_set[:,:,idx]
                    Th_XZ = th_xz_set[:,:,idx]
                    Th_YY = th_yy_set[:,:,idx]
                    Th_YZ = th_yz_set[:,:,idx]
                    Th_ZZ = th_zz_set[:,:,idx]
                elif len(mu_x_set.shape) == 2:
                    Mu_X = mu_x_set[:,:]
                    Mu_Y = mu_y_set[:,:]
                    Mu_Z = mu_z_set[:,:]
                    Th_XX = th_xx_set[:,:]
                    Th_XY = th_xy_set[:,:]
                    Th_XZ = th_xz_set[:,:]
                    Th_YY = th_yy_set[:,:]
                    Th_YZ = th_yz_set[:,:]
                    Th_ZZ = th_zz_set[:,:]
                else:
                    raise Exception("Something went wrong reading integral derivative file")
            return jnp.stack([Mu_X, Mu_Y, Mu_Z, Th_XX, Th_XY, Th_XZ, Th_YY, Th_YZ, Th_ZZ])

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

    def dipole_jvp(self, primals, tangents):
        geom, = primals
        primals_out = self.dipole(geom)
        tangents_out = self.dipole_deriv(geom, tangents[0])
        return primals_out, tangents_out

    def dipole_deriv_jvp(self, primals, tangents):
        geom, deriv_vec = primals
        primals_out = self.dipole_deriv(geom, deriv_vec)
        tangents_out = self.dipole_deriv(geom, deriv_vec + tangents[0])
        return primals_out, tangents_out

    def quadrupole_jvp(self, primals, tangents):
        geom, = primals
        primals_out = self.quadrupole(geom)
        tangents_out = self.quadrupole_deriv(geom, tangents[0])
        return primals_out, tangents_out

    def quadrupole_deriv_jvp(self, primals, tangents):
        geom, deriv_vec = primals
        primals_out = self.quadrupole_deriv(geom, deriv_vec)
        tangents_out = self.quadrupole_deriv(geom, deriv_vec + tangents[0])
        return primals_out, tangents_out

    # Define Batching rules, this is only needed since jax.jacfwd will call vmap on the JVP's
    # of each oei function
    # When the input argument of deriv_batch is batched along the 0'th axis
    # we want to evaluate every 2d slice, gather up a (ncart, n,n) array,
    # (expand dims at 0 and concatenate at 0)
    # and then return the results, indicating the out batch axis
    # is in the 0th position (return results, 0)

    def overlap_deriv_batch(self, batched_args, batch_dims):
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

    def dipole_deriv_batch(self, batched_args, batch_dims):
        geom_batch, deriv_batch = batched_args
        geom_dim, deriv_dim = batch_dims
        results = []
        for i in deriv_batch:
            tmp1, tmp2, tmp3 = self.dipole_deriv(geom_batch, i)
            mu_x = jnp.expand_dims(tmp1, axis=0)
            mu_y = jnp.expand_dims(tmp2, axis=0)
            mu_z = jnp.expand_dims(tmp3, axis=0)
            results.append(jnp.stack([mu_x, mu_y, mu_z], axis=1))
        results = jnp.concatenate(results, axis=0)
        return results, 0

    def quadrupole_deriv_batch(self, batched_args, batch_dims):
        geom_batch, deriv_batch = batched_args
        geom_dim, deriv_dim = batch_dims
        results = []
        for i in deriv_batch:
            tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9 = self.quadrupole_deriv(geom_batch, i)
            mu_x = jnp.expand_dims(tmp1, axis=0)
            mu_y = jnp.expand_dims(tmp2, axis=0)
            mu_z = jnp.expand_dims(tmp3, axis=0)
            th_xx = jnp.expand_dims(tmp4, axis=0)
            th_xy = jnp.expand_dims(tmp5, axis=0)
            th_xz = jnp.expand_dims(tmp6, axis=0)
            th_yy = jnp.expand_dims(tmp7, axis=0)
            th_yz = jnp.expand_dims(tmp8, axis=0)
            th_zz = jnp.expand_dims(tmp9, axis=0)
            results.append(jnp.stack([mu_x, mu_y, mu_z, th_xx, th_xy, th_xz, th_yy, th_yz, th_zz], axis=1))
        results = jnp.concatenate(results, axis=0)
        return results, 0

