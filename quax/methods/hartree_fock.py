import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .ints import compute_integrals, compute_dipole_ints, compute_quadrupole_ints
from .energy_utils import nuclear_repulsion, cholesky_orthogonalization

def restricted_hartree_fock(*args, options, deriv_order=0, return_aux_data=False):
    if options['electric_field'] == 1:
        efield, geom, basis_set, nelectrons, nuclear_charges, xyz_path = args
    elif options['electric_field'] == 2:
        efield_grad, efield, geom, basis_set, nelectrons, nuclear_charges, xyz_path = args
    else:
        geom, basis_set, nelectrons, nuclear_charges, xyz_path = args

    print("Running Hartree-Fock Computation...")
    # Load keyword options
    maxit = options['maxit']
    damping = options['damping']
    damp_factor = options['damp_factor']
    spectral_shift = options['spectral_shift']
    ndocc = nelectrons // 2

    # If we are doing MP2 or CCSD after, might as well use jit-compiled JK-build, since HF will not be memory bottleneck
    if return_aux_data:
        jk_build = jax.jit(jax.vmap(jax.vmap(lambda x,y: jnp.tensordot(x, y, axes=[(0,1), (0,1)]), in_axes=(0, None)), in_axes=(0, None)))
    else: 
        jk_build = jax.vmap(jax.vmap(lambda x,y: jnp.tensordot(x, y, axes=[(0,1), (0,1)]), in_axes=(0, None)), in_axes=(0, None))

    S, T, V, G = compute_integrals(geom, basis_set, xyz_path, deriv_order, options)
    # Canonical orthogonalization via cholesky decomposition
    A = cholesky_orthogonalization(S)

    nbf = S.shape[0]

    # For slightly shifting eigenspectrum of transformed Fock for degenerate eigenvalues 
    # (JAX cannot differentiate degenerate eigenvalue eigh) 
    def form_shift():
        fudge = jnp.asarray(jnp.linspace(0, 1, nbf)) * 1.e-9
        return jnp.diag(fudge)

    shift = jax.lax.cond(spectral_shift, lambda: form_shift(), lambda: jnp.zeros_like(S))

    # Shifting eigenspectrum requires lower convergence.
    convergence = jax.lax.cond(spectral_shift, lambda: 1.0e-9, lambda: 1.0e-10)

    H = T + V
    Enuc = nuclear_repulsion(geom.reshape(-1,3), nuclear_charges)

    if options['electric_field'] == 1:
        Mu_XYZ = compute_dipole_ints(geom, basis_set, xyz_path, deriv_order, options)
        H += jnp.einsum('x,xij->ij', efield, Mu_XYZ)
    elif options['electric_field'] == 2:
        Mu_Th = compute_quadrupole_ints(geom, basis_set, xyz_path, deriv_order, options)
        H += jnp.einsum('x,xij->ij', efield, Mu_Th[:3, :, :])
        H += jnp.einsum('x,xij->ij', efield_grad[jnp.triu_indices(3)], Mu_Th[3:, :, :])
    
    def rhf_iter(F, D):
        E_scf = jnp.einsum('pq,pq->', F + H, D) + Enuc
        Fp = A.T @ F @ A
        Fp = Fp + shift 
        eps, C2 = jnp.linalg.eigh(Fp)
        C = A @ C2
        Cocc = C[:, :ndocc]
        D = Cocc @ Cocc.T
        return E_scf, D, C, eps

    def DIIS(F, D, S):
        diis_e = jnp.einsum('ij,jk,kl->il', F, D, S) - jnp.einsum('ij,jk,kl->il', S, D, F)
        diis_e = A @ diis_e @ A
        return jnp.mean(diis_e ** 2) ** 0.5

    def scf_procedure(carry):
        iter, de_, drms_, eps_, C_, D_old, D_, e_old = carry

        D_ = jax.lax.cond(damping and (iter < 10), lambda: D_old * damp_factor + D_ * damp_factor, lambda: D_)
        D_old = jnp.copy(D_)
        # Build JK matrix: 2 * J - K
        JK = 2 * jk_build(G, D_)
        JK -= jk_build(G.transpose((0,2,1,3)), D_)
        # Build Fock
        F = H + JK
        # Compute energy, transform Fock and diagonalize, get new density
        e_scf, D_, C_, eps_ = rhf_iter(F, D_)

        de_, drms_ = jax.lax.cond(iter + 1 == maxit, lambda: (1.e-15, 1.e-15), lambda: (e_old - e_scf, DIIS(F, D_, S)))

        return (iter + 1, de_, drms_, eps_, C_, D_old, D_, e_scf)

    # Create Guess Density
    D = jnp.copy(H)
    JK = 2 * jk_build(G, D)
    JK -= jk_build(G.transpose((0,2,1,3)), D)
    F = H + JK
    E_init, D_init, C_init, eps_init = rhf_iter(F, D)

    # Perform SCF Procedure
    iteration, _, _, eps, C, _, D, E_scf = jax.lax.while_loop(lambda arr: (abs(arr[1]) > convergence) | (arr[2] > convergence),
                                                              scf_procedure, (0, 1.0, 1.0, eps_init, C_init, D, D_init, E_init))
                                                              # (iter, dE, dRMS, eps, C, D_old, D, E_scf)
    print(iteration, " RHF iterations performed")

    if options['electric_field'] > 0:
        E_scf += jnp.einsum('x,q,qx->', efield, nuclear_charges, geom.reshape(-1,3))
    if options['electric_field'] > 1:
        E_scf += jnp.einsum('ab,q,qa,qb->', jnp.triu(efield_grad), nuclear_charges, geom.reshape(-1,3), geom.reshape(-1,3))

    # If many orbitals are degenerate, warn that higher order derivatives may be unstable 
    tmp = jnp.round(eps, 6)
    ndegen_orbs =  tmp.shape[0] - jnp.unique(tmp).shape[0] 
    if (ndegen_orbs / nbf) > 0.20:
        print("Hartree-Fock warning: More than 20% of orbitals have degeneracies. Higher order derivatives may be unstable due to eigendecomposition AD rule")

    if not return_aux_data:
        return E_scf
    else:
        # print("RHF Energy:                ", E_scf)
        return E_scf, C, eps, G

