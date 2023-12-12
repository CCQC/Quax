import psi4
import jax
import jax.numpy as jnp
from jax.lax import fori_loop

from ..methods.ints import compute_f12_oeints
from ..methods.energy_utils import symmetric_orthogonalization

def build_RIBS(molecule, basis_set, cabs_name):
    """
    Builds basis set for
    CABS procedure
    """

    # Libint uses the suffix 'cabs' but Psi4 uses 'optri'
    basis_name = basis_set.name()
    try:
        psi4_name = cabs_name.lower().replace('cabs', 'optri')
    except:
        raise Exception("Must use a cc-pVXZ-F12 or aug-cc-pVXZ basis set for F12 methods.")

    keys = ["BASIS","CABS_BASIS"]
    targets = [basis_name, psi4_name]
    roles = ["ORBITAL","F12"]
    others = [basis_name, basis_name]

    # Creates combined basis set in Python
    ao_union = psi4.driver.qcdb.libmintsbasisset.BasisSet.pyconstruct_combined(molecule.save_string_xyz(), keys, targets, roles, others)
    ao_union['name'] = cabs_name
    ribs_set = psi4.core.BasisSet.construct_from_pydict(molecule, ao_union, 0)

    return ribs_set

def build_CABS(geom, basis_set, cabs_set, xyz_path, deriv_order, options):
    """
    Builds and returns 
    CABS transformation matrix
    """
    # Make Thread Safe
    threads = psi4.get_num_threads()
    psi4.set_num_threads(1)

    # Orthogonalize combined basis set
    S_ao_ribs_ribs = compute_f12_oeints(geom, cabs_set, cabs_set, xyz_path, deriv_order, options, True)
    C_ribs = symmetric_orthogonalization(S_ao_ribs_ribs, 1.0e-8)

    # Compute the overlap matrix between OBS and RIBS, then orthogonalizes the RIBS
    S_ao_obs_ribs = compute_f12_oeints(geom, basis_set, cabs_set, xyz_path, deriv_order, options, True)

    # Compute the eigenvectors and eigenvalues of C2.T @ S12.T @ S12 @ C2
    S22 = jnp.dot(S_ao_obs_ribs.T, S_ao_obs_ribs)
    CTC = C_ribs.T @ S22 @ C_ribs
    S2, V = jnp.linalg.eigh(CTC)

    def loop_zero_vals(idx, count):
        count += jax.lax.cond(abs(S2[idx]) < 1.0e-6, lambda: 1, lambda: 0)
        return count
    ncabs = jax.lax.fori_loop(0, S2.shape[0], loop_zero_vals, 0)

    V_N = V.at[:, :ncabs].get()

    C_cabs = jnp.dot(C_ribs, V_N)

    psi4.set_num_threads(threads)

    return C_cabs