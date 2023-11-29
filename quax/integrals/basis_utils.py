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
    C12 = jnp.dot(S_ao_obs_ribs, C_ribs)

    nN, Vt = null_svd(C12)

    V_N = jnp.transpose(Vt[nN:, :])
    C_cabs = jnp.dot(C_ribs, V_N)

    psi4.set_num_threads(threads)

    return C_cabs

@jax.custom_jvp
def null_svd(C12, cutoff = 1.0e-6):
    """
    Grabs the null vectors from the V matrix
    of an SVD procedure and returns the 
    number of null vecs and the null vec matrix
    """
    # Compute the eigenvectors and eigenvalues of C12.T @ C12
    _, S, Vt = jnp.linalg.svd(C12)

    # Collect the eigenvectors that are associated with (near) zero eignevalues
    def loop_zero_vals(idx, count):
        count += jax.lax.cond(abs(S[idx]) < cutoff, lambda: 1, lambda: 0)
        return count
    nN = fori_loop(0, S.shape[0], loop_zero_vals, S.shape[0])

    return nN, Vt

@null_svd.defjvp
def null_svd_jvp(primals, tangents):
  C12, cutoff = primals
  C12_dot, cutoff_dot = tangents
  primal_out = null_svd(C12, cutoff)
  tangent_out = null_svd(C12_dot, cutoff)
  return primal_out, tangent_out
