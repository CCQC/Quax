import psi4 
import jax.numpy as jnp
import numpy as np

def build_CABS(molecule, basis_name, cabs_name):
    """
    Builds and returns CABS
    Provide molecule from Psi4,
    OBS name, CABS name, and
    MO coefficients from RHF
    """
    cabs_name = cabs_name.lower().replace('cabs', 'optri')

    keys = ["BASIS","CABS_BASIS"]
    targets = [basis_name, cabs_name]
    roles = ["ORBITAL","F12"]
    others = [basis_name, basis_name]

    # Creates combined basis set in Python
    obs = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
    ao_union = psi4.driver.qcdb.libmintsbasisset.BasisSet.pyconstruct_combined(molecule.save_string_xyz(), keys, targets, roles, others)
    ao_union = psi4.core.BasisSet.construct_from_pydict(molecule, ao_union, 0)
    ri_space = psi4.core.OrbitalSpace.build_ri_space(ao_union, 1.0e-8)

    C_ribs = np.array(ri_space.C()) # Orthogonalizes the AOs of the RI space

    # Compute the overlap matrix between OBS and RIBS, then orthogonalizes the RIBS
    mints = psi4.core.MintsHelper(obs)
    S_ao_obs_ribs = np.array(mints.ao_overlap(obs, ri_space.basisset()))
    C12 = np.einsum('Pq,qQ->PQ', S_ao_obs_ribs, C_ribs)

    # Compute the eigenvectors and eigenvalues of S12.T * S12
    _, S, Vt = np.linalg.svd(C12)

    # Collect the eigenvectors that are associated with (near) zero eignevalues
    ncabs = S.shape[0]
    for eval_i in S:
        if abs(eval_i) < 1.0e-6: ncabs += 1
    V_N = Vt[ncabs:, :].T

    # Make sure the CABS is an orthonormal set
    C_cabs = np.einsum('pQ,QP->pP', C_ribs, V_N)

    return C_cabs
