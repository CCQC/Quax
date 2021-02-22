import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
#from integrals_utils import cartesian_product, old_cartesian_product
np.set_printoptions(linewidth=800, suppress=True)
from pprint import pprint
from oei_s import * 
from oei_p import * 
from oei_d import * 
from oei_f import * 

# Define molecule
molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         H 0.0 0.0 -0.249220457955
                         H 0.0 0.0  0.249220457955
                         units bohr
                         """)

# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))
# Get Psi Basis Set and basis set dictionary objects
basis_name = 'cc-pvdz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
# Number of basis functions, number of shells
nbf = basis_set.nbf()
print("number of basis functions", nbf)
nshells = len(basis_dict)

overlap_funcs = {}
overlap_funcs['00'] = jax.vmap(overlap_ss, (None,None,None,None,None,None,0,0,0,0))
overlap_funcs['10'] = jax.vmap(overlap_ps, (None,None,None,None,None,None,0,0,0,0))
overlap_funcs['11'] = jax.vmap(overlap_pp, (None,None,None,None,None,None,0,0,0,0))
overlap_funcs['20'] = jax.vmap(overlap_ds, (None,None,None,None,None,None,0,0,0,0))
overlap_funcs['21'] = jax.vmap(overlap_dp, (None,None,None,None,None,None,0,0,0,0))
overlap_funcs['22'] = jax.vmap(overlap_dd, (None,None,None,None,None,None,0,0,0,0))



# Maps over primitive computations (s|s) to (d|d)
def mapper(Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2, which):
    args = (Ax, Ay, Az, Cx, Cy, Cz, alpha_bra, alpha_ket, c1, c2)
    val = np.where(which == 0, np.pad(overlap_ss(*args).reshape(-1), (0,36)),
          np.where(which == 1, np.pad(overlap_ps(*args).reshape(-1), (0,33)),
          np.where(which == 2, np.pad(overlap_ds(*args).reshape(-1), (0,30)),
          np.where(which == 3, np.pad(overlap_pp(*args).reshape(-1), (0,27)),
          np.where(which == 4, np.pad(overlap_dp(*args).reshape(-1), (0,18)),
          np.where(which == 5, overlap_dd(*args).reshape(-1), np.zeros(36)))))))

dope = jax.vmap(mapper, (0,0,0,0,0,0,0,0,0,0,0))

dummy = np.arange(10)
which = np.array([0,0,1,1,2,0,0,1,1,2])
print(dope(dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, which))

