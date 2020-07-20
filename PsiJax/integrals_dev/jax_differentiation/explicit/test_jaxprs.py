import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product
np.set_printoptions(linewidth=800, suppress=True)
from pprint import pprint
from oei_s import * 
from oei_p import * 
from oei_d import * 
from oei_f import * 

# Approximate complexity by number of lines in pretty-printed jaxprs
args = (1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0)
jaxpr = jax.make_jaxpr(overlap_ss)(*args)
print('SS', len(str(jaxpr).splitlines()))

jaxpr = jax.make_jaxpr(overlap_ps)(*args)
print('PS', len(str(jaxpr).splitlines()))

jaxpr = jax.make_jaxpr(overlap_pp)(*args)
print('PP', len(str(jaxpr).splitlines()))

jaxpr = jax.make_jaxpr(overlap_ds)(*args)
print('DS', len(str(jaxpr).splitlines()))

jaxpr = jax.make_jaxpr(overlap_dp)(*args)
print('DP', len(str(jaxpr).splitlines()))

jaxpr = jax.make_jaxpr(overlap_dd)(*args)
print('DD', len(str(jaxpr).splitlines()))

# Orignal
#SS 44
#PS 103
#PP 198
#DS 249
#DP 486
#DP 1250

# Switching the jacfwd in p functions
# SS 44
# PS 90
# PP 153
# DS 200
# DP 305
# DD 718

# Removing array creation, expanding dot product
#SS 26
#PS 78
#PP 170
#DS 203
#DP 485
#DD 1387

