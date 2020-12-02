from . import tei
from . import oei
from . import libint_interface

from .tei import tei
#from .new_tei import tei
from .oei import overlap
from .oei import kinetic
from .oei import potential

from .tmp_potential import tmp_potential

def libint_initialize(xyz_path, basis_name, max_deriv_order=0):
    libint_interface.initialize(xyz_path, basis_name) 
    if max_deriv_order:
        libint_interface.eri_deriv_disk(max_deriv_order)
        #print("Finished precomputing ERI derivative tensors and saving to disk for up to order {}".format(max_deriv_order))
    return 0

def libint_finalize():
    libint_interface.finalize()
    return 0

