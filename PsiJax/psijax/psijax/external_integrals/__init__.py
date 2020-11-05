from . import tei
from . import oei
from . import libint_interface

from .tei import tei
from .oei import overlap
from .oei import kinetic
from .oei import potential

def libint_initialize(xyz_path, basis_name):
    libint_interface.initialize(xyz_path, basis_name) 
    return 0

def libint_finalize():
    libint_interface.finalize()
    return 0
