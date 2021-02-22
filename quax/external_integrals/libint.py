from . import libint_interface

def libint_initialize(xyz_path, basis_name, max_deriv_order=0):
    libint_interface.initialize(xyz_path, basis_name) 
    if max_deriv_order:
        libint_interface.oei_deriv_disk(max_deriv_order)
        libint_interface.eri_deriv_disk(max_deriv_order)
    return 0

def libint_finalize():
    libint_interface.finalize()
    return 0

