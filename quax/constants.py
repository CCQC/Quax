import os
import re
import sys

# Get absolute module path
module_path = os.path.dirname(os.path.abspath(__file__))

# Check if libint interface is found
libint_imported = False
lib = re.compile("libint_interface\.cpython.+")
for path in os.listdir(module_path + "/integrals"):
    if lib.match(path):
        from . import integrals
        libint_imported = True

if not libint_imported:
    sys.exit("Libint is a required dependency!")
