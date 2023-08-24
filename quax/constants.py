import os
import re

# Get absolute module path
module_path = os.path.dirname(os.path.abspath(__file__))

# Check if libint interface is being used
libint_imported = False
lib = re.compile("libint_interface\.cpython.+")
for path in os.listdir(module_path + "/external_integrals"):
    if lib.match(path):
        from . import external_integrals 
        libint_imported = True

if libint_imported:
    print("Using Libint integrals...")
else:
    print("Using Quax integrals...")
