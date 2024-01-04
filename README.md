# Quax: Quantum Chemistry, powered by JAX
![Continuous Integration](https://github.com/CCQC/Quax/actions/workflows/continuous_integration.yml/badge.svg)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

![Screenshot](quax.png)

You have found Quax. The paper outlining this work was just [recently published](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.1c00607). 
This library supports a simple and clean API for obtaining higher-order energy derivatives of electronic
structure computations such as Hartree-Fock, second-order Møller-Plesset perturbation theory (MP2), and
coupled cluster with singles, doubles, and perturbative triples excitations [CCSD(T)].
Whereas most codes support only analytic gradient and occasionally Hessian computations,
this code can compute analytic derivatives of arbitrary order. 
We use [JAX](https://github.com/google/jax) for automatically differentiating electronic structure computations.
The code can be easily extended to support other methods, for example
using the guidance offered by the [Psi4Numpy project](https://github.com/psi4/psi4numpy).

If you are interested in obtaining electronic energy derivatives with Quax,
but are wary of and/or not familiar with the concept of automatic differentiation, 
we recommend [this video](https://www.youtube.com/watch?v=wG_nF1awSSY) for a brief primer.

We should also note this project is mostly intended as an experimental proof-of-concept. 
While it can be used for research applications, users should always take steps to verify the accuracy of the results,
either by checking energies and derivatives against standard electronic structure codes or by using finite differences.
Generally, if the energy and gradient are correct, higher order derivatives are most likely correct to a high degree of numerical precision.
Note however the caveat (described below) that systems with highly degenerate orbitals will likely be numerically unstable at high derivative orders.

### Using Quax
The Quax API is very simple. We use Psi4 to handle molecule data like coordinates, charge, multiplicity, and basis set
information. Once a Psi4 Molecule object is defined, energies, derivatives, and partial derivatives can be computed with a single line of code.
In the following example, we perform Hartree-Fock computations with a sto-3g basis set: we compute the energy, gradient, Hessian, and single elements
of the gradient and Hessian:

```python
import quax
import psi4

molecule = psi4.geometry("""
                         0 1
                         O 0.0 0.0 0.0
                         H 0.0 0.0 1.0
                         H 0.0 1.0 0.0
                         units bohr 
                         """)

energy = quax.core.energy(molecule, 'sto-3g', 'hf')
print(energy)
gradient = quax.core.derivative(molecule, 'sto-3g', 'hf', deriv_order=1)
print(gradient)
hessian = quax.core.derivative(molecule, 'sto-3g', 'hf', deriv_order=2)
print(hessian)

dz1 = quax.core.partial_derivative(molecule, 'sto-3g', 'hf', deriv_order=1, partial=(2,))
print(dz1)

dz1_dz2 = quax.core.partial_derivative(molecule, 'sto-3g', 'hf', deriv_order=2, partial=(2,5))
print(dz1_dz2)

print('Partial gradient matches gradient element: ', dz1 == gradient[2])
print('Partial hessian matches hessian element: ', dz1_dz2 == hessian[2,5])
```

Above, in the `quax.core.partial_derivative` function calls, the `partial` arguments describe the address of the element in the _n_th order derivative
tensor you want to compute. The dimensions of a derivative tensor correspond to the row-wise flattened Cartesian coordinates, with 0-based indexing.
For _N_ Cartesian coordinates, gradient is a size _N_ vector, Hessian a _N_ by _N_ matrix, and cubic and quartic derivative tensors are rank-3 and rank-4 tensors with dimension size _N_.

Speaking of which, the Quax API currently supports up to 4th-order full-derivatives of energy methods, and up to 6th-order partial derivatives.
A full quartic derivative tensor at CCSD(T) can be computed like so: 

```python
import quax 
import psi4

molecule = psi4.geometry('''
                         0 1
                         H 0.0 0.0 -0.80000000000
                         H 0.0 0.0  0.80000000000
                         units bohr
                         ''')

quartic = quax.core.derivative(molecule, '6-31g', 'ccsd(t)', deriv_order=4)
```

Perhaps that's too expensive/slow. You can instead compute quartic partial derivatives:

```python
import quax 
import psi4

molecule = psi4.geometry('''
                         0 1
                         H 0.0 0.0 -0.80000000000
                         H 0.0 0.0  0.80000000000
                         units bohr
                         ''')

dz1_dz1_dz2_dz2 = quax.core.partial_derivative(molecule, '6-31g', 'ccsd(t)', deriv_order=4, partial=(2,2,5,5))
```

Similar computations can be split across multiple nodes in an embarassingly parallel fashion, and one can take full advantage of symmetry so that only the unique elements are computed.
The full quartic derivative tensor can then be constructed with the results.

It's important to note that full derivative tensor computations may easily run into memory issues. 
For example, the two-electron integrals fourth derivative tensor used in the above computation
for _n_ basis functions and _N_ cartesian coordinates at derivative order _k_ contains _n_<sup>4</sup> * _N_<sup>k</sup> double precision floating point numbers, which requires a great deal of memory. 
Not only that, but the regular two-electron integrals, and the first, second, and third-order derivative tensors are also held in memory.
The above computation therefore, from having 4 basis functions, stores 5 arrays associated with the two-electron integrals at run time:
each of shapes (4,4,4,4), (4,4,4,4,6), (4,4,4,4,6,6), (4,4,4,4,6,6,6), (4,4,4,4,6,6,6,6).
These issues also arise in the simulataneous storage of the old and new T1 and T2 amplitudes during coupled cluster iterations.
Obviously, for large basis sets and molecules, these arrays get very big very fast.
Unless you have impressive computing resources, partial derivatives are recommended for higher order derivatives.

### Caveats
The Libint interface is a necessary dependency for Quax. However, compiling Libint for support for very high order
derivatives (5th, 6th) takes a very long time and causes the library size to be very large (sometimes so large it's uncompilable).
We will incrementally roll out improvements which allow user specification for how to handle higher-order integral derivatives.
Contributions and suggestions are welcome.

Also, we do not recommend computing derivatives of systems with many degenerate orbitals.
The reason for this is because automatically differentiating through eigendecomposition involves denominators of eigenvalue differences, which blow up in the degenerate case.  
We cheat our way around this by shifting the eigenspectrum to lift the degeneracy, but this only works for systems with moderate degeneracy.
Workarounds for this are coming soon.

# Installation Instructions

### Anaconda Environment installation instructions
To use Quax, only a few dependencies are needed. We recommend using a clean Anaconda environment: 
```
conda create -n quax python=3.10
conda activate quax
conda install psi4 python=3.10 -c conda-forge/label/libint_dev -c conda-forge
python setup.py install
```

### Building the Libint Interface
For the Libint interface, you nust install those dependencies as well.
```
conda install libstdcxx-ng gcc_linux-64 gxx_linux-64 ninja boost eigen3 gmp bzip2 cmake pybind11
```

We note here that the default gcc version (4.8) that comes with `conda install gcc` is not recent enough to successfully compile the Quax-Libint interface.
You must instead use a more modern compiler. To do this in Anaconda, we need to use
`x86_64-conda_cos6-linux-gnu-gcc` as our compiler instead of gcc.
This is available by installing `gcc_linux-64` and `gxx_linux-64`.
Feel free to try other more advanced compilers. gcc >= 7.0 appears to work great. 

### Building Libint
Libint can be built to support specific maximum angular momentum, different types of integrals, and certain derivative orders.
The following is a build procedure supports up to _d_ functions and 4th-order derivatives. For more details,
see the [Libint](https://github.com/evaleev/libint) repo.
Note this build takes quite a long time! (on the order of hours to a couple days)
In the future we will look into supplying pre-built libint tarballs by some means.

```
git clone https://github.com/evaleev/libint.git
cd libint
./autogen.sh
mkdir BUILD
cd BUILD
mkdir PREFIX
 ../configure --prefix=/path/to/libint/build/PREFIX --with-max-am=2 --with-opt-am=0 --enable-1body=4 --enable-eri=4 --with-multipole-max-order=0 --enable-eri3=no --enable-eri2=no --enable-g12=no --enable-g12dkh=no --with-pic --enable-static --enable-single-evaltype --enable-generic-code --disable-unrolling

make export
```

The above will produce a file of the form `libint-*.tgz`, containing your custom Libint library that needs to be compiled.

### Compiling Libint
Now, given a Libint tarball which supports the desired maximum angular momentum and derivative order,
we need to unpack the library, `cd` into it, and `mkdir PREFIX` where the headers and static library will be stored.
The position independent code flag is required for Libint to play nice with pybind11.
The `-j4` flag instructs how many processors to use in compilation, and can be adjusted according to your system. The `--target check` runs the Libint test suite; it is not required.
The --target check runs test suite, and finally the install command installs the headers and static library into the PREFIX directory.
```
tar -xvf libint_*.tgz
cd libint-*/
mkdir PREFIX
cmake . -DCMAKE_INSTALL_PREFIX=/path/to/libint/PREFIX/ -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build . -- -j4
cmake --build . --target check
cmake --build . --target install
```

Note that the following cmake command may not find various libraries for the dependencies of Libint.
`cmake . -DCMAKE_INSTALL_PREFIX=/path/to/libint/PREFIX/ -DCMAKE_POSITION_INDEPENDENT_CODE=ON`
To fix this, you may need to explicitly point to it
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libint/dependency/lib/`
and then run the above cmake command.
If using Anaconda, the path is probably in the environment directory `/path/to/envs/quax/lib/`.

Also note that Libint recommends using Ninja to build for performance reasons. This can be done if Ninja is installed:
`cmake . -G Ninja -DCMAKE_INSTALL_PREFIX=/path/to/libint/PREFIX/ -DCMAKE_POSITION_INDEPENDENT_CODE=ON`

### Compiling the Libint-Quax interface
Once Libint is installed, the makefile in `quax/integrals/makefile` needs to be edited with your compiler and the proper paths specifying the locations
of headers and libraries for Libint, pybind11, HDF5, and python. 

The `LIBINT_PREFIX` path in the makefile is wherever you installed the headers and the static library `lib/libint2.a`. 
All of the required headers and libraries should be discoverable in the Anaconda environment's include and lib paths.
After editing the paths appropriately and setting the CC compiler to `x86_64-conda_cos6-linux-gnu-gcc`, or 
if you have a nice modern compiler available, use that.

Running `make` in the directory `quax/integrals/` to compile the Libint interface.

### Citing Quax
If you use Quax in your research, we would appreciate a citation:
```
@article{abbott2021,
  title={Arbitrary-Order Derivatives of Quantum Chemical Methods via Automatic Differentiation},
  author={Abbott, Adam S and Abbott, Boyi Z and Turney, Justin M and Schaefer III, Henry F},
  journal={The Journal of Physical Chemistry Letters},
  volume={12},
  pages={3232--3239},
  year={2021},
  publisher={ACS Publications}
}
```
We also kindly request you give credit to the projects which make up the dependencies of Quax.
