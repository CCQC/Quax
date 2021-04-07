# Quax: Quantum Chemistry, powered by JAX
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
Our integrals code is _slow_. Using the Libint interface is highly recommended. However, compiling Libint for support for very high order
derivatives (5th, 6th) takes a very long time and causes the library size to be very large (sometimes so large it's uncompilable), so using the Quax integrals
is the best bet at this time.
We will incrementally roll out improvements which allow user specification for how to handle higher-order integral derivatives.
For example, control over when to use disk vs core memory, and whether Libint or Quax integral derivatives are computed.
In principle, the Quax integrals code could also be improved.
Contributions and suggestions are welcome.

Also, we do not recommend computing derivatives of systems with many degenerate orbitals.
The reason for this is because automatically differentiating through eigendecomposition involves denominators of eigenvalue differences, which blow up in the degenerate case.  
We cheat our way around this by shifting the eigenspectrum to lift the degeneracy, but this only works for systems with moderate degeneracy.
Workarounds for this are coming soon.

# Installation Instructions

### Anaconda Environment installation instructions
To use Quax, only a few dependencies are needed. We recommend using a clean Anaconda environment: 
```
conda create -n quax python=3.7
conda activate quax
conda install -c psi4 psi4
python setup.py install
```

This is sufficient to use Quax without the Libint interface.

### Building the Libint Interface
If you plan to use the Libint interface (highly recommnded), you can install those dependencies as well.
```
conda install libstdcxx-ng
conda install gcc_linux-64
conda install gxx_linux-64
conda install ninja
conda install boost
conda install eigen3
conda install gmp
conda install bzip2
conda install cmake
conda install pybind11
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
 ../configure --prefix=/home/adabbott/Git/libint/libint/build/PREFIX --with-max-am=2 --with-opt-am=0 --enable-1body=4 --enable-eri=4 --with-multipole-max-order=0 --enable-eri3=no --enable-eri2=no --enable-g12=no --enable-g12dkh=no --with-pic --enable-static --enable-single-evaltype --enable-generic-code --disable-unrolling

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
Once Libint is installed, the makefile in `quax/external_integrals/makefile` needs to be edited with your compiler and the proper paths specifying the locations
of headers and libraries for Libint, pybind11, HDF5, and python. 

The `LIBINT_PREFIX` path in the makefile is wherever you installed the headers and the static library `lib/libint2.a`. 
All of the required headers and libraries should be discoverable in the Anaconda environment's include and lib paths.
After editing the paths appropriately and setting the CC compiler to `x86_64-conda_cos6-linux-gnu-gcc`, or 
if you have a nice modern compiler available, use that.

Running `make` in the directory `quax/external_integrals/` to compile the Libint interface.


<!---
The library requires several dependencies, most of which are taken care of with `setup.py`.
To install, clone this repository, and run 
```pip install .```
If you plan to develop/edit/play around with the code,
install with `pip install -e .` so that the installation will automatically update when changes are made.
This takes care of the following dependencies, according to the contents of `setup.py`.
```
numpy
jax
jaxlib
h5py
```

In addition to the dependencies in `setup.py`, this library requires an installation of Psi4.
The easiest way to install psi4 is with Anaconda:
`conda install -c psi4 psi4`
If you do not want to use Anaconda, you can install Psi4 from source (much more difficult).
These installation options (Psi4, and the dependencies in `setup.py`) are sufficient
for computing derivatives of electronic structure methods.

### Integral Derivative Computation
A primary bottleneck of the code is the computation of nuclear derivatives of one and two electron integrals over Gaussian basis functions.
We feature a very simple integral code built using entirely JAX utilities in the `integrals/oei.py` and `integrals/tei.py`. 
This code works for arbitrary angular momentum and arbitary order derivatives, however it is quite slow and has high memory usage
due to the overhead associated with JIT compilation and the derivative code generation which occurs every time the program is run.

To avoid that performance issue, simply use the library with [Libint](https://github.com/evaleev/libint) (**strongly** recommended).
Note that Libint needs to be configured for the order of differentation and maximum angular momentum
you wish to support. By default, higher order derivatives of one and two electron integrals are not configured,
they have to be specifically requested, e.g. for fourth derivatives, 
it must be compiled with configure flags `--enable-1body=4 --enable-eri=4`. See the Libint installation instructions for details.
Depending on these configuration options, the generation of a Libint library and subsequent compilation 
can take a few days or even over a week. A preconfigured tarball which supports up to f functions and
fourth order derivatives will be made available by some means in the future. 

For building with Libint, more dependencies are introduced, some of which are needed for Libint, and others
are needed for the Libint interface for this software. I strongly recommend dumping everything
into a clean Anaconda environment.
To generate a clean conda environment for running the code,
```
conda create -n psijax python=3.6
conda activate psijax 
conda install -c psi4 psi4
```

Then install the dependencies needed for the Libint interface:
```
conda install -c conda-forge pybind11
conda install -c omnia eigen3
conda install hdf5
conda install gmp
conda install bzip2
conda install boost
conda install cmake
conda install libstdcxx-ng
conda install -c conda-forge libcxx
```


NEW  have to install
```
conda create -n jax python=3.6
conda activate jax
conda install ninja
conda install -c omnia eigen3
conda install gcc
conda install -c conda-forge pybind11
conda install gcc_linux-64  ###THIS ONE
conda install boost
```

Libint's gmp issues can be taken care of by installing `conda install gcc_linux-64`
Also need `conda install gxx_linux-64` 


### Building the Libint Interface

The default gcc version 4.8 that comes with `conda install gcc` is not recent enough to successfully compile the Quax-Libint interface.
You must instead use a more modern compiler. To do this in anaconda, we need to use
`x86_64-conda_cos6-linux-gnu-gcc` as our compiler instead of gcc.
This is available by installing `gcc_linux-64` and `gxx_linux-64`.
Feel free to try other more 
Thus a complete anaconda envrionment, containing everything you need to run the code and compile the Libint interface,
would include:

```
conda create -n quax python=3.7
conda activate quax 
conda install -c psi4 psi4
conda install gcc_linux-64
conda install gxx_linux-64
conda install ninja
conda install boost
conda install eigen3
conda install gmp
conda install bzip2
conda install cmake
conda install pybind11

pip install jax
pip install jaxlib
conda install h5py
```

These are sufficient to compile the Libint interface.
Head over to `external_integrals/` directory and edit the makefile with the appropriate paths.
All of the required headers and libraries should be discoverable in the Anaconda environment's include and lib paths.
After editing the paths appropriately and setting the CC compiler to `x86_64-conda_cos6-linux-gnu-gcc`, or 
if you have a nice modern compiler available, use that.

Libint's gmp issues can be taken care of by installing `conda install gcc_linux-64`
Also need `conda install gxx_linux-64` 


Now, given a Libint tarball which supports the desired maximum angular momentum and derivative order,
we need to unpack the library, `cd` into it, and `mkdir PREFIX` where the headers and static library will be stored.
Then it is built and compiled. The position independent code flag is required for Libint to play nice with pybind11.
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


### Installing Libint in a clean conda environment
Note that the cmake command may not find various libraries for the dependencies of Libint.
`cmake . -DCMAKE_INSTALL_PREFIX=/path/to/libint/PREFIX/ -DCMAKE_POSITION_INDEPENDENT_CODE=ON`
To fix this, you may need to explicitly point to it
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/vulcan/adabbott/.conda/envs/quax/lib/`
and then run the above cmake command.

Also note that Libint recommends using Ninja to build for performance reasons. This can be done if Ninja is installed:
`cmake . -G Ninja -DCMAKE_INSTALL_PREFIX=/path/to/libint/PREFIX/ -DCMAKE_POSITION_INDEPENDENT_CODE=ON`

Once Libint is installed, the makefile in `external_integrals/makefile` needs to be edited to the proper paths specifying the locations
of headers and libraries for Libint, pybind11, HDF5, and python. Then run `make` to compile the Libint interface.

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


