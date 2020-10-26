#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdlib.h>
#include <libint2.hpp>

// Saw this on github.. maybe it works
void libint2start() {
  libint2::initialize();
}

void libint2stop() {
  libint2::finalize();
}

namespace py = pybind11;

// You can code all functions directly into here. Example:
int add(int i, int j) {
    return i + j;
}

// Function takes two strings: xyzfile absolute path (TODO in Angstroms) and basis set name  
// Then computes overlap integrals
//py::array_t<double> test(std::string xyzfilename, std::string basis_name) {
py::array overlap(std::string xyzfilename, std::string basis_name) {
    using namespace libint2;
    using namespace std;
    libint2::initialize();

    // Load basis set and geometry. TODO this assumes units are angstroms... 
    ifstream input_file(xyzfilename);
    vector<Atom> atoms = read_dotxyz(input_file);
    BasisSet obs(basis_name, atoms);
    obs.set_pure(false); // use cartesian gaussians


    //Engine eri_engine(Operator::coulomb, obs.max_nprim(), obs.max_l());
    // Overlap integrals engine 
    Engine s_engine(Operator::overlap,obs.max_nprim(),obs.max_l());

    size_t length = obs.size() * obs.size();
    std::vector<double> result(length);

    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
    const auto& buf_vec = s_engine.results(); // will point to computed shell sets
    
    for(auto s1=0; s1!=obs.size(); ++s1) {
      for(auto s2=0; s2!=obs.size(); ++s2) {
    
        //cout << "compute shell set {" << s1 << "," << s2 << "} ... ";
        s_engine.compute(obs[s1], obs[s2]);
        //cout << "done" << endl;
        auto ints_shellset = buf_vec[0];  // location of the computed integrals
        if (ints_shellset == nullptr)
          continue;  // nullptr returned if the entire shell-set was screened out
    
        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        auto bf2 = shell2bf[s2];  // first basis function in second shell
        auto n2 = obs[s2].size(); // number of basis functions in second shell
    
        // integrals are packed into ints_shellset in row-major (C) form
        // this iterates over integrals in this order
        // TODO how to pack ints_shellset into final vector representing entire array?
        for(auto f1=0; f1!=n1; ++f1)
          for(auto f2=0; f2!=n2; ++f2)
            //result[f1 * n2 + f2]; // dummy add
            //result[f1*n2+f2] = ints_shellset[f1 * n2 + f2];
            // true idx is probably  s1 * n2 + s2 + f1 * n2 + f2
            result[s1 * n2 + s2 + f1 * n2 + f2] = ints_shellset[f1 * n2 + f2];
            //cout << "  " << bf1+f1 << " " << bf2+f2 << " " << ints_shellset[f1*n2+f2] << endl;
      }
    }
    libint2::finalize();

    return py::array(result.size(), result.data()); // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042
    //return 0;
}

// Define module named 'libint_interface' which can be imported with python
// The second arg, 'm' defines a variable py::module_ which can be used to create
// bindings. the def() methods generates binding code that exposes new functions to Python.
PYBIND11_MODULE(libint_interface, m) {
    m.doc() = "pybind11 libint interface to molecular integrals"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers");
    m.def("overlap", &overlap, "Computes overlap integrals with libint");
}

// Temporary libint reference: new shared library compilation
// currently needs export LD_LIBRARY_PATH=/path/to/libint2.so
// Compilation script for libint with am=2 deriv=4, may need to set LD_LIBRARY_PATH = /path/to/libint2.so corresponding to this installation.
// g++ libint_interface.cc -o libint_interface`python3-config --extension-suffix`  -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ -lint2 

// Warning: above is very slow since its a huge copy of libint. can use smaller version, just s, p with gradients,
// Can do quick compile with the following:
//g++ -c libint_interface.cc -o libint_interface.o -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include/libint2 -lint2 -L/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/lib

//g++ libint_interface.o -o libint_interface`python3-config --extension-suffix` -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include/libint2 -lint2 -L/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/lib


//g++ -c libint_interface.cc -o libint_interface.o -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ -lint2 

//g++ libint_interface.o -o libint_interface`python3-config --extension-suffix`  -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6

// TODO let's try to simplify the above... can we combine into one compile?
//g++ -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ -lint2 libint_interface.cc -o libint_interface`python3-config --extension-suffix` 

// Okay it fails with the follwoing:
// /home/adabbott/Git/PsiTorch/PsiJax/psijax/psijax/external_integrals/libint_interface.cpython-36m-x86_64-linux-gnu.so: undefined symbol: libint2_build_default

// Does it work if i cll script first?
//g++ libint_interface.cc -o libint_interface`python3-config --extension-suffix`  -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ -lint2 

// Yes the above works just fine... why do things work fine when I put the .cc first?

//g++ libint_interface.cc -o libint_interface`python3-config --extension-suffix`  -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 


// Then two stage compile:

//g++ -c libint_interface.cc -o libint_interface.o -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include/libint2 -lint2 -L/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/lib

//g++ libint_interface.o -o libint_interface`python3-config --extension-suffix` -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include/libint2 -lint2 -L/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/lib


// After compilation, should be able to access this funciton:
// >>> import psijax
// >>> psijax.external_integrals.libint_interface.add(1,2)
// 3
// >>> psijax.external_integrals.libint_interface.test(1,2)
// 0


// TODO Define functions, figure out args stuff, e.g. py::array_t<double>
// also define in another .cc file containing these routines
// TODO figure out how to handle BasisSet objects here. 
//void compute_tei()
//void compute_tei_deriv()
//void compute_overlap()
//void compute_overlap_deriv()
//void compute_kinetic()
//void compute_kinetic_deriv()
//void compute_potential()
//void compute_potential_deriv()

//PYBIND11_PLUGIN(libint_tei) {
//    py::module m("libint_interface", "pybind11 interface to libint molecule integrals and their derivatives")
//    m.def("compute_tei", &compute_tei, "Compute two-electron integral array, shape (nbf,nbf,nbf,nbf)")
//    m.def("compute_tei_deriv", &compute_tei, "Compute partial derivative of two-electron integral array, shape (nbf,nbf,nbf,nbf)")
//    m.def("compute_overlap", &compute_overlap, "Compute overlap integral array, shape (nbf,nbf)")
//    m.def("compute_overlap_deriv", &compute_overlap, "Compute (nbf,nbf,nbf,nbf) nuclear partial derivative of two-electron integral array")
//    m.def("compute_kinetic", &compute_kinetic, "Compute (nbf,nbf,nbf,nbf) two-electron integral array")
//    m.def("compute_kinetic_deriv", &compute_kinetic, "Compute (nbf,nbf) nuclear partial derivative of two-electron integral array")
//
//    return m.ptr();
//}
