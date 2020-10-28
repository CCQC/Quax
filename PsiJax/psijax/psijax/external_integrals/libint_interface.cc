#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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

// Function takes two strings: xyzfile absolute path (TODO in Angstroms) and basis set name  
py::array overlap(std::string xyzfilename, std::string basis_name) {
    using namespace libint2;
    using namespace std;
    libint2::initialize();

    // Load basis set and geometry. TODO this assumes units are angstroms... 
    ifstream input_file(xyzfilename);
    vector<Atom> atoms = read_dotxyz(input_file);
    BasisSet obs(basis_name, atoms);
    obs.set_pure(false); // use cartesian gaussians
    int nbf = obs.nbf();

    Engine s_engine(Operator::overlap,obs.max_nprim(),obs.max_l());

    size_t length = nbf * nbf;
    std::vector<double> result(length);

    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
    const auto& buf_vec = s_engine.results(); // will point to computed shell sets
    
    for(auto s1=0; s1!=obs.size(); ++s1) {
        for(auto s2=0; s2!=obs.size(); ++s2) {
            s_engine.compute(obs[s1], obs[s2]); // Compute shell set
            auto ints_shellset = buf_vec[0];    // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out
    
            auto bf1 = shell2bf[s1];  // first basis function in first shell
            auto n1 = obs[s1].size(); // number of basis functions in first shell
            auto bf2 = shell2bf[s2];  // first basis function in second shell
            auto n2 = obs[s2].size(); // number of basis functions in second shell
    
            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1=0, idx=0; f1!=n1; ++f1) {
                for(auto f2=0; f2!=n2; ++f2, ++idx) {
                    // idx = x + (y * width) where x = bf2 + f2 and y = bf1 + f1 
                    result[ (bf1 + f1) * nbf + bf2 + f2 ] = ints_shellset[idx];
                }
            }
        }
    }
    libint2::finalize();
    return py::array(result.size(), result.data()); // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042 there's a workaround
}

// Function takes two strings: xyzfile absolute path (TODO in Angstroms) and basis set name  
py::array kinetic(std::string xyzfilename, std::string basis_name) {
    using namespace libint2;
    using namespace std;
    libint2::initialize();

    // Load basis set and geometry. TODO this assumes units are angstroms... 
    ifstream input_file(xyzfilename);
    vector<Atom> atoms = read_dotxyz(input_file);
    BasisSet obs(basis_name, atoms);
    obs.set_pure(false); // use cartesian gaussians
    int nbf = obs.nbf();

    Engine k_engine(Operator::kinetic,obs.max_nprim(),obs.max_l());
    size_t length = nbf * nbf;
    std::vector<double> result(length);

    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
    const auto& buf_vec = k_engine.results(); // will point to computed shell sets
    
    for(auto s1=0; s1!=obs.size(); ++s1) {
        for(auto s2=0; s2!=obs.size(); ++s2) {
            k_engine.compute(obs[s1], obs[s2]); // Compute shell set
            auto ints_shellset = buf_vec[0];    // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out
    
            auto bf1 = shell2bf[s1];  // first basis function in first shell
            auto n1 = obs[s1].size(); // number of basis functions in first shell
            auto bf2 = shell2bf[s2];  // first basis function in second shell
            auto n2 = obs[s2].size(); // number of basis functions in second shell
    
            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1=0, idx=0; f1!=n1; ++f1) {
                for(auto f2=0; f2!=n2; ++f2, ++idx) {
                    // idx = x + (y * width) where x = bf2 + f2 and y = bf1 + f1 
                    result[ (bf1 + f1) * nbf + bf2 + f2 ] = ints_shellset[idx];
                }
            }
        }
    }
    libint2::finalize();
    return py::array(result.size(), result.data()); // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042 there's a workaround
}

// Function takes two strings: xyzfile absolute path (TODO in Angstroms) and basis set name  
py::array potential(std::string xyzfilename, std::string basis_name) {
    using namespace libint2;
    using namespace std;
    libint2::initialize();

    // Load basis set and geometry. TODO this assumes units are angstroms... 
    ifstream input_file(xyzfilename);
    vector<Atom> atoms = read_dotxyz(input_file);
    BasisSet obs(basis_name, atoms);
    obs.set_pure(false); // use cartesian gaussians

    Engine v_engine(Operator::nuclear,obs.max_nprim(),obs.max_l());
    v_engine.set_params(make_point_charges(atoms));
    int nbf = obs.nbf();

    size_t length = nbf * nbf;
    std::vector<double> result(length);

    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
    const auto& buf_vec = v_engine.results(); // will point to computed shell sets
    
    for(auto s1=0; s1!=obs.size(); ++s1) {
        for(auto s2=0; s2!=obs.size(); ++s2) {
            v_engine.compute(obs[s1], obs[s2]); // Compute shell set
            auto ints_shellset = buf_vec[0];    // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out
    
            auto bf1 = shell2bf[s1];  // first basis function in first shell
            auto n1 = obs[s1].size(); // number of basis functions in first shell
            auto bf2 = shell2bf[s2];  // first basis function in second shell
            auto n2 = obs[s2].size(); // number of basis functions in second shell
    
            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1=0, idx=0; f1!=n1; ++f1) {
                for(auto f2=0; f2!=n2; ++f2, ++idx) {
                    // idx = x + (y * width) where x = bf2 + f2 and y = bf1 + f1 
                    result[ (bf1 + f1) * nbf + bf2 + f2 ] = ints_shellset[idx];
                }
            }
        }
    }
    libint2::finalize();
    return py::array(result.size(), result.data()); // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042 there's a workaround
}

py::array eri(std::string xyzfilename, std::string basis_name) {
    // workaround for data copying: perhaps pass an empty numpy array, then populate it in C++? avoids last line, which copies
    using namespace libint2;
    using namespace std;
    libint2::initialize();

    // Load basis set and geometry. TODO this assumes units are angstroms... 
    ifstream input_file(xyzfilename);
    vector<Atom> atoms = read_dotxyz(input_file);
    BasisSet obs(basis_name, atoms);
    obs.set_pure(false); // use cartesian gaussians
    int nbf = obs.nbf();
    Engine eri_engine(Operator::coulomb,obs.max_nprim(),obs.max_l());

    size_t length = nbf * nbf * nbf * nbf;
    std::vector<double> result(length);

    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
    const auto& buf_vec = eri_engine.results(); // will point to computed shell sets
    
    for(auto s1=0; s1!=obs.size(); ++s1) {
        for(auto s2=0; s2!=obs.size(); ++s2) {
            for(auto s3=0; s3!=obs.size(); ++s3) {
                for(auto s4=0; s4!=obs.size(); ++s4) {
                    eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]); // Compute shell set
                    auto ints_shellset = buf_vec[0];    // Location of the computed integrals
                    if (ints_shellset == nullptr)
                        continue;  // nullptr returned if the entire shell-set was screened out
    
                    auto bf1 = shell2bf[s1];  // first basis function in first shell
                    auto n1 = obs[s1].size(); // number of basis functions in first shell
                    auto bf2 = shell2bf[s2];  // first basis function in second shell
                    auto n2 = obs[s2].size(); // number of basis functions in second shell
                    auto bf3 = shell2bf[s3];  // first basis function in third shell
                    auto n3 = obs[s3].size(); // number of basis functions in third shell
                    auto bf4 = shell2bf[s4];  // first basis function in fourth shell
                    auto n4 = obs[s4].size(); // number of basis functions in fourth shell
    
                    // Loop over shell block, keeping a total count idx for the size of shell set
                    for(auto f1=0, idx=0; f1!=n1; ++f1) {
                        size_t offset_1 = (bf1 + f1) * nbf * nbf * nbf;
                        for(auto f2=0; f2!=n2; ++f2) {
                            size_t offset_2 = (bf2 + f2) * nbf * nbf;
                            for(auto f3=0; f3!=n3; ++f3) {
                                size_t offset_3 = (bf3 + f3) * nbf;
                                for(auto f4=0; f4!=n4; ++f4, ++idx) {
                                    result[offset_1 + offset_2 + offset_3 + bf4 + f4] = ints_shellset[idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    libint2::finalize();
    return py::array(result.size(), result.data()); // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042 there's a workaround
}


// Coulomb integral derivative function
py::array eri_deriv(std::string xyzfilename, std::string basis_name, std::vector<int> deriv_vec) {
    using namespace libint2;
    using namespace std;
    libint2::initialize();
    // Get order of differentiation
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);
    // Number of derivatives at this order of differentiation. 
    // 1/n! * Multiply(from i=0 to n-1)(3k + i) where n = deriv_order and k=4 number of centers
    //std::vector<int> total_shell_derivatives{0, 12, 78, 364, 1365};

    // Default: first derivatives, but if deriv_order=2 then use 2nd derivative buffer index lookup
    const std::vector<int> buffer_index_lookup1 = {0,1,2,3,4,5,6,7,8,9,10,11};
    const std::vector<std::vector<int>> buffer_index_lookup2 = {
        { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11},
        { 1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
        { 2, 13, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
        { 3, 14, 24, 33, 34, 35, 36, 37, 38, 39, 40, 41},
        { 4, 15, 25, 34, 42, 43, 44, 45, 46, 47, 48, 49},
        { 5, 16, 26, 35, 43, 50, 51, 52, 53, 54, 55, 56},
        { 6, 17, 27, 36, 44, 51, 57, 58, 59, 60, 61, 62},
        { 7, 18, 28, 37, 45, 52, 58, 63, 64, 65, 66, 67},
        { 8, 19, 29, 38, 46, 53, 59, 64, 68, 69, 70, 71},
        { 9, 20, 30, 39, 47, 54, 60, 65, 69, 72, 73, 74},
        {10, 21, 31, 40, 48, 55, 61, 66, 70, 73, 75, 76},
        {11, 22, 32, 41, 49, 56, 62, 67, 71, 74, 76, 77},};
    // TODO 
    // if deriv_order == 3
    // if deriv_order == 4

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    for (int i = 0; i < deriv_vec.size(); i++) {
        if (deriv_vec[i] > 0) {
            for (int j = 0; j < deriv_vec[i]; j++) {
                desired_atom_indices.push_back(i / 3);
                desired_coordinates.push_back(i % 3);
            }
        }
    }

    // Load basis set and geometry.
    ifstream input_file(xyzfilename);
    vector<Atom> atoms = read_dotxyz(input_file);
    BasisSet obs(basis_name, atoms);
    obs.set_pure(false); // use cartesian gaussians
    int nbf = obs.nbf();
    // ERI derivative integral engine
    Engine eri_engine(Operator::coulomb,obs.max_nprim(),obs.max_l(),deriv_order);

    // Get size of ERI array and allocate 
    size_t length = nbf * nbf * nbf * nbf;
    std::vector<double> result(length);

    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
    const auto shell2atom = obs.shell2atom(atoms); // maps shell index to atom index
    const auto& buf_vec = eri_engine.results(); // will point to computed shell sets
    
    for(auto s1=0; s1!=obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];     // Index of first basis function in shell 1
        auto atom1 = shell2atom[s1]; // Atom index of shell 1
        auto n1 = obs[s1].size();    // number of basis functions in shell 1
        for(auto s2=0; s2!=obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom[s2]; // Atom index of shell 2
            auto n2 = obs[s2].size();    // number of basis functions in shell 2
            for(auto s3=0; s3!=obs.size(); ++s3) {
                auto bf3 = shell2bf[s3];     // Index of first basis function in shell 3
                auto atom3 = shell2atom[s3]; // Atom index of shell 3
                auto n3 = obs[s3].size();    // number of basis functions in shell 3
                for(auto s4=0; s4!=obs.size(); ++s4) {
                    auto bf4 = shell2bf[s4];     // Index of first basis function in shell 4
                    auto atom4 = shell2atom[s4]; // Atom index of shell 4
                    auto n4 = obs[s4].size();    // number of basis functions in shell 4

                    // If the atoms are the same we ignore it as the derivatives will be zero.
                    if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;

                    // Create list of atom indices corresponding to each shell. Libint uses longs, so we will too.
                    std::vector<long> shell_atom_index_list{atom1,atom2,atom3,atom4};

                    // Every shell quartet has 4 atom indices. 
                    // We can check if EVERY differentiated atom according to deriv_vec is contained in this set of 4 atom indices
                    // This will ensure the derivative we want is in the buffer.
                    std::vector<int> desired_shell_atoms; 
                    for (int i=0; i < deriv_order; i++){
                        int desired_atom = desired_atom_indices[i];
                        if (shell_atom_index_list[0] == desired_atom) desired_shell_atoms.push_back(0); 
                        else if (shell_atom_index_list[1] == desired_atom) desired_shell_atoms.push_back(1); 
                        else if (shell_atom_index_list[2] == desired_atom) desired_shell_atoms.push_back(2); 
                        else if (shell_atom_index_list[3] == desired_atom) desired_shell_atoms.push_back(3); 
                    }

                    // If the length of this vector is not == deriv_order, this shell quartet can be skipped, since it does not contain desired derivative
                    if (desired_shell_atoms.size() != deriv_order) continue;

                    // If we made it this far, the shell derivative we want is in the buffer, perhaps even more than once.
                    // Compute this shell quartet derivative, and find one of the buffer indices which gives it to us.
                    eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]); // Compute shell set, fills buf_vec

                    // Now convert these shell atom indices into a shell derivative index
                    // shell_derivative is a set of indices of length deriv_order with values between 0 and 11, representing the 12 possible shell center coordinates.
                    // Index 0 represents d^n/dx1^n, etc.
                    std::vector<int> shell_derivative;
                    for (int i=0; i < deriv_order; i++){
                        shell_derivative.push_back(3 * desired_shell_atoms[i] + desired_coordinates[i]);
                    }

                    //TODO The buffer index converts the multidimensional index shell_derivative into a one-dimensional buffer index
                    // according to the layout defined in the Libint wiki
                    // Depending on deriv_order, buffer_index_lookup will be a different dimension of array.
                    int buffer_idx;
                    if (deriv_order == 1) { 
                        buffer_idx = buffer_index_lookup1[shell_derivative[0]];
                    }
                    else if (deriv_order == 2) { 
                        buffer_idx = buffer_index_lookup2[shell_derivative[0]][shell_derivative[1]];
                    }
                    
                    auto ints_shellset = buf_vec[buffer_idx]; // Location of the computed integrals

                    if (ints_shellset == nullptr)
                        continue;  // nullptr returned if the entire shell-set was screened out
    
                    // Loop over shell block, keeping a total count idx for the size of shell set
                    for(auto f1=0, idx=0; f1!=n1; ++f1) {
                        size_t offset_1 = (bf1 + f1) * nbf * nbf * nbf;
                        for(auto f2=0; f2!=n2; ++f2) {
                            size_t offset_2 = (bf2 + f2) * nbf * nbf;
                            for(auto f3=0; f3!=n3; ++f3) {
                                size_t offset_3 = (bf3 + f3) * nbf;
                                for(auto f4=0; f4!=n4; ++f4, ++idx) {
                                    result[offset_1 + offset_2 + offset_3 + bf4 + f4] = ints_shellset[idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    libint2::finalize();
    return py::array(result.size(), result.data()); // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042 there's a workaround
}




// Define module named 'libint_interface' which can be imported with python
// The second arg, 'm' defines a variable py::module_ which can be used to create
// bindings. the def() methods generates binding code that exposes new functions to Python.
PYBIND11_MODULE(libint_interface, m) {
    m.doc() = "pybind11 libint interface to molecular integrals"; // optional module docstring
    m.def("overlap", &overlap, "Computes overlap integrals with libint");
    m.def("kinetic", &kinetic, "Computes kinetic integrals with libint");
    m.def("potential", &potential, "Computes potential integrals with libint");
    m.def("eri", &eri, "Computes electron repulsion integrals with libint");
    //m.def("overlap_deriv", &overlap_deriv, "Computes overlap integral nuclear derivatives with libint");
    //m.def("kinetic_deriv", &kinetic_deriv, "Computes kinetic integral nuclear derivatives with libint");
    //m.def("potential_deriv", &potential_deriv, "Computes potential integral nuclear derivatives with libint");
    m.def("eri_deriv", &eri_deriv, "Computes electron repulsion integral nuclear derivatives with libint");

}

// Temporary libint reference: new shared library compilation
// currently needs export LD_LIBRARY_PATH=/path/to/libint2.so. Alternatively, add compiler flag -Wl,-rpath /path/to/where/libint2.so/is/located
// Compilation script for libint with am=2 deriv=4, may need to set LD_LIBRARY_PATH = /path/to/libint2.so corresponding to this installation.
// g++ libint_interface.cc -o libint_interface`python3-config --extension-suffix`  -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ -lint2 

// Warning: above is very slow since its a huge copy of libint. can use smaller version, just s, p with gradients,
// Can do quick compile with the following:
//g++ -c libint_interface.cc -o libint_interface.o -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include/libint2 -lint2 -L/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/lib

//g++ libint_interface.o -o libint_interface`python3-config --extension-suffix` -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include/libint2 -lint2 -L/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/lib

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
