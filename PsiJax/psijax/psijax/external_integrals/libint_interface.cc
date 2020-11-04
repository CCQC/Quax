#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdlib.h>
#include <libint2.hpp>

namespace py = pybind11;

std::vector<libint2::Atom> atoms;
libint2::BasisSet obs;
int nbf;
int natom;
int ncart;
std::vector<size_t> shell2bf;
std::vector<long> shell2atom;

// Creates atom objects from xyz file path
std::vector<libint2::Atom> get_atoms(std::string xyzfilename) 
{
    std::ifstream input_file(xyzfilename);
    std::vector<libint2::Atom> atoms = libint2::read_dotxyz(input_file);
    return atoms;
}

// Must call initialize before computing ints 
void initialize(std::string xyzfilename, std::string basis_name) {
    libint2::initialize();
    atoms = get_atoms(xyzfilename);
    // Move harddrive load of basis and xyz to happen only once
    obs = libint2::BasisSet(basis_name, atoms);
    obs.set_pure(false); // use cartesian gaussians
    // Get size of potential derivative array and allocate 
    nbf = obs.nbf();
    natom = atoms.size();
    ncart = natom * 3;
    shell2bf = obs.shell2bf(); // maps shell index to basis function index
    shell2atom = obs.shell2atom(atoms); // maps shell index to atom index
}

void finalize() {
    libint2::finalize();
}

// TODO re formulate all functions in terms of global var's
// TODO get rid of cartesian gaussians
// TODO test third, fourth derivs
// TODO move static const buffer index lookups to global variable at top of file, give unique names for each integral type
// TODO consider changing overlap, kinetic deriv algos to cartesian product algo used by eri and potential derivs, for consistency
// TODO anyway to get around the if derivorder==1 if derivorder==2 etc in every shell quartet loop? would be nice to have dynamic indices for buffer lookups

// Cartesian product of arbitrary number of vectors, given a vector of vectors
// Used to find all possible combinations of indices which correspond to desired nuclear derivatives
// For example, if molecule has two atoms, A and B, and we want nuclear derivative d^2/dAz dBz, represented by deriv_vec = [0,0,1,0,0,1], 
// and we are looping over 4 shells in ERI's, and the four shells are atoms (0,0,1,1), then possible indices 
// of the 0-11 shell cartesian component indices are {2,5} for d/dAz and {8,11} for d/dBz.
// So the vector passed to cartesian_product is { {{2,5},{8,11}}, and all combinations of elements from first and second subvectors
// are produced, and the total nuclear derivative of the shell is obtained by summing all of these pieces together.
// These resulting indices are converted to flattened Libint buffer indices using the generate_*_lookup functions, explained below.
std::vector<std::vector<int>> cartesian_product (const std::vector<std::vector<int>>& v) {
    std::vector<std::vector<int>> s = {{}};
    for (const auto& u : v) {
        std::vector<std::vector<int>> r;
        for (const auto& x : s) {
            for (const auto y : u) {
                r.push_back(x);
                r.back().push_back(y);
            }
        }
        s = std::move(r);
    }
    return s;
}

// These functions, generate_*_lookup, create the buffer index lookup arrays. 
// When given a set of indices which represent a Shell derivative operator, e.g. 0,0 == d/dx1 d/dx1, 0,1 = d/dx1 d/dx2, etc
// these arrays, when indexed with those indices, give the flattened buffer index according to the order these shell derivatives
// are packed into a Libint integral Engine buffer.  
// These arrays are always the same for finding the shell derivative mapping for overlap, kinetic, and ERI for a given derivative order. 
// These are also used for nuclear derivatives of nuclear attraction integrals,
// which vary in size dynamically due to the presence of additional nuclear derivatives 
std::vector<int> generate_1d_lookup(int dim_size) { 
    std::vector<int> lookup(dim_size, 0);
    for (int i = 0; i < dim_size; i++){
        lookup[i] = i; 
    }
    return lookup;
}

std::vector<std::vector<int>> generate_2d_lookup(int dim_size) { 
    using namespace std;
    vector<vector<int>> lookup(dim_size, vector<int> (dim_size, 0));
    vector<vector<int>> combos; // always the same, list of lists

    // Collect multidimensional indices corresponding to generalized upper triangle 
    for (int i = 0; i < dim_size; i++) {
      for (int j = i; j < dim_size; j++) {
        vector<int> tmp = {i, j};
        combos.push_back(tmp);
      }
    }
    // Build lookup array and return
    for (int i = 0; i < combos.size(); i++){
        auto multi_idx = combos[i];
        // Loop over all permutations, assign 1d buffer index to appropriate addresses in totally symmetric lookup array
        do { 
        lookup[multi_idx[0]][multi_idx[1]] = i; 
        } 
        while (next_permutation(multi_idx.begin(),multi_idx.end())); 
    }
    return lookup;
}

std::vector<std::vector<std::vector<int>>> generate_3d_lookup(int dim_size) { 
    //TODO test this.
    using namespace std;
    vector<vector<vector<int>>> lookup(dim_size, vector<vector<int>>(dim_size, vector<int>(dim_size)));
    vector<vector<int>> combos; // always the same, list of lists
    // Collect multidimensional indices corresponding to generalized upper triangle 
    for (int i = 0; i < dim_size; i++) {
      for (int j = i; j < dim_size; j++) {
        for (int k = j; k < dim_size; k++) {
          vector<int> tmp = {i, j, k};
          combos.push_back(tmp);
        }
      }
    }
    // Build lookup array and return
    for (int i = 0; i < combos.size(); i++){
        auto multi_idx = combos[i];
        // Loop over all permutations, assign 1d buffer index to appropriate addresses in totally symmetric lookup array
        do { 
        lookup[multi_idx[0]][multi_idx[1]][multi_idx[2]] = i; 
        } 
        while (next_permutation(multi_idx.begin(),multi_idx.end())); 
    }
    return lookup;
}

std::vector<std::vector<std::vector<std::vector<int>>>> generate_4d_lookup(int dim_size) { 
    //TODO test this.
    using namespace std;
    vector<vector<vector<vector<int>>>> lookup(dim_size, vector<vector<vector<int>>>(dim_size, vector<vector<int>>(dim_size, vector<int>(dim_size))));
    vector<vector<int>> combos; // always the same, list of lists
    // Collect multidimensional indices corresponding to generalized upper triangle 
    for (int i = 0; i < dim_size; i++) {
      for (int j = i; j < dim_size; j++) {
        for (int k = j; k < dim_size; k++) {
          for (int l = k; l < dim_size; l++) {
            vector<int> tmp = {i, j, k, l};
            combos.push_back(tmp);
          }
        }
      }
    }
    // Build lookup array and return
    for (int i = 0; i < combos.size(); i++){
        auto multi_idx = combos[i];
        // Loop over all permutations, assign 1d buffer index to appropriate addresses in totally symmetric lookup array
        do { 
        lookup[multi_idx[0]][multi_idx[1]][multi_idx[2]][multi_idx[3]] = i; 
        } 
        while (next_permutation(multi_idx.begin(),multi_idx.end())); 
    }
    return lookup;
}

// Converts a derivative vector (3*Natom array of integers defining which coordinates to 
// differentiate wrt and how many times) to a set of atom indices and coordinate indices 0,1,2->x,y,z
void process_deriv_vec(std::vector<int> deriv_vec, 
                       std::vector<int> *desired_atoms, 
                       std::vector<int> *desired_coordinates) 
{
    for (int i = 0; i < deriv_vec.size(); i++) {
        if (deriv_vec[i] > 0) {
            for (int j = 0; j < deriv_vec[i]; j++) {
                desired_atoms->push_back(i / 3);
                desired_coordinates->push_back(i % 3);
            }
        }
    }
}


// Compute overlap integrals
py::array overlap() {
    // Overlap integral engine
    libint2::Engine s_engine(libint2::Operator::overlap,obs.max_nprim(),obs.max_l());
    const auto& buf_vec = s_engine.results(); // will point to computed shell sets
    size_t length = nbf * nbf;
    std::vector<double> result(length); // vector to store integral array

    for(auto s1=0; s1!=obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        for(auto s2=0; s2!=obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];  // first basis function in second shell
            auto n2 = obs[s2].size(); // number of basis functions in second shell

            s_engine.compute(obs[s1], obs[s2]); // Compute shell set
            auto ints_shellset = buf_vec[0];    // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out
            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1=0, idx=0; f1!=n1; ++f1) {
                for(auto f2=0; f2!=n2; ++f2, ++idx) {
                    result[(bf1 + f1) * nbf + bf2 + f2] = ints_shellset[idx];
                }
            }
        }
    }
    return py::array(result.size(), result.data()); 
}

// Compute kinetic energy integrals
py::array kinetic() {
    // Kinetic energy integral engine
    libint2::Engine t_engine(libint2::Operator::kinetic,obs.max_nprim(),obs.max_l());
    const auto& buf_vec = t_engine.results(); // will point to computed shell sets
    size_t length = nbf * nbf;
    std::vector<double> result(length);

    for(auto s1=0; s1!=obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        for(auto s2=0; s2!=obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];  // first basis function in second shell
            auto n2 = obs[s2].size(); // number of basis functions in second shell

            t_engine.compute(obs[s1], obs[s2]); // Compute shell set
            auto ints_shellset = buf_vec[0];    // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out
            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1=0, idx=0; f1!=n1; ++f1) {
                for(auto f2=0; f2!=n2; ++f2, ++idx) {
                    result[ (bf1 + f1) * nbf + bf2 + f2 ] = ints_shellset[idx];
                }
            }
        }
    }
    return py::array(result.size(), result.data());
}

// Compute nuclear-electron potential energy integrals
py::array potential() {
    // Potential integral engine
    libint2::Engine v_engine(libint2::Operator::nuclear,obs.max_nprim(),obs.max_l());
    v_engine.set_params(make_point_charges(atoms));
    const auto& buf_vec = v_engine.results(); // will point to computed shell sets

    size_t length = nbf * nbf;
    std::vector<double> result(length);
    
    for(auto s1=0; s1!=obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        for(auto s2=0; s2!=obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];  // first basis function in second shell
            auto n2 = obs[s2].size(); // number of basis functions in second shell

            v_engine.compute(obs[s1], obs[s2]); // Compute shell set
            auto ints_shellset = buf_vec[0];    // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out
            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1=0, idx=0; f1!=n1; ++f1) {
                for(auto f2=0; f2!=n2; ++f2, ++idx) {
                    // idx = x + (y * width) where x = bf2 + f2 and y = bf1 + f1 
                    result[ (bf1 + f1) * nbf + bf2 + f2 ] = ints_shellset[idx];
                }
            }
        }
    }
    return py::array(result.size(), result.data());
}

// Computes electron repulsion integrals
py::array eri() {
    // workaround for data copying: perhaps pass an empty numpy array, then populate it in C++? avoids last line, which copies
    libint2::Engine eri_engine(libint2::Operator::coulomb,obs.max_nprim(),obs.max_l());
    const auto& buf_vec = eri_engine.results(); // will point to computed shell sets

    size_t length = nbf * nbf * nbf * nbf;
    std::vector<double> result(length);
    
    for(auto s1=0; s1!=obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        for(auto s2=0; s2!=obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];  // first basis function in second shell
            auto n2 = obs[s2].size(); // number of basis functions in second shell
            for(auto s3=0; s3!=obs.size(); ++s3) {
                auto bf3 = shell2bf[s3];  // first basis function in third shell
                auto n3 = obs[s3].size(); // number of basis functions in third shell
                for(auto s4=0; s4!=obs.size(); ++s4) {
                    auto bf4 = shell2bf[s4];  // first basis function in fourth shell
                    auto n4 = obs[s4].size(); // number of basis functions in fourth shell

                    eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]); // Compute shell set
                    auto ints_shellset = buf_vec[0];    // Location of the computed integrals
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
    return py::array(result.size(), result.data()); // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042 there's a workaround
}

// Computes nuclear derivatives of overlap integrals
py::array overlap_deriv(std::vector<int> deriv_vec) {
    // Get order of differentiation
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Lookup arrays for mapping shell derivative index to buffer index 
    static const std::vector<int> buffer_index_lookup1 = generate_1d_lookup(6);
    static const std::vector<std::vector<int>> buffer_index_lookup2 = generate_2d_lookup(6); 
    // TODO add 3rd, 4th order

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    // Make sure number of atoms match size of deriv vec
    assert(3 * atoms.size() == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");

    // Overlap integral derivative engine
    libint2::Engine s_engine(libint2::Operator::overlap,obs.max_nprim(),obs.max_l(),deriv_order);

    // Get size of overlap derivative array and allocate 
    size_t length = nbf * nbf;
    std::vector<double> result(length);

    const auto& buf_vec = s_engine.results(); // will point to computed shell sets
    
    for(auto s1=0; s1!=obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];     // Index of first basis function in shell 1
        auto atom1 = shell2atom[s1]; // Atom index of shell 1
        auto n1 = obs[s1].size();    // number of basis functions in shell 1
        for(auto s2=0; s2!=obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom[s2]; // Atom index of shell 2
            auto n2 = obs[s2].size();    // number of basis functions in shell 2
            // If the atoms are the same we ignore it as the derivatives will be zero.
            if (atom1 == atom2) continue;

            // Create list of atom indices corresponding to each shell. Libint uses longs, so we will too.
            std::vector<long> shell_atom_index_list{atom1,atom2};

            // We can check if EVERY differentiated atom according to deriv_vec is contained in this set of 2 atom indices
            // This will ensure the derivative we want is in the buffer.
            std::vector<int> desired_shell_atoms; 
            for (int i=0; i < deriv_order; i++){
                int desired_atom = desired_atom_indices[i];
                if (shell_atom_index_list[0] == desired_atom) desired_shell_atoms.push_back(0); 
                else if (shell_atom_index_list[1] == desired_atom) desired_shell_atoms.push_back(1); 
            }

            // If the length of this vector is not == deriv_order, this shell duet can be skipped, since it does not contain desired derivative
            if (desired_shell_atoms.size() != deriv_order) continue;

            // If we made it this far, the shell derivative we want is in the buffer, perhaps even more than once. 
            s_engine.compute(obs[s1], obs[s2]); 

            // Now convert these shell atom indices into a shell derivative index, a set of indices length deriv_order with values between 0 and 5, corresponding to 6 possible shell center coordinates
            std::vector<int> shell_derivative;
            for (int i=0; i < deriv_order; i++){
                shell_derivative.push_back(3 * desired_shell_atoms[i] + desired_coordinates[i]);
            }

            // Now we must convert our multidimensional shell_derivative index into a one-dimensional buffer index. 
            // We know how to do this since libint tells us what order they come in. The lookup arrays above map the multidim index to the buffer idx
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
                for(auto f2=0; f2!=n2; ++f2, ++idx) {
                    result[(bf1 + f1) * nbf + bf2 + f2 ] = ints_shellset[idx];
                }
            }
        }
    }
    return py::array(result.size(), result.data()); 
}

// Computes nuclear derivatives of kinetic energy integrals
py::array kinetic_deriv(std::vector<int> deriv_vec) {
    // Get order of differentiation
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Lookup arrays for mapping shell derivative index to buffer index 
    static const std::vector<int> buffer_index_lookup1 = {0,1,2,3,4,5};
    static const std::vector<std::vector<int>> buffer_index_lookup2 = generate_2d_lookup(6); 
    // TODO add 3rd, 4th order

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    assert(3 * atoms.size() == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");

    // Kinetic integral derivative engine
    libint2::Engine t_engine(libint2::Operator::kinetic,obs.max_nprim(),obs.max_l(),deriv_order);
    const auto& buf_vec = t_engine.results(); // will point to computed shell sets

    size_t length = nbf * nbf;
    std::vector<double> result(length);
    
    for(auto s1=0; s1!=obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];     // Index of first basis function in shell 1
        auto atom1 = shell2atom[s1]; // Atom index of shell 1
        auto n1 = obs[s1].size();    // number of basis functions in shell 1
        for(auto s2=0; s2!=obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom[s2]; // Atom index of shell 2
            auto n2 = obs[s2].size();    // number of basis functions in shell 2
            // If the atoms are the same we ignore it as the derivatives will be zero.
            if (atom1 == atom2) continue;

            // Create list of atom indices corresponding to each shell. Libint uses longs, so we will too.
            std::vector<long> shell_atom_index_list{atom1,atom2};

            // We can check if EVERY differentiated atom according to deriv_vec is contained in this set of 2 atom indices
            // This will ensure the derivative we want is in the buffer.
            std::vector<int> desired_shell_atoms; 
            for (int i=0; i < deriv_order; i++){
                int desired_atom = desired_atom_indices[i];
                if (shell_atom_index_list[0] == desired_atom) desired_shell_atoms.push_back(0); 
                else if (shell_atom_index_list[1] == desired_atom) desired_shell_atoms.push_back(1); 
            }

            // If the length of this vector is not == deriv_order, this shell duet can be skipped, since it does not contain desired derivative
            if (desired_shell_atoms.size() != deriv_order) continue;

            // If we made it this far, the shell derivative we want is in the buffer, perhaps even more than once. 
            t_engine.compute(obs[s1], obs[s2]); 

            // Now convert these shell atom indices into a shell derivative index, a set of indices length deriv_order with values between 0 and 5, corresponding to 6 possible shell center coordinates
            std::vector<int> shell_derivative;
            for (int i=0; i < deriv_order; i++){
                shell_derivative.push_back(3 * desired_shell_atoms[i] + desired_coordinates[i]);
            }

            // Now we must convert our multidimensional shell_derivative index into a one-dimensional buffer index. 
            // We know how to do this since libint tells us what order they come in. The lookup arrays above map the multidim index to the buffer idx
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
                for(auto f2=0; f2!=n2; ++f2, ++idx) {
                    result[(bf1 + f1) * nbf + bf2 + f2 ] = ints_shellset[idx];
                }
            }
        }
    }
    return py::array(result.size(), result.data()); 
}

// Computes nuclear derivatives of potential energy integrals 
py::array potential_deriv(std::vector<int> deriv_vec) {
    // Get order of differentiation
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Load basis set and geometry.
    assert(3 * atoms.size() == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");

    // Lookup arrays for mapping shell derivative index to buffer index 
    // Potential derivatives are weird. The dimension size is 6 + ncart + ncart 
    // I believe only the first 6 and last ncart are relevent. Idk what is with the ghost dimension 
    int dimensions = 6 + 2 * 3 * atoms.size();
    static const std::vector<int> buffer_index_lookup1 = generate_1d_lookup(dimensions);
    static const std::vector<std::vector<int>> buffer_index_lookup2 = generate_2d_lookup(dimensions);

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    // Potential integral derivative engine
    libint2::Engine v_engine(libint2::Operator::nuclear,obs.max_nprim(),obs.max_l(),deriv_order);
    v_engine.set_params(libint2::make_point_charges(atoms));
    const auto& buf_vec = v_engine.results(); // will point to computed shell sets

    // Get size of potential derivative array and allocate 
    size_t length = nbf * nbf;
    std::vector<double> result(length);

    for(auto s1=0; s1!=obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];     // Index of first basis function in shell 1
        auto atom1 = shell2atom[s1]; // Atom index of shell 1
        auto n1 = obs[s1].size();    // number of basis functions in shell 1
        for(auto s2=0; s2!=obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom[s2]; // Atom index of shell 2
            auto n2 = obs[s2].size();    // number of basis functions in shell 2

            // Create list of atom indices corresponding to each shell. Libint uses longs, so we will too.
            std::vector<long> shell_atom_index_list{atom1,atom2};

            // Initialize 2d vector, with DERIV_ORDER subvectors
            // Each subvector contains index candidates which are possible choices for each partial derivative operator
            // In other words, indices looks like { {choices for first deriv operator} {choices for second deriv op} {third} ...}
            // The cartesian product of these subvectors gives all combos that need to be summed to form total nuclear derivative of integrals
            std::vector<std::vector<int>> indices; 
            for (int i=0;i<deriv_order; i++){
                std::vector<int> new_vec;
                indices.push_back(new_vec);
            }

            // For every desired atom derivative, check shell and nuclear indices for a match, add it to subvector for that derivative
            // Add in the coordinate index 0,1,2 (x,y,z) in desired coordinates and offset the index appropriately.
            for (int j=0; j < desired_atom_indices.size(); j++){
                int desired_atom_idx = desired_atom_indices[j];
                // Shell indices
                for (int i=0; i<2; i++){
                    int atom_idx = shell_atom_index_list[i];
                    if (atom_idx == desired_atom_idx) { 
                        int tmp = 3 * i + desired_coordinates[j];
                        indices[j].push_back(tmp);
                    }
                }
                // TODO weird action here by libint, theres a NCART block of zeros introduced between shell derivs and real NCART derivs
                // So we compensate by starting from 2 + natom
                // If this is ever changed, this needs to be edited.
                for (int i=0; i<natom; i++){
                    // i = shell_atom_index_list[i];
                    if (i == desired_atom_idx) { 
                        int offset_i = i + 2 + natom;
                        int tmp = 3 * offset_i + desired_coordinates[j];
                        indices[j].push_back(tmp);
                    }
                }
            }

            // Create index combos representing every mixed partial derivative operator which contributes to nuclear derivative
            std::vector<std::vector<int>> index_combos = cartesian_product(indices);

            // Compute the integrals
            v_engine.compute(obs[s1], obs[s2]); 
            
            // Loop over every subvector of index_combos and lookup buffer index.
            std::vector<int> buffer_indices;
            if (deriv_order == 1){
                for (int i=0; i < index_combos.size(); i++){
                    int idx1 = index_combos[i][0];
                    buffer_indices.push_back(buffer_index_lookup1[idx1]);
                }
            }
            else if (deriv_order == 2){
                for (int i=0; i < index_combos.size(); i++){
                    int idx1 = index_combos[i][0];
                    int idx2 = index_combos[i][1];
                    buffer_indices.push_back(buffer_index_lookup2[idx1][idx2]);
                }
            }

            // Loop over every buffer index and accumulate for every shell set.
            for(auto i=0; i<buffer_indices.size(); ++i) {
              auto ints_shellset = buf_vec[buffer_indices[i]]; 
              if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
              for(auto f1=0, idx=0; f1!=n1; ++f1) {
                for(auto f2=0; f2!=n2; ++f2, ++idx) {
                  result[(bf1 + f1) * nbf + bf2 + f2] += ints_shellset[idx]; 
                }
              }
            }
        }
    }
    return py::array(result.size(), result.data()); 
}

// Computes nuclear derivatives of electron repulsion integrals
py::array eri_deriv(std::vector<int> deriv_vec) {
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);
    assert(deriv_order <= 2);
    // Lookup arrays for mapping shell derivative index to buffer index 
    // static const std::vector<int> buffer_index_lookup1 = {0,1,2,3,4,5,6,7,8,9,10,11};
    static const std::vector<int> buffer_index_lookup1 = generate_1d_lookup(12);
    static const std::vector<std::vector<int>> buffer_index_lookup2 = generate_2d_lookup(12);

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    assert(3 * atoms.size() == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");

    // ERI derivative integral engine
    libint2::Engine eri_engine(libint2::Operator::coulomb,obs.max_nprim(),obs.max_l(),deriv_order);
    const auto& buf_vec = eri_engine.results(); // will point to computed shell sets

    size_t length = nbf * nbf * nbf * nbf;
    std::vector<double> result(length);

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

                    // Initialize 2d vector, with DERIV_ORDER subvectors
                    // Each subvector contains index candidates which are possible choices for each partial derivative operator
                    // In other words, indices looks like { {choices for first deriv operator} {choices for second deriv op} {third} ...}
                    // The cartesian product of these subvectors gives all combos that need to be summed to form total nuclear derivative of integrals
                    std::vector<std::vector<int>> indices;
                    for (int i=0;i<deriv_order; i++){
                        std::vector<int> new_vec;
                        indices.push_back(new_vec);
                    }
                
                    // For every desired atom derivative, check shell and nuclear indices for a match, add it to subvector for that derivative
                    // Add in the coordinate index 0,1,2 (x,y,z) in desired coordinates and offset the index appropriately.
                    for (int j=0; j < desired_atom_indices.size(); j++){
                        int desired_atom_idx = desired_atom_indices[j];
                        // Shell indices
                        for (int i=0; i<4; i++){
                            int atom_idx = shell_atom_index_list[i];
                            if (atom_idx == desired_atom_idx) {
                                int tmp = 3 * i + desired_coordinates[j];
                                indices[j].push_back(tmp);
                            }
                        }
                    }
                    // If one of the derivative operators cannot be satisfied by any of the shell's centers, 
                    // derivative is 0. skip this shell quartet.
                    bool check = false;
                    for (int i=0;i<deriv_order; i++){
                        if (indices[i].size() == 0) check = true;
                    }
                    if (check) continue;
                    
                    // Now indices is a vector of vectors, where each subvector is your choices for the first derivative operator, second, third, etc
                    // and the total number of subvectors is the order of differentiation
                    // Now we want all combinations where we pick exactly one index from each subvector.
                    // This is achievable through a cartesian product 
                    std::vector<std::vector<int>> index_combos = cartesian_product(indices);

                    // Now create buffer_indices from these index combos using lookup array
                    std::vector<int> buffer_indices;
                    if (deriv_order == 1){ 
                        for (int i=0; i < index_combos.size(); i++){
                            int idx1 = index_combos[i][0];
                            buffer_indices.push_back(buffer_index_lookup1[idx1]);
                        }
                    }
                    else if (deriv_order == 2){ 
                        for (int i=0; i < index_combos.size(); i++){
                            int idx1 = index_combos[i][0];
                            int idx2 = index_combos[i][1];
                            buffer_indices.push_back(buffer_index_lookup2[idx1][idx2]);
                        }
                    }

                    // If we made it this far, the shell derivative we want is contained in the buffer. 
                    eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]); // Compute shell set, fills buf_vec

                    for(auto i=0; i<buffer_indices.size(); ++i) {
                        auto ints_shellset = buf_vec[buffer_indices[i]];
                        if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                        for(auto f1=0, idx=0; f1!=n1; ++f1) {
                            size_t offset_1 = (bf1 + f1) * nbf * nbf * nbf;
                            for(auto f2=0; f2!=n2; ++f2) {
                                size_t offset_2 = (bf2 + f2) * nbf * nbf;
                                for(auto f3=0; f3!=n3; ++f3) {
                                    size_t offset_3 = (bf3 + f3) * nbf;
                                    for(auto f4=0; f4!=n4; ++f4, ++idx) {
                                        result[offset_1 + offset_2 + offset_3 + bf4 + f4] += ints_shellset[idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return py::array(result.size(), result.data()); // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042 there's a workaround
}


// Define module named 'libint_interface' which can be imported with python
// The second arg, 'm' defines a variable py::module_ which can be used to create
// bindings. the def() methods generates binding code that exposes new functions to Python.
PYBIND11_MODULE(libint_interface, m) {
    m.doc() = "pybind11 libint interface to molecular integrals"; // optional module docstring
    m.def("initialize", &initialize, "Initializes libint, builds geom and basis, assigns globals");
    m.def("finalize", &finalize, "Kills libint");
    m.def("overlap", &overlap, "Computes overlap integrals with libint");
    m.def("kinetic", &kinetic, "Computes kinetic integrals with libint");
    m.def("potential", &potential, "Computes potential integrals with libint");
    m.def("eri", &eri, "Computes electron repulsion integrals with libint");
    m.def("overlap_deriv", &overlap_deriv, "Computes overlap integral nuclear derivatives with libint");
    m.def("kinetic_deriv", &kinetic_deriv, "Computes kinetic integral nuclear derivatives with libint");
    m.def("potential_deriv", &potential_deriv, "Computes potential integral nuclear derivatives with libint");
    m.def("eri_deriv", &eri_deriv, "Computes electron repulsion integral nuclear derivatives with libint");
}

// Temporary libint reference: new shared library compilation
// currently needs export LD_LIBRARY_PATH=/path/to/libint2.so. Alternatively, add compiler flag -Wl,-rpath /path/to/where/libint2.so/is/located
// Compilation script for libint with am=2 deriv=4, may need to set LD_LIBRARY_PATH = /path/to/libint2.so corresponding to this installation.
// g++ libint_interface.cc -o libint_interface`python3-config --extension-suffix`  -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ -lint2 


// NEW ninja install with am=2 deriv=2
// g++ libint_interface.cc -o libint_interface`python3-config --extension-suffix`  -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint_trial3/libint/BUILD/libint-2.7.0-beta.6include/libint2 -I/home/adabbott/Git/libint_trial3/libint/BUILD/libint-2.7.0-beta.6/include -L/home/adabbott/Git/libint_trial3/libint/BUILD/libint-2.7.0-beta.6 -lint2 


// Warning: above is very slow since its a huge copy of libint. can use smaller version, just s, p with gradients,
// Can do quick compile with the following:
//g++ -c libint_interface.cc -o libint_interface.o -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include/libint2 -lint2 -L/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/lib

//g++ libint_interface.o -o libint_interface`python3-config --extension-suffix` -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include/libint2 -lint2 -L/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/lib

