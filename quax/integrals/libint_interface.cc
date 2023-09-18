#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdlib.h>
#include <libint2.hpp>
#include <H5Cpp.h>
#include <iostream>

#include "buffer_lookups.h"

// TODO support spherical harmonic gaussians, try parallelization with openmp, implement symmetry considerations, support 5th, 6th derivs

namespace py = pybind11;
using namespace H5;

std::vector<libint2::Atom> atoms;
libint2::BasisSet obs;
unsigned int nbf;
unsigned int natom;
unsigned int ncart;
std::vector<size_t> shell2bf;
std::vector<long> shell2atom;

// These lookup arrays are for mapping Libint's computed shell-set integrals and integral derivatives to the proper index 
// in the full OEI/TEI array or derivative array.
// ERI,overlap,kinetic buffer lookup arrays are always the same, create at compile time.
// Potential buffer lookups have to be created at runtime since they are dependent on natoms
// Total size of these is (12 + 12^2 + 12^3 + 12^4 + 6 + 6^2 + 6^3 + 6^4) * 2 bytes = 48 kB 
// Note quintic, sextics will likely require long int, probably a different algo.
static const std::vector<int> buffer_index_eri1d = generate_1d_lookup(12);
static const std::vector<std::vector<int>> buffer_index_eri2d = generate_2d_lookup(12);
static const std::vector<std::vector<std::vector<int>>> buffer_index_eri3d = generate_3d_lookup(12);
static const std::vector<std::vector<std::vector<std::vector<int>>>> buffer_index_eri4d = generate_4d_lookup(12);
//static const std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>> buffer_index_eri5d = generate_5d_lookup(12);
//static const std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>> buffer_index_eri6d = generate_6d_lookup(12);
static const std::vector<int> buffer_index_oei1d = generate_1d_lookup(6);
static const std::vector<std::vector<int>> buffer_index_oei2d = generate_2d_lookup(6);
static const std::vector<std::vector<std::vector<int>>> buffer_index_oei3d = generate_3d_lookup(6);
static const std::vector<std::vector<std::vector<std::vector<int>>>> buffer_index_oei4d = generate_4d_lookup(6);
//static const std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>> buffer_index_oei5d = generate_5d_lookup(6);
//static const std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>> buffer_index_oei6d = generate_6d_lookup(6);

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

// Returns total size of the libint integral derivative buffer, which is how many unique nth order derivatives
// wrt k objects which have 3 differentiable coordinates each
// k: how many centers
// n: order of differentiation
// l: how many atoms (needed for potential integrals only!)
int how_many_derivs(int k, int n, int l = 0) {
    int val = 1;
    int factorial = 1;
    for (int i=0; i < n; i++) {
        val *= (3 * (k + l) + i);
        factorial *= i + 1;
    }
    val /= factorial;
    return val;
}

void cwr_recursion(std::vector<int> inp,
                   std::vector<int> &out,
                   std::vector<std::vector<int>> &result,
                   int k, int i, int n)
{
    // base case: if combination size is k, add to result 
    if (out.size() == k){
        result.push_back(out);
        return;
    }
    for (int j = i; j < n; j++){
        out.push_back(inp[j]);
        cwr_recursion(inp, out, result, k, j, n);
        // backtrack - remove current element from solution
        out.pop_back();
    }
}

std::vector<std::vector<int>> generate_multi_index_lookup(int nparams, int deriv_order) {
    using namespace std;
    // Generate vector of indices 0 through nparams-1
    vector<int> inp;
    for (int i = 0; i < nparams; i++) {
        inp.push_back(i);
    }
    // Generate all possible combinations with repitition. 
    // These are upper triangle indices, and the length of them is the total number of derivatives
    vector<int> out;
    vector<vector<int>> combos;
    cwr_recursion(inp, out, combos, deriv_order, 0, nparams);
    return combos;
}

// Compute overlap integrals
py::array overlap() {
    // Overlap integral engine
    libint2::Engine s_engine(libint2::Operator::overlap, obs.max_nprim(), obs.max_l());
    const auto& buf_vec = s_engine.results(); // will point to computed shell sets
    size_t length = nbf * nbf;
    std::vector<double> result(length); // vector to store integral array

    for(auto s1 = 0; s1 != obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        for(auto s2 = 0; s2 != obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];  // first basis function in second shell
            auto n2 = obs[s2].size(); // number of basis functions in second shell

            s_engine.compute(obs[s1], obs[s2]); // Compute shell set
            auto ints_shellset = buf_vec[0];    // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out
            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
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
    libint2::Engine t_engine(libint2::Operator::kinetic, obs.max_nprim(), obs.max_l());
    const auto& buf_vec = t_engine.results(); // will point to computed shell sets
    size_t length = nbf * nbf;
    std::vector<double> result(length);

    for(auto s1 = 0; s1 != obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        for(auto s2 = 0; s2 != obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];  // first basis function in second shell
            auto n2 = obs[s2].size(); // number of basis functions in second shell

            t_engine.compute(obs[s1], obs[s2]); // Compute shell set
            auto ints_shellset = buf_vec[0];    // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out
            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
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
    libint2::Engine v_engine(libint2::Operator::nuclear, obs.max_nprim(), obs.max_l());
    v_engine.set_params(make_point_charges(atoms));
    const auto& buf_vec = v_engine.results(); // will point to computed shell sets

    size_t length = nbf * nbf;
    std::vector<double> result(length);

    for(auto s1 = 0; s1 != obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        for(auto s2 = 0; s2 != obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];  // first basis function in second shell
            auto n2 = obs[s2].size(); // number of basis functions in second shell

            v_engine.compute(obs[s1], obs[s2]); // Compute shell set
            auto ints_shellset = buf_vec[0];    // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out
            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
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
    libint2::Engine eri_engine(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l());
    const auto& buf_vec = eri_engine.results(); // will point to computed shell sets

    size_t length = nbf * nbf * nbf * nbf;
    std::vector<double> result(length);
    
    for(auto s1 = 0; s1 != obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        for(auto s2 = 0; s2 != obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];  // first basis function in second shell
            auto n2 = obs[s2].size(); // number of basis functions in second shell
            for(auto s3=0; s3 != obs.size(); ++s3) {
                auto bf3 = shell2bf[s3];  // first basis function in third shell
                auto n3 = obs[s3].size(); // number of basis functions in third shell
                for(auto s4 = 0; s4 != obs.size(); ++s4) {
                    auto bf4 = shell2bf[s4];  // first basis function in fourth shell
                    auto n4 = obs[s4].size(); // number of basis functions in fourth shell

                    eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]); // Compute shell set
                    auto ints_shellset = buf_vec[0];    // Location of the computed integrals
                    if (ints_shellset == nullptr)
                        continue;  // nullptr returned if the entire shell-set was screened out
                    // Loop over shell block, keeping a total count idx for the size of shell set
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        size_t offset_1 = (bf1 + f1) * nbf * nbf * nbf;
                        for(auto f2 = 0; f2 != n2; ++f2) {
                            size_t offset_2 = (bf2 + f2) * nbf * nbf;
                            for(auto f3 = 0; f3 != n3; ++f3) {
                                size_t offset_3 = (bf3 + f3) * nbf;
                                for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
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
    assert(ncart == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");
    // Get order of differentiation
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    // Overlap integral derivative engine
    libint2::Engine s_engine(libint2::Operator::overlap, obs.max_nprim(), obs.max_l(), deriv_order);

    // Get size of overlap derivative array and allocate 
    size_t length = nbf * nbf;
    std::vector<double> result(length);

    const auto& buf_vec = s_engine.results(); // will point to computed shell sets

    for(auto s1 = 0; s1 != obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];     // Index of first basis function in shell 1
        auto atom1 = shell2atom[s1]; // Atom index of shell 1
        auto n1 = obs[s1].size();    // number of basis functions in shell 1
        for(auto s2 = 0; s2 != obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom[s2]; // Atom index of shell 2
            auto n2 = obs[s2].size();    // number of basis functions in shell 2
            // If the atoms are the same we ignore it as the derivatives will be zero.
            if (atom1 == atom2) continue;

            // Create list of atom indices corresponding to each shell. Libint uses longs, so we will too.
            std::vector<long> shell_atom_index_list{atom1, atom2};

            // We can check if EVERY differentiated atom according to deriv_vec is contained in this set of 2 atom indices
            // This will ensure the derivative we want is in the buffer.
            std::vector<int> desired_shell_atoms; 
            for (int i = 0; i < deriv_order; i++){
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
            for (int i = 0; i < deriv_order; i++){
                shell_derivative.push_back(3 * desired_shell_atoms[i] + desired_coordinates[i]);
            }

            // Now we must convert our multidimensional shell_derivative index into a one-dimensional buffer index. 
            // We know how to do this since libint tells us what order they come in. The lookup arrays above map the multidim index to the buffer idx
            int buffer_idx;
            if (deriv_order == 1) { 
                buffer_idx = buffer_index_oei1d[shell_derivative[0]];
            }
            else if (deriv_order == 2) { 
                buffer_idx = buffer_index_oei2d[shell_derivative[0]][shell_derivative[1]];
            }
            else if (deriv_order == 3) { 
                buffer_idx = buffer_index_oei3d[shell_derivative[0]][shell_derivative[1]][shell_derivative[2]];
            }
            else if (deriv_order == 4) { 
                buffer_idx = buffer_index_oei4d[shell_derivative[0]][shell_derivative[1]][shell_derivative[2]][shell_derivative[3]];
            }

            auto ints_shellset = buf_vec[buffer_idx]; // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out

            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                    result[(bf1 + f1) * nbf + bf2 + f2 ] = ints_shellset[idx];
                }
            }
        }
    }
    return py::array(result.size(), result.data()); 
}

// Computes nuclear derivatives of kinetic energy integrals
py::array kinetic_deriv(std::vector<int> deriv_vec) {
    assert(ncart == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");
    // Get order of differentiation
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    // Kinetic integral derivative engine
    libint2::Engine t_engine(libint2::Operator::kinetic, obs.max_nprim(), obs.max_l(), deriv_order);
    const auto& buf_vec = t_engine.results(); // will point to computed shell sets

    size_t length = nbf * nbf;
    std::vector<double> result(length);

    for(auto s1 = 0; s1 != obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];     // Index of first basis function in shell 1
        auto atom1 = shell2atom[s1]; // Atom index of shell 1
        auto n1 = obs[s1].size();    // number of basis functions in shell 1
        for(auto s2 = 0; s2 != obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom[s2]; // Atom index of shell 2
            auto n2 = obs[s2].size();    // number of basis functions in shell 2
            // If the atoms are the same we ignore it as the derivatives will be zero.
            if (atom1 == atom2) continue;

            // Create list of atom indices corresponding to each shell. Libint uses longs, so we will too.
            std::vector<long> shell_atom_index_list{atom1, atom2};

            // We can check if EVERY differentiated atom according to deriv_vec is contained in this set of 2 atom indices
            // This will ensure the derivative we want is in the buffer.
            std::vector<int> desired_shell_atoms; 
            for (int i = 0; i < deriv_order; i++){
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
            for (int i = 0; i < deriv_order; i++){
                shell_derivative.push_back(3 * desired_shell_atoms[i] + desired_coordinates[i]);
            }

            // Now we must convert our multidimensional shell_derivative index into a one-dimensional buffer index. 
            // We know how to do this since libint tells us what order they come in. The lookup arrays above map the multidim index to the buffer idx
            int buffer_idx;
            if (deriv_order == 1) { 
                buffer_idx = buffer_index_oei1d[shell_derivative[0]];
            }
            else if (deriv_order == 2) { 
                buffer_idx = buffer_index_oei2d[shell_derivative[0]][shell_derivative[1]];
            }
            else if (deriv_order == 3) { 
                buffer_idx = buffer_index_oei3d[shell_derivative[0]][shell_derivative[1]][shell_derivative[2]];
            }
            else if (deriv_order == 4) { 
                buffer_idx = buffer_index_oei4d[shell_derivative[0]][shell_derivative[1]][shell_derivative[2]][shell_derivative[3]];
            }

            auto ints_shellset = buf_vec[buffer_idx]; // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out

            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                    result[(bf1 + f1) * nbf + bf2 + f2 ] = ints_shellset[idx];
                }
            }
        }
    }
    return py::array(result.size(), result.data()); 
}

// Computes nuclear derivatives of potential energy integrals 
py::array potential_deriv(std::vector<int> deriv_vec) {
    assert(ncart == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");
    // Get order of differentiation
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Lookup arrays for mapping shell derivative index to buffer index 
    // Potential lookup arrays depend on atom size
    int dimensions = 6 + ncart;
    static const std::vector<int> buffer_index_potential1d = generate_1d_lookup(dimensions);
    static const std::vector<std::vector<int>> buffer_index_potential2d = generate_2d_lookup(dimensions);
    static const std::vector<std::vector<std::vector<int>>> buffer_index_potential3d = generate_3d_lookup(dimensions);
    static const std::vector<std::vector<std::vector<std::vector<int>>>> buffer_index_potential4d = generate_4d_lookup(dimensions);

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    // Potential integral derivative engine
    libint2::Engine v_engine(libint2::Operator::nuclear, obs.max_nprim(), obs.max_l(), deriv_order);
    v_engine.set_params(libint2::make_point_charges(atoms));
    const auto& buf_vec = v_engine.results(); // will point to computed shell sets

    // Get size of potential derivative array and allocate 
    size_t length = nbf * nbf;
    std::vector<double> result(length);

    for(auto s1 = 0; s1 != obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];     // Index of first basis function in shell 1
        auto atom1 = shell2atom[s1]; // Atom index of shell 1
        auto n1 = obs[s1].size();    // number of basis functions in shell 1
        for(auto s2 = 0; s2 != obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom[s2]; // Atom index of shell 2
            auto n2 = obs[s2].size();    // number of basis functions in shell 2

            // Create list of atom indices corresponding to each shell. Libint uses longs, so we will too.
            std::vector<long> shell_atom_index_list{atom1, atom2};

            // Initialize 2d vector, with DERIV_ORDER subvectors
            // Each subvector contains index candidates which are possible choices for each partial derivative operator
            // In other words, indices looks like { {choices for first deriv operator} {choices for second deriv op} {third} ...}
            // The cartesian product of these subvectors gives all combos that need to be summed to form total nuclear derivative of integrals
            std::vector<std::vector<int>> indices; 
            for (int i = 0; i < deriv_order; i++){
                std::vector<int> new_vec;
                indices.push_back(new_vec);
            }

            // For every desired atom derivative, check shell and nuclear indices for a match, add it to subvector for that derivative
            // Add in the coordinate index 0,1,2 (x,y,z) in desired coordinates and offset the index appropriately.
            for (int j = 0; j < desired_atom_indices.size(); j++){
                int desired_atom_idx = desired_atom_indices[j];
                // Shell indices
                for (int i = 0; i < 2; i++){
                    int atom_idx = shell_atom_index_list[i];
                    if (atom_idx == desired_atom_idx) { 
                        int tmp = 3 * i + desired_coordinates[j];
                        indices[j].push_back(tmp);
                    }
                }
                
                for (int i = 0; i < natom; i++){
                    // i = shell_atom_index_list[i];
                    if (i == desired_atom_idx) { 
                        int offset_i = i + 2;
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
                for (int i = 0; i < index_combos.size(); i++){
                    int idx1 = index_combos[i][0];
                    buffer_indices.push_back(buffer_index_potential1d[idx1]);
                }
            }
            else if (deriv_order == 2){
                for (int i = 0; i < index_combos.size(); i++){
                    int idx1 = index_combos[i][0];
                    int idx2 = index_combos[i][1];
                    buffer_indices.push_back(buffer_index_potential2d[idx1][idx2]);
                }
            }
            else if (deriv_order == 3){
                for (int i = 0; i < index_combos.size(); i++){
                    int idx1 = index_combos[i][0];
                    int idx2 = index_combos[i][1];
                    int idx3 = index_combos[i][2];
                    buffer_indices.push_back(buffer_index_potential3d[idx1][idx2][idx3]);
                }
            }
            else if (deriv_order == 4){
                for (int i = 0; i < index_combos.size(); i++){
                    int idx1 = index_combos[i][0];
                    int idx2 = index_combos[i][1];
                    int idx3 = index_combos[i][2];
                    int idx4 = index_combos[i][3];
                    buffer_indices.push_back(buffer_index_potential4d[idx1][idx2][idx3][idx4]);
                }
            }

            // Loop over every buffer index and accumulate for every shell set.
            for(auto i = 0; i < buffer_indices.size(); ++i) {
                auto ints_shellset = buf_vec[buffer_indices[i]];
                if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                    for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
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

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    assert(ncart == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");

    // ERI derivative integral engine
    libint2::Engine eri_engine(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l(), deriv_order);
    const auto& buf_vec = eri_engine.results(); // will point to computed shell sets
    size_t length = nbf * nbf * nbf * nbf;
    std::vector<double> result(length);

    for(auto s1 = 0; s1 != obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];     // Index of first basis function in shell 1
        auto atom1 = shell2atom[s1]; // Atom index of shell 1
        auto n1 = obs[s1].size();    // number of basis functions in shell 1
        for(auto s2 = 0; s2 != obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom[s2]; // Atom index of shell 2
            auto n2 = obs[s2].size();    // number of basis functions in shell 2
            for(auto s3 = 0; s3 != obs.size(); ++s3) {
                auto bf3 = shell2bf[s3];     // Index of first basis function in shell 3
                auto atom3 = shell2atom[s3]; // Atom index of shell 3
                auto n3 = obs[s3].size();    // number of basis functions in shell 3
                for(auto s4 = 0; s4 != obs.size(); ++s4) {
                    auto bf4 = shell2bf[s4];     // Index of first basis function in shell 4
                    auto atom4 = shell2atom[s4]; // Atom index of shell 4
                    auto n4 = obs[s4].size();    // number of basis functions in shell 4

                    // If the atoms are the same we ignore it as the derivatives will be zero.
                    if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
                    // Ensure all desired_atoms correspond to at least one shell atom to ensure desired derivative exists. else, skip this shell quartet.
                    bool atoms_not_present = false;
                    for (int i = 0; i < deriv_order; i++){
                        if (atom1 == desired_atom_indices[i]) continue; 
                        else if (atom2 == desired_atom_indices[i]) continue;
                        else if (atom3 == desired_atom_indices[i]) continue;
                        else if (atom4 == desired_atom_indices[i]) continue;
                        else {atoms_not_present = true; break;}
                    }
                    if (atoms_not_present) continue;

                    // Create list of atom indices corresponding to each shell. Libint uses longs, so we will too.
                    std::vector<long> shell_atom_index_list{atom1, atom2, atom3, atom4};

                    // Initialize 2d vector, with DERIV_ORDER subvectors
                    // Each subvector contains index candidates which are possible choices for each partial derivative operator
                    // In other words, indices looks like { {choices for first deriv operator} {choices for second deriv op} {third} ...}
                    // The cartesian product of these subvectors gives all combos that need to be summed to form total nuclear derivative of integrals
                    std::vector<std::vector<int>> indices;
                    for (int i = 0; i < deriv_order; i++){
                        std::vector<int> new_vec;
                        indices.push_back(new_vec);
                    }
                
                    // For every desired atom derivative, check shell indices for a match, add it to subvector for that derivative
                    // Add in the coordinate index 0,1,2 (x,y,z) in desired coordinates and offset the index appropriately.
                    for (int j = 0; j < desired_atom_indices.size(); j++){
                        int desired_atom_idx = desired_atom_indices[j];
                        // Shell indices
                        for (int i = 0; i < 4; i++){
                            int atom_idx = shell_atom_index_list[i];
                            if (atom_idx == desired_atom_idx) {
                                int tmp = 3 * i + desired_coordinates[j];
                                indices[j].push_back(tmp);
                            }
                        }
                    }
                    
                    // Now indices is a vector of vectors, where each subvector is your choices for the first derivative operator, second, third, etc
                    // and the total number of subvectors is the order of differentiation
                    // Now we want all combinations where we pick exactly one index from each subvector.
                    // This is achievable through a cartesian product 
                    std::vector<std::vector<int>> index_combos = cartesian_product(indices);

                    // Now create buffer_indices from these index combos using lookup array
                    std::vector<int> buffer_indices;
                    if (deriv_order == 1){ 
                        for (int i = 0; i < index_combos.size(); i++){
                            int idx1 = index_combos[i][0];
                            buffer_indices.push_back(buffer_index_eri1d[idx1]);
                        }
                    }
                    else if (deriv_order == 2){ 
                        for (int i = 0; i < index_combos.size(); i++){
                            int idx1 = index_combos[i][0];
                            int idx2 = index_combos[i][1];
                            buffer_indices.push_back(buffer_index_eri2d[idx1][idx2]);
                        }
                    }
                    else if (deriv_order == 3){ 
                        for (int i = 0; i < index_combos.size(); i++){
                            int idx1 = index_combos[i][0];
                            int idx2 = index_combos[i][1];
                            int idx3 = index_combos[i][2];
                            buffer_indices.push_back(buffer_index_eri3d[idx1][idx2][idx3]);
                        }
                    }
                    else if (deriv_order == 4){ 
                        for (int i = 0; i < index_combos.size(); i++){
                            int idx1 = index_combos[i][0];
                            int idx2 = index_combos[i][1];
                            int idx3 = index_combos[i][2];
                            int idx4 = index_combos[i][3];
                            buffer_indices.push_back(buffer_index_eri4d[idx1][idx2][idx3][idx4]);
                        }
                    }

                    // If we made it this far, the shell derivative we want is contained in the buffer. 
                    eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]); // Compute shell set, fills buf_vec

                    for(auto i = 0; i<buffer_indices.size(); ++i) {
                        auto ints_shellset = buf_vec[buffer_indices[i]];
                        if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            size_t offset_1 = (bf1 + f1) * nbf * nbf * nbf;
                            for(auto f2 = 0; f2 != n2; ++f2) {
                                size_t offset_2 = (bf2 + f2) * nbf * nbf;
                                for(auto f3 = 0; f3 != n3; ++f3) {
                                    size_t offset_3 = (bf3 + f3) * nbf;
                                    for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
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
    // This is not the bottleneck
    return py::array(result.size(), result.data()); // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042 there's a workaround
}

// The following function writes all overlap, kinetic, and potential derivatives up to `max_deriv_order` to disk
// HDF5 File Name: oei_derivs.h5 
//      HDF5 Dataset names within the file:
//      overlap_deriv1 
//          shape (nbf,nbf,n_unique_1st_derivs)
//      overlap_deriv2 
//          shape (nbf,nbf,n_unique_2nd_derivs)
//      overlap_deriv3 
//          shape (nbf,nbf,n_unique_3rd_derivs)
//      ...
//      kinetic_deriv1 
//          shape (nbf,nbf,n_unique_1st_derivs)
//      kinetic_deriv2 
//          shape (nbf,nbf,n_unique_2nd_derivs)
//      kinetic_deriv3 
//          shape (nbf,nbf,n_unique_3rd_derivs)
//      ...
//      potential_deriv1 
//          shape (nbf,nbf,n_unique_1st_derivs)
//      potential_deriv2 
//          shape (nbf,nbf,n_unique_2nd_derivs)
//      potential_deriv3 
//          shape (nbf,nbf,n_unique_3rd_derivs)
// The number of unique derivatives is essentially equal to the size of the generalized upper triangle of the derivative tensor.
void oei_deriv_disk(int max_deriv_order) {
    std::cout << "Writing one-electron integral derivative tensors up to order " << max_deriv_order << " to disk...";
    long total_deriv_slices = 0;
    for (int i = 1; i <= max_deriv_order; i++){
        total_deriv_slices += how_many_derivs(natom, i);
        }

    // Create H5 File and prepare to fill with 0.0's
    const H5std_string file_name("oei_derivs.h5");
    H5File* file = new H5File(file_name,H5F_ACC_TRUNC);
    double fillvalue = 0.0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);

    for (int deriv_order = 1; deriv_order <= max_deriv_order; deriv_order++){
        // how many shell derivatives in the Libint buffer for overlap/kinetic integrals
        // how many shell and operator derivatives for potential integrals 
        int nshell_derivs = how_many_derivs(2, deriv_order);
        int nshell_derivs_potential = how_many_derivs(2, deriv_order, natom);
        // how many unique cartesian nuclear derivatives (e.g., so we only save one of d^2/dx1dx2 and d^2/dx2dx1, etc)
        unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);
        // Create mappings from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
        // Overlap and kinetic have different mappings than potential since potential has more elements in the buffer 
        const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(6, deriv_order);
        // Potential integrals buffer is flattened upper triangle of (6 + NCART) dimensional deriv_order tensor
        int dimensions = 6 + ncart;
        const std::vector<std::vector<int>> potential_buffer_multidim_lookup = generate_multi_index_lookup(dimensions, deriv_order);

        // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
        const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

        // Define engines and buffers
        libint2::Engine overlap_engine(libint2::Operator::overlap,obs.max_nprim(),obs.max_l(),deriv_order);
        const auto& overlap_buffer = overlap_engine.results(); 
        libint2::Engine kinetic_engine(libint2::Operator::kinetic,obs.max_nprim(),obs.max_l(),deriv_order);
        const auto& kinetic_buffer = kinetic_engine.results(); 
        libint2::Engine potential_engine(libint2::Operator::nuclear,obs.max_nprim(),obs.max_l(),deriv_order);
        potential_engine.set_params(libint2::make_point_charges(atoms));
        const auto& potential_buffer = potential_engine.results(); 

        // Define HDF5 dataset names
        const H5std_string overlap_dset_name("overlap_deriv" + std::to_string(deriv_order));
        const H5std_string kinetic_dset_name("kinetic_deriv" + std::to_string(deriv_order));
        const H5std_string potential_dset_name("potential_deriv" + std::to_string(deriv_order));

        // Define rank and dimensions of data that will be written to the file
        hsize_t file_dims[] = {nbf, nbf, nderivs_triu};
        DataSpace fspace(3, file_dims);
        // Create dataset for each integral type and write 0.0's into the file 
        DataSet* overlap_dataset = new DataSet(file->createDataSet(overlap_dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        DataSet* kinetic_dataset = new DataSet(file->createDataSet(kinetic_dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        DataSet* potential_dataset = new DataSet(file->createDataSet(potential_dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        hsize_t stride[3] = {1, 1, 1}; // stride and block can be used to 
        hsize_t block[3] = {1, 1, 1};  // add values to multiple places, useful if symmetry ever used.
        hsize_t zerostart[3] = {0, 0, 0};

        for(auto s1 = 0; s1 != obs.size(); ++s1) {
            auto bf1 = shell2bf[s1];  // first basis function in first shell
            auto atom1 = shell2atom[s1]; // Atom index of shell 1
            auto n1 = obs[s1].size(); // number of basis functions in first shell
            for(auto s2 = 0; s2 != obs.size(); ++s2) {
                auto bf2 = shell2bf[s2];  // first basis function in second shell
                auto atom2 = shell2atom[s2]; // Atom index of shell 2
                auto n2 = obs[s2].size(); // number of basis functions in second shell
                std::vector<long> shell_atom_index_list{atom1,atom2};

                overlap_engine.compute(obs[s1], obs[s2]);
                kinetic_engine.compute(obs[s1], obs[s2]);
                potential_engine.compute(obs[s1], obs[s2]);

                // Define shell set slabs
                double overlap_shellset_slab [n1][n2][nderivs_triu] = {};
                double kinetic_shellset_slab [n1][n2][nderivs_triu] = {};
                double potential_shellset_slab [n1][n2][nderivs_triu] = {};
                
                // Loop over every possible unique nuclear cartesian derivative index (flattened upper triangle)
                // For 1st derivatives of 2 atom system, this is 6. 2nd derivatives of 2 atom system: 21, etc
                for(int nuc_idx = 0; nuc_idx < nderivs_triu; ++nuc_idx) {
                    // Look up multidimensional cartesian derivative index
                    auto multi_cart_idx = cart_multidim_lookup[nuc_idx];
                    // For overlap/kinetic and potential sepearately, create a vector of vectors called `indices`, where each subvector
                    // is your possible choices for the first derivative operator, second, third, etc and the total number of subvectors is order of differentiation
                    // What follows fills these indices
                    std::vector<std::vector<int>> indices(deriv_order, std::vector<int> (0,0));
                    std::vector<std::vector<int>> potential_indices(deriv_order, std::vector<int> (0,0));
                
                    // Loop over each cartesian coordinate index which we are differentiating wrt for this nuclear cartesian derivative index
                    // and check to see if it is present in the shell duet, and where it is present in the potential operator 
                    for (int j = 0; j < multi_cart_idx.size(); j++){
                        int desired_atom_idx = multi_cart_idx[j] / 3;
                        int desired_coord = multi_cart_idx[j] % 3;
                        // Loop over shell indices
                        for (int i = 0; i < 2; i++){
                            int atom_idx = shell_atom_index_list[i];
                            if (atom_idx == desired_atom_idx) {
                                int tmp = 3 * i + desired_coord;
                                indices[j].push_back(tmp);
                                potential_indices[j].push_back(tmp);
                            }
                        }
                        // Now for potentials only, loop over each atom in molecule, and if this derivative
                        // differentiates wrt that atom, we also need to collect that index.
                        for (int i = 0; i < natom; i++){
                            if (i == desired_atom_idx) {
                                int offset_i = i + 2;
                                int tmp = 3 * offset_i + desired_coord;
                                potential_indices[j].push_back(tmp);
                            }
                        }
                    }
                    // Now indices is a vector of vectors, where each subvector is your choices for the first derivative operator, second, third, etc
                    // and the total number of subvectors is the order of differentiation
                    // Now we want all combinations where we pick exactly one index from each subvector.
                    // This is achievable through a cartesian product 
                    std::vector<std::vector<int>> index_combos = cartesian_product(indices);
                    std::vector<std::vector<int>> potential_index_combos = cartesian_product(potential_indices);
                    std::vector<int> buffer_indices;
                    std::vector<int> potential_buffer_indices;
                    // Overlap/Kinetic integrals: collect needed buffer indices which we need to sum for this nuclear cartesian derivative
                    for (auto vec : index_combos)  {
                        std::sort(vec.begin(), vec.end());
                        int buf_idx = 0;
                        auto it = lower_bound(buffer_multidim_lookup.begin(), buffer_multidim_lookup.end(), vec);
                        if (it != buffer_multidim_lookup.end()) buf_idx = it - buffer_multidim_lookup.begin();
                        buffer_indices.push_back(buf_idx);
                    }
                    // Potential integrals: collect needed buffer indices which we need to sum for this nuclear cartesian derivative
                    for (auto vec : potential_index_combos)  {
                        std::sort(vec.begin(), vec.end());
                        int buf_idx = 0;
                        auto it = lower_bound(potential_buffer_multidim_lookup.begin(), potential_buffer_multidim_lookup.end(), vec);
                        if (it != potential_buffer_multidim_lookup.end()) buf_idx = it - potential_buffer_multidim_lookup.begin();
                        potential_buffer_indices.push_back(buf_idx);
                    }

                    // Loop over shell block for each buffer index which contributes to this derivative
                    // Overlap and Kinetic
                    for(auto i = 0; i < buffer_indices.size(); ++i) {
                        auto overlap_shellset = overlap_buffer[buffer_indices[i]];
                        auto kinetic_shellset = kinetic_buffer[buffer_indices[i]];
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                                overlap_shellset_slab[f1][f2][nuc_idx] += overlap_shellset[idx];
                                kinetic_shellset_slab[f1][f2][nuc_idx] += kinetic_shellset[idx];
                            }
                        }
                    }
                    // Potential
                    for(auto i = 0; i < potential_buffer_indices.size(); ++i) {
                        auto potential_shellset = potential_buffer[potential_buffer_indices[i]];
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                                potential_shellset_slab[f1][f2][nuc_idx] += potential_shellset[idx];
                            }
                        }
                    }
                } // Unique nuclear cartesian derivative indices loop

                // Now write this shell set slab to HDF5 file
                // Create file space hyperslab, defining where to write data to in file
                hsize_t count[3] = {n1, n2, nderivs_triu};
                hsize_t start[3] = {bf1, bf2, 0};
                fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
                // Create dataspace defining for memory dataset to write to file
                hsize_t mem_dims[] = {n1, n2, nderivs_triu};
                DataSpace mspace(3, mem_dims);
                mspace.selectHyperslab(H5S_SELECT_SET, count, zerostart, stride, block);
                // Write buffer data 'shellset_slab' with data type double from memory dataspace `mspace` to file dataspace `fspace`
                overlap_dataset->write(overlap_shellset_slab, PredType::NATIVE_DOUBLE, mspace, fspace);
                kinetic_dataset->write(kinetic_shellset_slab, PredType::NATIVE_DOUBLE, mspace, fspace);
                potential_dataset->write(potential_shellset_slab, PredType::NATIVE_DOUBLE, mspace, fspace);
            }
        } // shell duet loops
    // Delete datasets for this derivative order
    delete overlap_dataset;
    delete kinetic_dataset;
    delete potential_dataset;
    } // deriv order loop
// close the file
delete file;
std::cout << " done" << std::endl;
} //oei_deriv_disk 


// Writes all ERI's up to `max_deriv_order` to disk.
// HDF5 File Name: eri_derivs.h5 
//      HDF5 Dataset names within the file:
//      eri_deriv1 
//          shape (nbf,nbf,nbf,nbf,n_unique_1st_derivs)
//      eri_deriv2
//          shape (nbf,nbf,nbf,nbf,n_unique_2nd_derivs)
//      eri_deriv3
//          shape (nbf,nbf,nbf,nbf,n_unique_3rd_derivs)
//      ...
void eri_deriv_disk(int max_deriv_order) { 
    std::cout << "Writing two-electron integral derivative tensors up to order " << max_deriv_order << " to disk...";
    const H5std_string file_name("eri_derivs.h5");
    H5File* file = new H5File(file_name,H5F_ACC_TRUNC);
    double fillvalue = 0.0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);

    // Check to make sure you are not flooding the disk.
    long total_deriv_slices = 0;
    for (int i = 1; i <= max_deriv_order; i++){
        total_deriv_slices += how_many_derivs(natom, i);
        }
    double check = (nbf * nbf * nbf * nbf * total_deriv_slices * 8) * (1e-9);
    assert(check < 10 && "Total disk space required for ERI's exceeds 10 GB. Increase threshold and recompile to proceed.");

    for (int deriv_order = 1; deriv_order <= max_deriv_order; deriv_order++){
        // Number of unique shell derivatives output by libint (number of indices in buffer)
        int nshell_derivs = how_many_derivs(4, deriv_order);
        // Number of unique nuclear derivatives of ERI's
        unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

        // Create mapping from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
        const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);

        // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
        const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

        // Libint engine for computing shell quartet derivatives
        libint2::Engine eri_engine(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l(), deriv_order);
        const auto& eri_buffer = eri_engine.results(); // will point to computed shell sets

        // Define HDF5 dataset name
        const H5std_string eri_dset_name("eri_deriv" + std::to_string(deriv_order));
        hsize_t file_dims[] = {nbf, nbf, nbf, nbf, nderivs_triu};
        DataSpace fspace(5, file_dims);
        // Create dataset for each integral type and write 0.0's into the file 
        DataSet* eri_dataset = new DataSet(file->createDataSet(eri_dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        hsize_t stride[5] = {1, 1, 1, 1, 1}; // stride and block can be used to 
        hsize_t block[5] = {1, 1, 1, 1, 1};  // add values to multiple places, useful if symmetry ever used.
        hsize_t zerostart[5] = {0, 0, 0, 0, 0};

        // Begin shell quartet loops
        for(auto s1 = 0; s1 != obs.size(); ++s1) {
            auto bf1 = shell2bf[s1];     // Index of first basis function in shell 1
            auto atom1 = shell2atom[s1]; // Atom index of shell 1
            auto n1 = obs[s1].size();    // number of basis functions in shell 1
            for(auto s2 = 0; s2 != obs.size(); ++s2) {
                auto bf2 = shell2bf[s2];     // Index of first basis function in shell 2
                auto atom2 = shell2atom[s2]; // Atom index of shell 2
                auto n2 = obs[s2].size();    // number of basis functions in shell 2
                for(auto s3 = 0; s3 != obs.size(); ++s3) {
                    auto bf3 = shell2bf[s3];     // Index of first basis function in shell 3
                    auto atom3 = shell2atom[s3]; // Atom index of shell 3
                    auto n3 = obs[s3].size();    // number of basis functions in shell 3
                    for(auto s4 = 0; s4 != obs.size(); ++s4) {
                        auto bf4 = shell2bf[s4];     // Index of first basis function in shell 4
                        auto atom4 = shell2atom[s4]; // Atom index of shell 4
                        auto n4 = obs[s4].size();    // number of basis functions in shell 4

                        if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
                        std::vector<long> shell_atom_index_list{atom1, atom2, atom3, atom4};

                        eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]); // Compute shell set

                        // Define shell set slab, with extra dimension for unique derivatives, initialized with 0.0's
                        double eri_shellset_slab [n1][n2][n3][n4][nderivs_triu] = {};
                        // Loop over every possible unique nuclear cartesian derivative index (flattened upper triangle)
                        for(int nuc_idx = 0; nuc_idx < nderivs_triu; ++nuc_idx) {
                            // Look up multidimensional cartesian derivative index
                            auto multi_cart_idx = cart_multidim_lookup[nuc_idx];
    
                            std::vector<std::vector<int>> indices(deriv_order, std::vector<int> (0,0));
    
                            // Find out which 
                            for (int j = 0; j < multi_cart_idx.size(); j++){
                                int desired_atom_idx = multi_cart_idx[j] / 3;
                                int desired_coord = multi_cart_idx[j] % 3;
                                for (int i = 0; i < 4; i++){
                                    int atom_idx = shell_atom_index_list[i];
                                    if (atom_idx == desired_atom_idx) {
                                        int tmp = 3 * i + desired_coord;
                                        indices[j].push_back(tmp);
                                    }
                                }
                            }

                            // Now indices is a vector of vectors, where each subvector is your choices for the first derivative operator, second, third, etc
                            // and the total number of subvectors is the order of differentiation
                            // Now we want all combinations where we pick exactly one index from each subvector.
                            // This is achievable through a cartesian product 
                            std::vector<std::vector<int>> index_combos = cartesian_product(indices);
                            std::vector<int> buffer_indices;
                            
                            // Binary search to find 1d buffer index from multidimensional shell derivative index in `index_combos`
                            //for (auto vec : index_combos)  {
                            //    std::sort(vec.begin(), vec.end());
                            //    int buf_idx = 0;
                            //    // buffer_multidim_lookup
                            //    auto it = lower_bound(buffer_multidim_lookup.begin(), buffer_multidim_lookup.end(), vec);
                            //    if (it != buffer_multidim_lookup.end()) buf_idx = it - buffer_multidim_lookup.begin();
                            //    buffer_indices.push_back(buf_idx);
                            //}
                            // Eventually, if you stop using lookup arrays, use above implementation, but these are sitting around so might as well use them 
                            for (auto vec : index_combos)  {
                                if (deriv_order == 1) buffer_indices.push_back(buffer_index_eri1d[vec[0]]);
                                else if (deriv_order == 2) buffer_indices.push_back(buffer_index_eri2d[vec[0]][vec[1]]);
                                else if (deriv_order == 3) buffer_indices.push_back(buffer_index_eri3d[vec[0]][vec[1]][vec[2]]);
                                else if (deriv_order == 4) buffer_indices.push_back(buffer_index_eri4d[vec[0]][vec[1]][vec[2]][vec[3]]);
                            }

                            // Loop over shell block, keeping a total count idx for the size of shell set
                            for(auto i = 0; i < buffer_indices.size(); ++i) {
                                auto eri_shellset = eri_buffer[buffer_indices[i]];
                                if (eri_shellset == nullptr) continue;
                                for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                                    for(auto f2 = 0; f2 != n2; ++f2) {
                                        for(auto f3 = 0; f3 != n3; ++f3) {
                                            for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                                eri_shellset_slab[f1][f2][f3][f4][nuc_idx] += eri_shellset[idx];
                                            }
                                        }
                                    }
                                }
                            }
                        } // For every nuc_idx 0, nderivs_triu
                        // Now write this shell set slab to HDF5 file
                        hsize_t count[5] = {n1, n2, n3, n4, nderivs_triu};
                        hsize_t start[5] = {bf1, bf2, bf3, bf4, 0};
                        fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
                        // Create dataspace defining for memory dataset to write to file
                        hsize_t mem_dims[] = {n1, n2, n3, n4, nderivs_triu};
                        DataSpace mspace(5, mem_dims);
                        mspace.selectHyperslab(H5S_SELECT_SET, count, zerostart, stride, block);
                        // Write buffer data 'shellset_slab' with data type double from memory dataspace `mspace` to file dataspace `fspace`
                        eri_dataset->write(eri_shellset_slab, PredType::NATIVE_DOUBLE, mspace, fspace);
                    }
                }
            }
        } // shell quartet loops
    // Close the dataset for this derivative order
    delete eri_dataset;
    } // deriv order loop 
// Close the file
delete file;
std::cout << " done" << std::endl;
} // eri_deriv_disk function

// Computes a single 'deriv_order' derivative tensor of overlap integrals, keeps everything in core memory
py::array overlap_deriv_core(int deriv_order) {
    int nshell_derivs = how_many_derivs(2, deriv_order);
    // how many unique cartesian nuclear derivatives (e.g., so we only save one of d^2/dx1dx2 and d^2/dx2dx1, etc)
    unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

    // Create mappings from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
    const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(6, deriv_order);

    // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
    const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

    // Define engines and buffers
    libint2::Engine overlap_engine(libint2::Operator::overlap, obs.max_nprim(), obs.max_l(), deriv_order);
    const auto& overlap_buffer = overlap_engine.results();

    size_t length = nbf * nbf * nderivs_triu;
    std::vector<double> result(length);

    for(auto s1 = 0; s1 != obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];     // Index of first basis function in shell 1
        auto atom1 = shell2atom[s1]; // Atom index of shell 1
        auto n1 = obs[s1].size();    // number of basis functions in shell 1
        for(auto s2 = 0; s2 != obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom[s2]; // Atom index of shell 2
            auto n2 = obs[s2].size();    // number of basis functions in shell 2
            std::vector<long> shell_atom_index_list{atom1, atom2};

            overlap_engine.compute(obs[s1], obs[s2]);

            // Loop over every possible unique nuclear cartesian derivative index (flattened upper triangle)
            // For 1st derivatives of 2 atom system, this is 6. 2nd derivatives of 2 atom system: 21, etc
            for(int nuc_idx = 0; nuc_idx < nderivs_triu; ++nuc_idx) {
                size_t offset_nuc_idx = nuc_idx * nbf * nbf;
                // Look up multidimensional cartesian derivative index
                auto multi_cart_idx = cart_multidim_lookup[nuc_idx];
                // Create a vector of vectors called `indices`, where each subvector is your possible choices
                // for the first derivative operator, second, third, etc and the total number of subvectors is order of differentiation
                // What follows fills these indices
                std::vector<std::vector<int>> indices(deriv_order, std::vector<int> (0,0));

                // Loop over each cartesian coordinate index which we are differentiating wrt for this nuclear cartesian derivative index
                // and check to see if it is present in the shell duet
                for (int j = 0; j < multi_cart_idx.size(); j++){
                    int desired_atom_idx = multi_cart_idx[j] / 3;
                    int desired_coord = multi_cart_idx[j] % 3;
                    // Loop over shell indices
                    for (int i = 0; i < 2; i++){
                        int atom_idx = shell_atom_index_list[i];
                        if (atom_idx == desired_atom_idx) {
                            int tmp = 3 * i + desired_coord;
                            indices[j].push_back(tmp);
                        }
                    }
                }

                // Now indices is a vector of vectors, where each subvector is your choices for the first derivative operator, second, third, etc
                // and the total number of subvectors is the order of differentiation
                // Now we want all combinations where we pick exactly one index from each subvector.
                // This is achievable through a cartesian product
                std::vector<std::vector<int>> index_combos = cartesian_product(indices);
                std::vector<int> buffer_indices;
                // Collect needed buffer indices which we need to sum for this nuclear cartesian derivative
                for (auto vec : index_combos)  {
                    std::sort(vec.begin(), vec.end());
                    int buf_idx = 0;
                    auto it = lower_bound(buffer_multidim_lookup.begin(), buffer_multidim_lookup.end(), vec);
                    if (it != buffer_multidim_lookup.end()) buf_idx = it - buffer_multidim_lookup.begin();
                    buffer_indices.push_back(buf_idx);
                }

                // Loop over shell block for each buffer index which contributes to this derivative
                // Overlap and Kinetic
                for(auto i = 0; i < buffer_indices.size(); ++i) {
                    auto overlap_shellset = overlap_buffer[buffer_indices[i]];
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                            result[(bf1 + f1) * nbf + bf2 + f2 + offset_nuc_idx] += overlap_shellset[idx];
                        }
                    }
                }
            } // Unique nuclear cartesian derivative indices loop
        }
    } // shell duet loops
    return py::array(result.size(), result.data()); // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042 there's a workaround
} // overlap_deriv_core function

// Computes a single 'deriv_order' derivative tensor of kinetic integrals, keeps everything in core memory
py::array kinetic_deriv_core(int deriv_order) {
    int nshell_derivs = how_many_derivs(2, deriv_order);
    // how many unique cartesian nuclear derivatives (e.g., so we only save one of d^2/dx1dx2 and d^2/dx2dx1, etc)
    unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

    // Create mappings from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
    const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(6, deriv_order);

    // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
    const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

    // Define engines and buffers
    libint2::Engine kinetic_engine(libint2::Operator::kinetic, obs.max_nprim(), obs.max_l(), deriv_order);
    const auto& kinetic_buffer = kinetic_engine.results();

    size_t length = nbf * nbf * nderivs_triu;
    std::vector<double> result(length);

    for(auto s1 = 0; s1 != obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];     // Index of first basis function in shell 1
        auto atom1 = shell2atom[s1]; // Atom index of shell 1
        auto n1 = obs[s1].size();    // number of basis functions in shell 1
        for(auto s2 = 0; s2 != obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom[s2]; // Atom index of shell 2
            auto n2 = obs[s2].size();    // number of basis functions in shell 2
            std::vector<long> shell_atom_index_list{atom1, atom2};

            kinetic_engine.compute(obs[s1], obs[s2]);

            // Loop over every possible unique nuclear cartesian derivative index (flattened upper triangle)
            // For 1st derivatives of 2 atom system, this is 6. 2nd derivatives of 2 atom system: 21, etc
            for(int nuc_idx = 0; nuc_idx < nderivs_triu; ++nuc_idx) {
                size_t offset_nuc_idx = nuc_idx * nbf * nbf;
                // Look up multidimensional cartesian derivative index
                auto multi_cart_idx = cart_multidim_lookup[nuc_idx];
                // Create a vector of vectors called `indices`, where each subvector is your possible choices
                // for the first derivative operator, second, third, etc and the total number of subvectors is order of differentiation
                // What follows fills these indices
                std::vector<std::vector<int>> indices(deriv_order, std::vector<int> (0,0));

                // Loop over each cartesian coordinate index which we are differentiating wrt for this nuclear cartesian derivative index
                // and check to see if it is present in the shell duet
                for (int j = 0; j < multi_cart_idx.size(); j++){
                    int desired_atom_idx = multi_cart_idx[j] / 3;
                    int desired_coord = multi_cart_idx[j] % 3;
                    // Loop over shell indices
                    for (int i = 0; i < 2; i++){
                        int atom_idx = shell_atom_index_list[i];
                        if (atom_idx == desired_atom_idx) {
                            int tmp = 3 * i + desired_coord;
                            indices[j].push_back(tmp);
                        }
                    }
                }

                // Now indices is a vector of vectors, where each subvector is your choices for the first derivative operator, second, third, etc
                // and the total number of subvectors is the order of differentiation
                // Now we want all combinations where we pick exactly one index from each subvector.
                // This is achievable through a cartesian product
                std::vector<std::vector<int>> index_combos = cartesian_product(indices);
                std::vector<int> buffer_indices;
                // Collect needed buffer indices which we need to sum for this nuclear cartesian derivative
                for (auto vec : index_combos)  {
                    std::sort(vec.begin(), vec.end());
                    int buf_idx = 0;
                    auto it = lower_bound(buffer_multidim_lookup.begin(), buffer_multidim_lookup.end(), vec);
                    if (it != buffer_multidim_lookup.end()) buf_idx = it - buffer_multidim_lookup.begin();
                    buffer_indices.push_back(buf_idx);
                }

                // Loop over shell block for each buffer index which contributes to this derivative
                // Overlap and Kinetic
                for(auto i = 0; i < buffer_indices.size(); ++i) {
                    auto kinetic_shellset = kinetic_buffer[buffer_indices[i]];
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                            result[(bf1 + f1) * nbf + bf2 + f2 + offset_nuc_idx] += kinetic_shellset[idx];
                        }
                    }
                }
            } // Unique nuclear cartesian derivative indices loop
        }
    } // shell duet loops
    return py::array(result.size(), result.data()); // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042 there's a workaround
} // kinetic_deriv_core function

// Computes a single 'deriv_order' derivative tensor of potential integrals, keeps everything in core memory
py::array potential_deriv_core(int deriv_order) {
    int nshell_derivs = how_many_derivs(2, deriv_order, natom);
    // how many unique cartesian nuclear derivatives (e.g., so we only save one of d^2/dx1dx2 and d^2/dx2dx1, etc)
    unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

    // Potential integrals buffer is flattened upper triangle of (6 + NCART) dimensional deriv_order tensor
    int dimensions = 6 + ncart;
    const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(dimensions, deriv_order);

    // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
    const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

    // Define engines and buffers
    libint2::Engine potential_engine(libint2::Operator::nuclear, obs.max_nprim(), obs.max_l(), deriv_order);
    potential_engine.set_params(libint2::make_point_charges(atoms));
    const auto& potential_buffer = potential_engine.results();

    size_t length = nbf * nbf * nderivs_triu;
    std::vector<double> result(length);

    for(auto s1 = 0; s1 != obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];     // Index of first basis function in shell 1
        auto atom1 = shell2atom[s1]; // Atom index of shell 1
        auto n1 = obs[s1].size();    // number of basis functions in shell 1
        for(auto s2 = 0; s2 != obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom[s2]; // Atom index of shell 2
            auto n2 = obs[s2].size();    // number of basis functions in shell 2
            std::vector<long> shell_atom_index_list{atom1, atom2};

            potential_engine.compute(obs[s1], obs[s2]);

            // Loop over every possible unique nuclear cartesian derivative index (flattened upper triangle)
            // For 1st derivatives of 2 atom system, this is 6. 2nd derivatives of 2 atom system: 21, etc
            for(int nuc_idx = 0; nuc_idx < nderivs_triu; ++nuc_idx) {
                size_t offset_nuc_idx = nuc_idx * nbf * nbf;
                // Look up multidimensional cartesian derivative index
                auto multi_cart_idx = cart_multidim_lookup[nuc_idx];
                // Create a vector of vectors called `indices`, where each subvector is your possible choices
                // for the first derivative operator, second, third, etc and the total number of subvectors is order of differentiation
                // What follows fills these indices
                std::vector<std::vector<int>> indices(deriv_order, std::vector<int> (0,0));

                // Loop over each cartesian coordinate index which we are differentiating wrt for this nuclear cartesian derivative index
                // and check to see if it is present in the shell duet, and where it is present in the potential operator
                for (int j = 0; j < multi_cart_idx.size(); j++){
                    int desired_atom_idx = multi_cart_idx[j] / 3;
                    int desired_coord = multi_cart_idx[j] % 3;
                    // Loop over shell indices
                    for (int i=0; i < 2; i++){
                        int atom_idx = shell_atom_index_list[i];
                        if (atom_idx == desired_atom_idx) {
                            int tmp = 3 * i + desired_coord;
                            indices[j].push_back(tmp);
                        }
                    }
                    // Loop over each atom in molecule, and if this derivative
                    // differentiates wrt that atom, we also need to collect that index.
                    for (int i = 0; i < natom; i++){
                        if (i == desired_atom_idx) {
                            int offset_i = i + 2;
                            int tmp = 3 * offset_i + desired_coord;
                            indices[j].push_back(tmp);
                        }
                    }
                }

                // Now indices is a vector of vectors, where each subvector is your choices for the first derivative operator, second, third, etc
                // and the total number of subvectors is the order of differentiation
                // Now we want all combinations where we pick exactly one index from each subvector.
                // This is achievable through a cartesian product
                std::vector<std::vector<int>> index_combos = cartesian_product(indices);
                std::vector<int> buffer_indices;
                // Collect needed buffer indices which we need to sum for this nuclear cartesian derivative
                for (auto vec : index_combos)  {
                    std::sort(vec.begin(), vec.end());
                    int buf_idx = 0;
                    auto it = lower_bound(buffer_multidim_lookup.begin(), buffer_multidim_lookup.end(), vec);
                    if (it != buffer_multidim_lookup.end()) buf_idx = it - buffer_multidim_lookup.begin();
                    buffer_indices.push_back(buf_idx);
                }

                for(auto i = 0; i < buffer_indices.size(); ++i) {
                    auto potential_shellset = potential_buffer[buffer_indices[i]];
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                            result[(bf1 + f1) * nbf + bf2 + f2  + offset_nuc_idx] += potential_shellset[idx];
                        }
                    }
                }
            } // Unique nuclear cartesian derivative indices loop
        }
    } // shell duet loops
    return py::array(result.size(), result.data()); // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042 there's a workaround
} // potential_deriv_core function

// Computes a single 'deriv_order' derivative tensor of electron repulsion integrals, keeps everything in core memory
py::array eri_deriv_core(int deriv_order) {
    // Number of unique shell derivatives output by libint (number of indices in buffer)
    int nshell_derivs = how_many_derivs(4, deriv_order);
    // Number of unique nuclear derivatives of ERI's
    unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

    // Create mapping from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
    const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);

    // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
    const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

    // Libint engine for computing shell quartet derivatives
    libint2::Engine eri_engine(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l(), deriv_order);
    const auto& eri_buffer = eri_engine.results(); // will point to computed shell sets

    size_t length = nbf * nbf * nbf * nbf * nderivs_triu;
    std::vector<double> result(length);

    // Begin shell quartet loops
    for(auto s1 = 0; s1 != obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];     // Index of first basis function in shell 1
        auto atom1 = shell2atom[s1]; // Atom index of shell 1
        auto n1 = obs[s1].size();    // number of basis functions in shell 1
        for(auto s2 = 0; s2 != obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom[s2]; // Atom index of shell 2
            auto n2 = obs[s2].size();    // number of basis functions in shell 2
            for(auto s3 = 0; s3 != obs.size(); ++s3) {
                auto bf3 = shell2bf[s3];     // Index of first basis function in shell 3
                auto atom3 = shell2atom[s3]; // Atom index of shell 3
                auto n3 = obs[s3].size();    // number of basis functions in shell 3
                for(auto s4 = 0; s4 != obs.size(); ++s4) {
                    auto bf4 = shell2bf[s4];     // Index of first basis function in shell 4
                    auto atom4 = shell2atom[s4]; // Atom index of shell 4
                    auto n4 = obs[s4].size();    // number of basis functions in shell 4

                    if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
                    std::vector<long> shell_atom_index_list{atom1, atom2, atom3, atom4};

                    eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]); // Compute shell set

                    // Loop over every possible unique nuclear cartesian derivative index (flattened upper triangle)
                    for(int nuc_idx = 0; nuc_idx < nderivs_triu; ++nuc_idx) {
                        size_t offset_nuc_idx = nuc_idx * nbf * nbf * nbf * nbf;

                        // Look up multidimensional cartesian derivative index
                        auto multi_cart_idx = cart_multidim_lookup[nuc_idx];
    
                        // Find out which shell derivatives provided by Libint correspond to this nuclear cartesian derivative
                        std::vector<std::vector<int>> indices(deriv_order, std::vector<int> (0,0));
                        for (int j = 0; j < multi_cart_idx.size(); j++){
                            int desired_atom_idx = multi_cart_idx[j] / 3;
                            int desired_coord = multi_cart_idx[j] % 3;
                            for (int i = 0; i < 4; i++){
                                int atom_idx = shell_atom_index_list[i];
                                if (atom_idx == desired_atom_idx) {
                                    int tmp = 3 * i + desired_coord;
                                    indices[j].push_back(tmp);
                                }
                            }
                        }

                        // Now indices is a vector of vectors, where each subvector is your choices for the first derivative operator, second, third, etc
                        // and the total number of subvectors is the order of differentiation
                        // Now we want all combinations where we pick exactly one index from each subvector.
                        // This is achievable through a cartesian product 
                        std::vector<std::vector<int>> index_combos = cartesian_product(indices);
                        std::vector<int> buffer_indices;
                        
                        // Binary search to find 1d buffer index from multidimensional shell derivative index in `index_combos`
                        //for (auto vec : index_combos)  {
                        //    std::sort(vec.begin(), vec.end());
                        //    int buf_idx = 0;
                        //    // buffer_multidim_lookup
                        //    auto it = lower_bound(buffer_multidim_lookup.begin(), buffer_multidim_lookup.end(), vec);
                        //    if (it != buffer_multidim_lookup.end()) buf_idx = it - buffer_multidim_lookup.begin();
                        //    buffer_indices.push_back(buf_idx);
                        //}
                        // Eventually, if you stop using lookup arrays, use above implementation, but these are sitting around so might as well use them 
                        for (auto vec : index_combos)  {
                            if (deriv_order == 1) buffer_indices.push_back(buffer_index_eri1d[vec[0]]);
                            else if (deriv_order == 2) buffer_indices.push_back(buffer_index_eri2d[vec[0]][vec[1]]);
                            else if (deriv_order == 3) buffer_indices.push_back(buffer_index_eri3d[vec[0]][vec[1]][vec[2]]);
                            else if (deriv_order == 4) buffer_indices.push_back(buffer_index_eri4d[vec[0]][vec[1]][vec[2]][vec[3]]);
                        }

                        // Loop over shell block, keeping a total count idx for the size of shell set
                        for(auto i = 0; i < buffer_indices.size(); ++i) {
                            auto eri_shellset = eri_buffer[buffer_indices[i]];
                            if (eri_shellset == nullptr) continue;
                            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                                size_t offset_1 = (bf1 + f1) * nbf * nbf * nbf;
                                for(auto f2 = 0; f2 != n2; ++f2) {
                                    size_t offset_2 = (bf2 + f2) * nbf * nbf;
                                    for(auto f3 = 0; f3 != n3; ++f3) {
                                        size_t offset_3 = (bf3 + f3) * nbf;
                                        for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                            size_t offset_4 = bf4 + f4;
                                            result[offset_1 + offset_2 + offset_3 + offset_4 + offset_nuc_idx] += eri_shellset[idx];
                                        }
                                    }
                                }
                            }
                        }
                    } // For every nuc_idx 0, nderivs_triu
                }
            }
        }
    } // shell quartet loops
    return py::array(result.size(), result.data()); // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042 there's a workaround
} // eri_deriv_core function

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
    m.def("oei_deriv_disk", &oei_deriv_disk, "Computes overlap, kinetic, and potential integral derivative tensors from 1st order up to nth order and writes them to disk with HDF5");
    m.def("eri_deriv_disk", &eri_deriv_disk, "Computes coulomb integral nuclear derivative tensors from 1st order up to nth order and writes them to disk with HDF5");
    m.def("overlap_deriv_core", &overlap_deriv_core, "Computes a single overlap integral derivative tensor, in memory.");
    m.def("kinetic_deriv_core", &kinetic_deriv_core, "Computes a single kinetic integral derivative tensor, in memory.");
    m.def("potential_deriv_core", &potential_deriv_core, "Computes a single potential integral nuclear derivative tensor, in memory.");
    m.def("eri_deriv_core", &eri_deriv_core, "Computes a single coulomb integral nuclear derivative tensor, in memory.");
    //TODO partial derivative impl's
    //m.def("eri_partial_deriv_disk", &eri_partial_deriv_disk, "Computes a subset of the full coulomb integral nuclear derivative tensor and writes them to disk with HDF5");
     m.attr("LIBINT2_MAX_DERIV_ORDER") = LIBINT2_MAX_DERIV_ORDER;
}

