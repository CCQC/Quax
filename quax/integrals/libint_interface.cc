#include <stdlib.h>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <H5Cpp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <libint2.hpp>

#include "buffer_lookups.h"

// TODO support spherical harmonic gaussians, implement symmetry considerations, support 5th, 6th derivs

namespace py = pybind11;
using namespace H5;

std::vector<libint2::Atom> atoms;
unsigned int natom;
unsigned int ncart;
libint2::BasisSet bs1, bs2, bs3, bs4;
unsigned int nbf1, nbf2, nbf3, nbf4;
std::vector<size_t> shell2bf_1, shell2bf_2, shell2bf_3, shell2bf_4;
std::vector<long> shell2atom_1, shell2atom_2, shell2atom_3, shell2atom_4;
int nthreads = 1;

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
void initialize(std::string xyzfilename, std::string basis1, std::string basis2,
                std::string basis3, std::string basis4) {
    libint2::initialize();
    atoms = get_atoms(xyzfilename);
    natom = atoms.size();
    ncart = natom * 3;

    // Move harddrive load of basis and xyz to happen only once
    bs1 = libint2::BasisSet(basis1, atoms);
    bs1.set_pure(false); // use cartesian gaussians
    bs2 = libint2::BasisSet(basis2, atoms);
    bs2.set_pure(false); // use cartesian gaussians
    bs3 = libint2::BasisSet(basis3, atoms);
    bs3.set_pure(false); // use cartesian gaussians
    bs4 = libint2::BasisSet(basis4, atoms);
    bs4.set_pure(false); // use cartesian gaussians

    nbf1 = bs1.nbf();
    nbf2 = bs2.nbf();
    nbf3 = bs3.nbf();
    nbf4 = bs4.nbf();
    shell2bf_1 = bs1.shell2bf();
    shell2bf_2 = bs2.shell2bf();
    shell2bf_3 = bs3.shell2bf();
    shell2bf_4 = bs4.shell2bf();
    shell2atom_1 = bs1.shell2atom(atoms);
    shell2atom_2 = bs2.shell2atom(atoms);
    shell2atom_3 = bs3.shell2atom(atoms);
    shell2atom_4 = bs4.shell2atom(atoms);

    // Get number of OMP threads
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    if (basis1 == basis2 && basis3 == basis4 && basis2 == basis4) {
        py::print("Number of OMP Threads:", nthreads);
    }
}

void finalize() {
    libint2::finalize();
}

// Used to make contracted Gaussian-type geminal for F12 methods
std::vector<std::pair<double, double>> make_cgtg(double exponent) {
    // The fitting coefficients and the exponents from MPQC
    std::vector<std::pair<double, double>> exp_coeff = {};
    std::vector<double> coeffs = {-0.31442480597241274, -0.30369575353387201, -0.16806968430232927,
                                  -0.098115812152857612, -0.060246640234342785, -0.037263541968504843};
    std::vector<double> exps = {0.22085085450735284, 1.0040191632019282, 3.6212173098378728,
                                12.162483236221904, 45.855332448029337, 254.23460688554644};

    for (int i = 0; i < exps.size(); i++){
        auto exp_scaled = (exponent * exponent) * exps[i];
        exp_coeff.push_back(std::make_pair(exp_scaled, coeffs[i]));
    }
    
    return exp_coeff;
}

// Returns square of cgtg
std::vector<std::pair<double, double>> take_square(std::vector<std::pair<double, double>> input) {
    auto n = input.size();
    std::vector<std::pair<double, double>> output;
    for (int i = 0; i < n; ++i) {
        auto e_i = input[i].first;
        auto c_i = input[i].second;
        for (int j = i; j < n; ++j) {
            auto e_j = input[j].first;
            auto c_j = input[j].second;
            double scale = i == j ? 1.0 : 2.0;
            output.emplace_back(std::make_pair(e_i + e_j, scale * c_i * c_j));
        }
    }
    return output;
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
    std::vector<libint2::Engine> s_engines(nthreads);
    size_t max_nprim = std::max(bs1.max_nprim(), bs2.max_nprim());
    int max_l = std::max(bs1.max_l(), bs2.max_l());
    s_engines[0] = libint2::Engine(libint2::Operator::overlap, max_nprim, max_l);
    for (size_t i = 1; i != nthreads; ++i) {
        s_engines[i] = s_engines[0];
    }

    size_t length = nbf1 * nbf2;
    std::vector<double> result(length); // vector to store integral array

#pragma omp parallel for collapse(2) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            auto bf1 = shell2bf_1[s1];  // first basis function in first shell
            auto n1 = bs1[s1].size(); // number of basis functions in first shell
            auto bf2 = shell2bf_2[s2];  // first basis function in second shell
            auto n2 = bs2[s2].size(); // number of basis functions in second shell

            size_t thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            s_engines[thread_id].compute(bs1[s1], bs2[s2]); // Compute shell set
            const auto& buf_vec = s_engines[thread_id].results(); // will point to computed shell sets

            auto ints_shellset = buf_vec[0];    // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out

            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                    result[(bf1 + f1) * nbf2 + bf2 + f2] = ints_shellset[idx];
                }
            }
        }
    }
    return py::array(result.size(), result.data()); 
}

// Compute kinetic energy integrals
py::array kinetic() {
    // Kinetic energy integral engine
    std::vector<libint2::Engine> t_engines(nthreads);
    size_t max_nprim = std::max(bs1.max_nprim(), bs2.max_nprim());
    int max_l = std::max(bs1.max_l(), bs2.max_l());
    t_engines[0] = libint2::Engine(libint2::Operator::kinetic, max_nprim, max_l);
    for (size_t i = 1; i != nthreads; ++i) {
        t_engines[i] = t_engines[0];
    }

    size_t length = nbf1 * nbf2;
    std::vector<double> result(length);

#pragma omp parallel for collapse(2) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            auto bf1 = shell2bf_1[s1];  // first basis function in first shell
            auto n1 = bs1[s1].size(); // number of basis functions in first shell
            auto bf2 = shell2bf_2[s2];  // first basis function in second shell
            auto n2 = bs2[s2].size(); // number of basis functions in second shell

            size_t thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            t_engines[thread_id].compute(bs1[s1], bs2[s2]); // Compute shell set
            const auto& buf_vec = t_engines[thread_id].results(); // will point to computed shell sets

            auto ints_shellset = buf_vec[0];    // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out

            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                    result[(bf1 + f1) * nbf2 + bf2 + f2] = ints_shellset[idx];
                }
            }
        }
    }
    return py::array(result.size(), result.data());
}

// Compute nuclear-electron potential energy integrals
py::array potential() {
    // Potential integral engine
    std::vector<libint2::Engine> v_engines(nthreads);
    size_t max_nprim = std::max(bs1.max_nprim(), bs2.max_nprim());
    int max_l = std::max(bs1.max_l(), bs2.max_l());
    v_engines[0] = libint2::Engine(libint2::Operator::nuclear, max_nprim, max_l);
    v_engines[0].set_params(make_point_charges(atoms));
    for (size_t i = 1; i != nthreads; ++i) {
        v_engines[i] = v_engines[0];
    }

    size_t length = nbf1 * nbf2;
    std::vector<double> result(length);

#pragma omp parallel for collapse(2) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            auto bf1 = shell2bf_1[s1];  // first basis function in first shell
            auto n1 = bs1[s1].size(); // number of basis functions in first shell
            auto bf2 = shell2bf_2[s2];  // first basis function in second shell
            auto n2 = bs2[s2].size(); // number of basis functions in second shell

            size_t thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            v_engines[thread_id].compute(bs1[s1], bs2[s2]); // Compute shell set
            const auto& buf_vec = v_engines[thread_id].results(); // will point to computed shell sets

            auto ints_shellset = buf_vec[0];    // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out

            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                    // idx = x + (y * width) where x = bf2 + f2 and y = bf1 + f1 
                    result[(bf1 + f1) * nbf2 + bf2 + f2] = ints_shellset[idx];
                }
            }
        }
    }
    return py::array(result.size(), result.data());
}

// Computes electron repulsion integrals
py::array eri() {
    // workaround for data copying: perhaps pass an empty numpy array, then populate it in C++? avoids last line, which copies
    std::vector<libint2::Engine> eri_engines(nthreads);
    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
    eri_engines[0] = libint2::Engine(libint2::Operator::coulomb, max_nprim, max_l);
    for (size_t i = 1; i != nthreads; ++i) {
        eri_engines[i] = eri_engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4;
    std::vector<double> result(length);
    
#pragma omp parallel for collapse(4) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            for(auto s3=0; s3 != bs3.size(); ++s3) {
                for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                    auto bf1 = shell2bf_1[s1];  // first basis function in first shell
                    auto n1 = bs1[s1].size(); // number of basis functions in first shell
                    auto bf2 = shell2bf_2[s2];  // first basis function in second shell
                    auto n2 = bs2[s2].size(); // number of basis functions in second shell
                    auto bf3 = shell2bf_3[s3];  // first basis function in third shell
                    auto n3 = bs3[s3].size(); // number of basis functions in third shell
                    auto bf4 = shell2bf_4[s4];  // first basis function in fourth shell
                    auto n4 = bs4[s4].size(); // number of basis functions in fourth shell

                    size_t thread_id = 0;
#ifdef _OPENMP
                    thread_id = omp_get_thread_num();
#endif
                    eri_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                    const auto& buf_vec = eri_engines[thread_id].results(); // will point to computed shell sets

                    auto ints_shellset = buf_vec[0];    // Location of the computed integrals
                    if (ints_shellset == nullptr)
                        continue;  // nullptr returned if the entire shell-set was screened out

                    // Loop over shell block, keeping a total count idx for the size of shell set
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                        for(auto f2 = 0; f2 != n2; ++f2) {
                            size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                            for(auto f3 = 0; f3 != n3; ++f3) {
                                size_t offset_3 = (bf3 + f3) * nbf4;
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

// Computes integrals of contracted Gaussian-type geminal
py::array f12(double beta) {
    // workaround for data copying: perhaps pass an empty numpy array, then populate it in C++? avoids last line, which copies
    auto cgtg_params = make_cgtg(beta);
    std::vector<libint2::Engine> cgtg_engines(nthreads);
    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
    cgtg_engines[0] = libint2::Engine(libint2::Operator::cgtg, max_nprim, max_l);
    cgtg_engines[0].set_params(cgtg_params);
    for (size_t i = 1; i != nthreads; ++i) {
        cgtg_engines[i] = cgtg_engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4;
    std::vector<double> result(length);
    
#pragma omp parallel for collapse(4) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            for(auto s3=0; s3 != bs3.size(); ++s3) {
                for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                    auto bf1 = shell2bf_1[s1];  // first basis function in first shell
                    auto n1 = bs1[s1].size(); // number of basis functions in first shell
                    auto bf2 = shell2bf_2[s2];  // first basis function in second shell
                    auto n2 = bs2[s2].size(); // number of basis functions in second shell
                    auto bf3 = shell2bf_3[s3];  // first basis function in third shell
                    auto n3 = bs3[s3].size(); // number of basis functions in third shell
                    auto bf4 = shell2bf_4[s4];  // first basis function in fourth shell
                    auto n4 = bs4[s4].size(); // number of basis functions in fourth shell

                    size_t thread_id = 0;
#ifdef _OPENMP
                    thread_id = omp_get_thread_num();
#endif
                    cgtg_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                    const auto& buf_vec = cgtg_engines[thread_id].results(); // will point to computed shell sets

                    auto ints_shellset = buf_vec[0];    // Location of the computed integrals
                    if (ints_shellset == nullptr)
                        continue;  // nullptr returned if the entire shell-set was screened out

                    // Loop over shell block, keeping a total count idx for the size of shell set
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                        for(auto f2 = 0; f2 != n2; ++f2) {
                            size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                            for(auto f3 = 0; f3 != n3; ++f3) {
                                size_t offset_3 = (bf3 + f3) * nbf4;
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

// Computes integrals of squared contracted Gaussian-type geminal
py::array f12_squared(double beta) {
    // workaround for data copying: perhaps pass an empty numpy array, then populate it in C++? avoids last line, which copies
    auto cgtg_params = take_square(make_cgtg(beta));
    std::vector<libint2::Engine> cgtg_squared_engines(nthreads);
    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
    cgtg_squared_engines[0] = libint2::Engine(libint2::Operator::cgtg, max_nprim, max_l);
    cgtg_squared_engines[0].set_params(cgtg_params);
    for (size_t i = 1; i != nthreads; ++i) {
        cgtg_squared_engines[i] = cgtg_squared_engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4;
    std::vector<double> result(length);
    
#pragma omp parallel for collapse(4) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            for(auto s3=0; s3 != bs3.size(); ++s3) {
                for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                    auto bf1 = shell2bf_1[s1];  // first basis function in first shell
                    auto n1 = bs1[s1].size(); // number of basis functions in first shell
                    auto bf2 = shell2bf_2[s2];  // first basis function in second shell
                    auto n2 = bs2[s2].size(); // number of basis functions in second shell
                    auto bf3 = shell2bf_3[s3];  // first basis function in third shell
                    auto n3 = bs3[s3].size(); // number of basis functions in third shell
                    auto bf4 = shell2bf_4[s4];  // first basis function in fourth shell
                    auto n4 = bs4[s4].size(); // number of basis functions in fourth shell

                    size_t thread_id = 0;
#ifdef _OPENMP
                    thread_id = omp_get_thread_num();
#endif
                    cgtg_squared_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                    const auto& buf_vec = cgtg_squared_engines[thread_id].results(); // will point to computed shell sets

                    auto ints_shellset = buf_vec[0];    // Location of the computed integrals
                    if (ints_shellset == nullptr)
                        continue;  // nullptr returned if the entire shell-set was screened out

                    // Loop over shell block, keeping a total count idx for the size of shell set
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                        for(auto f2 = 0; f2 != n2; ++f2) {
                            size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                            for(auto f3 = 0; f3 != n3; ++f3) {
                                size_t offset_3 = (bf3 + f3) * nbf4;
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

// Computes electron repulsion integrals of contracted Gaussian-type geminal
py::array f12g12(double beta) {
    // workaround for data copying: perhaps pass an empty numpy array, then populate it in C++? avoids last line, which copies
    auto cgtg_params = make_cgtg(beta);
    std::vector<libint2::Engine> cgtg_coulomb_engines(nthreads);
    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
    cgtg_coulomb_engines[0] = libint2::Engine(libint2::Operator::cgtg_x_coulomb, max_nprim, max_l);
    cgtg_coulomb_engines[0].set_params(cgtg_params);
    for (size_t i = 1; i != nthreads; ++i) {
        cgtg_coulomb_engines[i] = cgtg_coulomb_engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4;
    std::vector<double> result(length);
    
#pragma omp parallel for collapse(4) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            for(auto s3=0; s3 != bs3.size(); ++s3) {
                for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                    auto bf1 = shell2bf_1[s1];  // first basis function in first shell
                    auto n1 = bs1[s1].size(); // number of basis functions in first shell
                    auto bf2 = shell2bf_2[s2];  // first basis function in second shell
                    auto n2 = bs2[s2].size(); // number of basis functions in second shell
                    auto bf3 = shell2bf_3[s3];  // first basis function in third shell
                    auto n3 = bs3[s3].size(); // number of basis functions in third shell
                    auto bf4 = shell2bf_4[s4];  // first basis function in fourth shell
                    auto n4 = bs4[s4].size(); // number of basis functions in fourth shell

                    size_t thread_id = 0;
#ifdef _OPENMP
                    thread_id = omp_get_thread_num();
#endif
                    cgtg_coulomb_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                    const auto& buf_vec = cgtg_coulomb_engines[thread_id].results(); // will point to computed shell sets

                    auto ints_shellset = buf_vec[0];    // Location of the computed integrals
                    if (ints_shellset == nullptr)
                        continue;  // nullptr returned if the entire shell-set was screened out

                    // Loop over shell block, keeping a total count idx for the size of shell set
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                        for(auto f2 = 0; f2 != n2; ++f2) {
                            size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                            for(auto f3 = 0; f3 != n3; ++f3) {
                                size_t offset_3 = (bf3 + f3) * nbf4;
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

// Computes gradient norm of contracted Gaussian-type geminal
py::array f12_double_commutator(double beta) {
    // workaround for data copying: perhaps pass an empty numpy array, then populate it in C++? avoids last line, which copies
    auto cgtg_params = make_cgtg(beta);
    std::vector<libint2::Engine> cgtg_del_engines(nthreads);
    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
    // Returns Runtime Error: bad any_cast if shorthand version is used, may be an error on the Libint side since Psi4 works with this as well
    cgtg_del_engines[0] = libint2::Engine(libint2::Operator::delcgtg2, max_nprim, max_l, 0, 0., cgtg_params, libint2::BraKet::xx_xx);
    for (size_t i = 1; i != nthreads; ++i) {
        cgtg_del_engines[i] = cgtg_del_engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4;
    std::vector<double> result(length);
    
#pragma omp parallel for collapse(4) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            for(auto s3=0; s3 != bs3.size(); ++s3) {
                for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                    auto bf1 = shell2bf_1[s1];  // first basis function in first shell
                    auto n1 = bs1[s1].size(); // number of basis functions in first shell
                    auto bf2 = shell2bf_2[s2];  // first basis function in second shell
                    auto n2 = bs2[s2].size(); // number of basis functions in second shell
                    auto bf3 = shell2bf_3[s3];  // first basis function in third shell
                    auto n3 = bs3[s3].size(); // number of basis functions in third shell
                    auto bf4 = shell2bf_4[s4];  // first basis function in fourth shell
                    auto n4 = bs4[s4].size(); // number of basis functions in fourth shell

                    size_t thread_id = 0;
#ifdef _OPENMP
                    thread_id = omp_get_thread_num();
#endif
                    cgtg_del_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                    const auto& buf_vec = cgtg_del_engines[thread_id].results(); // will point to computed shell sets

                    auto ints_shellset = buf_vec[0];    // Location of the computed integrals
                    if (ints_shellset == nullptr)
                        continue;  // nullptr returned if the entire shell-set was screened out

                    // Loop over shell block, keeping a total count idx for the size of shell set
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                        for(auto f2 = 0; f2 != n2; ++f2) {
                            size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                            for(auto f3 = 0; f3 != n3; ++f3) {
                                size_t offset_3 = (bf3 + f3) * nbf4;
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
    std::vector<libint2::Engine> s_engines(nthreads);
    size_t max_nprim = std::max(bs1.max_nprim(), bs2.max_nprim());
    int max_l = std::max(bs1.max_l(), bs2.max_l());
    s_engines[0] = libint2::Engine(libint2::Operator::overlap, max_nprim, max_l, deriv_order);
    for (size_t i = 1; i != nthreads; ++i) {
        s_engines[i] = s_engines[0];
    }

    // Get size of overlap derivative array and allocate 
    size_t length = nbf1 * nbf2;
    std::vector<double> result(length);

#pragma omp parallel for collapse(2) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
            auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
            auto n1 = bs1[s1].size();    // number of basis functions in shell 1
            auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
            auto n2 = bs2[s2].size();    // number of basis functions in shell 2
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
            size_t thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            s_engines[thread_id].compute(bs1[s1], bs2[s2]); // Compute shell set
            const auto& buf_vec = s_engines[thread_id].results(); // will point to computed shell sets

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
                    result[(bf1 + f1) * nbf2 + bf2 + f2 ] = ints_shellset[idx];
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
    std::vector<libint2::Engine> t_engines(nthreads);
    size_t max_nprim = std::max(bs1.max_nprim(), bs2.max_nprim());
    int max_l = std::max(bs1.max_l(), bs2.max_l());
    t_engines[0] = libint2::Engine(libint2::Operator::kinetic, max_nprim, max_l, deriv_order);
    for (size_t i = 1; i != nthreads; ++i) {
        t_engines[i] = t_engines[0];
    }

    size_t length = nbf1 * nbf2;
    std::vector<double> result(length);

#pragma omp parallel for collapse(2) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
            auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
            auto n1 = bs1[s1].size();    // number of basis functions in shell 1
            auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
            auto n2 = bs2[s2].size();    // number of basis functions in shell 2
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
            size_t thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            t_engines[thread_id].compute(bs1[s1], bs2[s2]); // Compute shell set
            const auto& buf_vec = t_engines[thread_id].results(); // will point to computed shell sets

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
                    result[(bf1 + f1) * nbf2 + bf2 + f2] = ints_shellset[idx];
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
    std::vector<libint2::Engine> v_engines(nthreads);
    size_t max_nprim = std::max(bs1.max_nprim(), bs2.max_nprim());
    int max_l = std::max(bs1.max_l(), bs2.max_l());
    v_engines[0] = libint2::Engine(libint2::Operator::nuclear, max_nprim, max_l, deriv_order);
    v_engines[0].set_params(make_point_charges(atoms));
    for (size_t i = 1; i != nthreads; ++i) {
        v_engines[i] = v_engines[0];
    }

    // Get size of potential derivative array and allocate
    size_t length = nbf1 * nbf2;
    std::vector<double> result(length);

#pragma omp parallel for collapse(2) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
            auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
            auto n1 = bs1[s1].size();    // number of basis functions in shell 1
            auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
            auto n2 = bs2[s2].size();    // number of basis functions in shell 2

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
                        int tmp = 3 * (i +2) + desired_coordinates[j];
                        indices[j].push_back(tmp);
                    }
                }
            }

            // Create index combos representing every mixed partial derivative operator which contributes to nuclear derivative
            std::vector<std::vector<int>> index_combos = cartesian_product(indices);

            // Compute the integrals
            size_t thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            v_engines[thread_id].compute(bs1[s1], bs2[s2]); // Compute shell set
            const auto& buf_vec = v_engines[thread_id].results(); // will point to computed shell sets
            
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
                        result[(bf1 + f1) * nbf2 + bf2 + f2] += ints_shellset[idx];
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
    std::vector<libint2::Engine> eri_engines(nthreads);
    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
    eri_engines[0] = libint2::Engine(libint2::Operator::coulomb, max_nprim, max_l, deriv_order);
    for (size_t i = 1; i != nthreads; ++i) {
        eri_engines[i] = eri_engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4;
    std::vector<double> result(length);

#pragma omp parallel for collapse(4) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            for(auto s3 = 0; s3 != bs3.size(); ++s3) {
                for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                    auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                    auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                    auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                    auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                    auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                    auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                    auto bf3 = shell2bf_3[s3];     // Index of first basis function in shell 3
                    auto atom3 = shell2atom_3[s3]; // Atom index of shell 3
                    auto n3 = bs3[s3].size();    // number of basis functions in shell 3
                    auto bf4 = shell2bf_4[s4];     // Index of first basis function in shell 4
                    auto atom4 = shell2atom_4[s4]; // Atom index of shell 4
                    auto n4 = bs4[s4].size();    // number of basis functions in shell 4

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
                    size_t thread_id = 0;
#ifdef _OPENMP
                    thread_id = omp_get_thread_num();
#endif
                    eri_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                    const auto& buf_vec = eri_engines[thread_id].results(); // will point to computed shell sets

                    for(auto i = 0; i<buffer_indices.size(); ++i) {
                        auto ints_shellset = buf_vec[buffer_indices[i]];
                        if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                            for(auto f2 = 0; f2 != n2; ++f2) {
                                size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                                for(auto f3 = 0; f3 != n3; ++f3) {
                                    size_t offset_3 = (bf3 + f3) * nbf4;
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

// Computes nuclear derivatives of contracted Gaussian-type geminal integrals
py::array f12_deriv(double beta, std::vector<int> deriv_vec) {
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    assert(ncart == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");

    // F12 derivative integral engine
    auto cgtg_params = make_cgtg(beta);
    std::vector<libint2::Engine> cgtg_engines(nthreads);
    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
    cgtg_engines[0] = libint2::Engine(libint2::Operator::cgtg, max_nprim, max_l, deriv_order);
    cgtg_engines[0].set_params(cgtg_params);
    for (size_t i = 1; i != nthreads; ++i) {
        cgtg_engines[i] = cgtg_engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4;
    std::vector<double> result(length);

#pragma omp parallel for collapse(4) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            for(auto s3 = 0; s3 != bs3.size(); ++s3) {
                for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                    auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                    auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                    auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                    auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                    auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                    auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                    auto bf3 = shell2bf_3[s3];     // Index of first basis function in shell 3
                    auto atom3 = shell2atom_3[s3]; // Atom index of shell 3
                    auto n3 = bs3[s3].size();    // number of basis functions in shell 3
                    auto bf4 = shell2bf_4[s4];     // Index of first basis function in shell 4
                    auto atom4 = shell2atom_4[s4]; // Atom index of shell 4
                    auto n4 = bs4[s4].size();    // number of basis functions in shell 4

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
                    size_t thread_id = 0;
#ifdef _OPENMP
                    thread_id = omp_get_thread_num();
#endif
                    cgtg_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                    const auto& buf_vec = cgtg_engines[thread_id].results(); // will point to computed shell sets

                    for(auto i = 0; i<buffer_indices.size(); ++i) {
                        auto ints_shellset = buf_vec[buffer_indices[i]];
                        if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                            for(auto f2 = 0; f2 != n2; ++f2) {
                                size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                                for(auto f3 = 0; f3 != n3; ++f3) {
                                    size_t offset_3 = (bf3 + f3) * nbf4;
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

// Computes nuclear derivatives of squared contracted Gaussian-type geminal integrals
py::array f12_squared_deriv(double beta, std::vector<int> deriv_vec) {
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    assert(ncart == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");

    // F12 Squared derivative integral engine
    auto cgtg_params = take_square(make_cgtg(beta));
    std::vector<libint2::Engine> cgtg_squared_engines(nthreads);
    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
    cgtg_squared_engines[0] = libint2::Engine(libint2::Operator::cgtg, max_nprim, max_l, deriv_order);
    cgtg_squared_engines[0].set_params(cgtg_params);
    for (size_t i = 1; i != nthreads; ++i) {
        cgtg_squared_engines[i] = cgtg_squared_engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4;
    std::vector<double> result(length);

#pragma omp parallel for collapse(4) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            for(auto s3 = 0; s3 != bs3.size(); ++s3) {
                for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                    auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                    auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                    auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                    auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                    auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                    auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                    auto bf3 = shell2bf_3[s3];     // Index of first basis function in shell 3
                    auto atom3 = shell2atom_3[s3]; // Atom index of shell 3
                    auto n3 = bs3[s3].size();    // number of basis functions in shell 3
                    auto bf4 = shell2bf_4[s4];     // Index of first basis function in shell 4
                    auto atom4 = shell2atom_4[s4]; // Atom index of shell 4
                    auto n4 = bs4[s4].size();    // number of basis functions in shell 4

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
                    size_t thread_id = 0;
#ifdef _OPENMP
                    thread_id = omp_get_thread_num();
#endif
                    cgtg_squared_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                    const auto& buf_vec = cgtg_squared_engines[thread_id].results(); // will point to computed shell sets

                    for(auto i = 0; i<buffer_indices.size(); ++i) {
                        auto ints_shellset = buf_vec[buffer_indices[i]];
                        if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                            for(auto f2 = 0; f2 != n2; ++f2) {
                                size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                                for(auto f3 = 0; f3 != n3; ++f3) {
                                    size_t offset_3 = (bf3 + f3) * nbf4;
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

// Computes nuclear derivatives of contracted Gaussian-type geminal times Coulomb replusion integrals
py::array f12g12_deriv(double beta, std::vector<int> deriv_vec) {
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    assert(ncart == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");

    // F12 derivative integral engine
    auto cgtg_params = make_cgtg(beta);
    std::vector<libint2::Engine> cgtg_coulomb_engines(nthreads);
    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
    cgtg_coulomb_engines[0] = libint2::Engine(libint2::Operator::cgtg_x_coulomb, max_nprim, max_l, deriv_order);
    cgtg_coulomb_engines[0].set_params(cgtg_params);
    for (size_t i = 1; i != nthreads; ++i) {
        cgtg_coulomb_engines[i] = cgtg_coulomb_engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4;
    std::vector<double> result(length);

#pragma omp parallel for collapse(4) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            for(auto s3 = 0; s3 != bs3.size(); ++s3) {
                for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                    auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                    auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                    auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                    auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                    auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                    auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                    auto bf3 = shell2bf_3[s3];     // Index of first basis function in shell 3
                    auto atom3 = shell2atom_3[s3]; // Atom index of shell 3
                    auto n3 = bs3[s3].size();    // number of basis functions in shell 3
                    auto bf4 = shell2bf_4[s4];     // Index of first basis function in shell 4
                    auto atom4 = shell2atom_4[s4]; // Atom index of shell 4
                    auto n4 = bs4[s4].size();    // number of basis functions in shell 4

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
                    size_t thread_id = 0;
#ifdef _OPENMP
                    thread_id = omp_get_thread_num();
#endif
                    cgtg_coulomb_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                    const auto& buf_vec = cgtg_coulomb_engines[thread_id].results(); // will point to computed shell sets

                    for(auto i = 0; i<buffer_indices.size(); ++i) {
                        auto ints_shellset = buf_vec[buffer_indices[i]];
                        if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                            for(auto f2 = 0; f2 != n2; ++f2) {
                                size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                                for(auto f3 = 0; f3 != n3; ++f3) {
                                    size_t offset_3 = (bf3 + f3) * nbf4;
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

// Computes nuclear derivatives of gradient norm of contracted Gaussian-type geminal integrals
py::array f12_double_commutator_deriv(double beta, std::vector<int> deriv_vec) {
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    assert(ncart == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");

    // F12 derivative integral engine
    auto cgtg_params = make_cgtg(beta);
    std::vector<libint2::Engine> cgtg_del_engines(nthreads);
    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
    // Returns Runtime Error: bad any_cast if shorthand version is used, may be an error on the Libint side since Psi4 works with this as well
    cgtg_del_engines[0] = libint2::Engine(libint2::Operator::delcgtg2, max_nprim, max_l, deriv_order, 0., cgtg_params, libint2::BraKet::xx_xx);
    for (size_t i = 1; i != nthreads; ++i) {
        cgtg_del_engines[i] = cgtg_del_engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4;
    std::vector<double> result(length);

#pragma omp parallel for collapse(4) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            for(auto s3 = 0; s3 != bs3.size(); ++s3) {
                for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                    auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                    auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                    auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                    auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                    auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                    auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                    auto bf3 = shell2bf_3[s3];     // Index of first basis function in shell 3
                    auto atom3 = shell2atom_3[s3]; // Atom index of shell 3
                    auto n3 = bs3[s3].size();    // number of basis functions in shell 3
                    auto bf4 = shell2bf_4[s4];     // Index of first basis function in shell 4
                    auto atom4 = shell2atom_4[s4]; // Atom index of shell 4
                    auto n4 = bs4[s4].size();    // number of basis functions in shell 4

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
                    size_t thread_id = 0;
#ifdef _OPENMP
                    thread_id = omp_get_thread_num();
#endif
                    cgtg_del_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                    const auto& buf_vec = cgtg_del_engines[thread_id].results(); // will point to computed shell sets

                    for(auto i = 0; i<buffer_indices.size(); ++i) {
                        auto ints_shellset = buf_vec[buffer_indices[i]];
                        if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                            for(auto f2 = 0; f2 != n2; ++f2) {
                                size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                                for(auto f3 = 0; f3 != n3; ++f3) {
                                    size_t offset_3 = (bf3 + f3) * nbf4;
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

    size_t max_nprim = std::max(bs1.max_nprim(), bs2.max_nprim());
    int max_l = std::max(bs1.max_l(), bs2.max_l());

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
        const std::vector<std::vector<int>> potential_buffer_multidim_lookup = generate_multi_index_lookup(6 + ncart, deriv_order);

        // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
        const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

        // Define engines and buffers
        std::vector<libint2::Engine> s_engines(nthreads), t_engines(nthreads), v_engines(nthreads);
        s_engines[0] = libint2::Engine(libint2::Operator::overlap, max_nprim, max_l, deriv_order);
        t_engines[0] = libint2::Engine(libint2::Operator::kinetic, max_nprim, max_l, deriv_order);
        v_engines[0] = libint2::Engine(libint2::Operator::nuclear, max_nprim, max_l, deriv_order);
        v_engines[0].set_params(make_point_charges(atoms));
        for (size_t i = 1; i != nthreads; ++i) {
            s_engines[i] = s_engines[0];
            t_engines[i] = t_engines[0];
            v_engines[i] = v_engines[0];
        }

        // Define HDF5 dataset names
        const H5std_string overlap_dset_name("overlap_deriv" + std::to_string(deriv_order));
        const H5std_string kinetic_dset_name("kinetic_deriv" + std::to_string(deriv_order));
        const H5std_string potential_dset_name("potential_deriv" + std::to_string(deriv_order));

        // Define rank and dimensions of data that will be written to the file
        hsize_t file_dims[] = {nbf1, nbf2, nderivs_triu};
        DataSpace fspace(3, file_dims);
        // Create dataset for each integral type and write 0.0's into the file 
        DataSet* overlap_dataset = new DataSet(file->createDataSet(overlap_dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        DataSet* kinetic_dataset = new DataSet(file->createDataSet(kinetic_dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        DataSet* potential_dataset = new DataSet(file->createDataSet(potential_dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        hsize_t stride[3] = {1, 1, 1}; // stride and block can be used to 
        hsize_t block[3] = {1, 1, 1};  // add values to multiple places, useful if symmetry ever used.
        hsize_t zerostart[3] = {0, 0, 0};

#pragma omp parallel for collapse(2) num_threads(nthreads)
        for(auto s1 = 0; s1 != bs1.size(); ++s1) {
            for(auto s2 = 0; s2 != bs2.size(); ++s2) {
                auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                std::vector<long> shell_atom_index_list{atom1, atom2};

                size_t thread_id = 0;
#ifdef _OPENMP
                thread_id = omp_get_thread_num();
#endif
                s_engines[thread_id].compute(bs1[s1], bs2[s2]); // Compute shell set
                t_engines[thread_id].compute(bs1[s1], bs2[s2]); // Compute shell set
                v_engines[thread_id].compute(bs1[s1], bs2[s2]); // Compute shell set
                const auto& overlap_buffer = s_engines[thread_id].results(); // will point to computed shell sets
                const auto& kinetic_buffer = t_engines[thread_id].results(); // will point to computed shell sets
                const auto& potential_buffer = v_engines[thread_id].results(); // will point to computed shell sets;

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
                                int tmp = 3 * (i + 2) + desired_coord;
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

    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));

    // Check to make sure you are not flooding the disk.
    long total_deriv_slices = 0;
    for (int i = 1; i <= max_deriv_order; i++){
        total_deriv_slices += how_many_derivs(natom, i);
    }
    double check = (nbf1 * nbf2 * nbf3 * nbf4 * total_deriv_slices * 8) * (1e-9);
    assert(check < 10 && "Total disk space required for ERI's exceeds 10 GB. Increase threshold and recompile to proceed.");

    for (int deriv_order = 1; deriv_order <= max_deriv_order; deriv_order++){
        // Number of unique shell derivatives output by libint (number of indices in buffer)
        int nshell_derivs = how_many_derivs(4, deriv_order);
        // Number of unique nuclear derivatives of ERI's
        unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

        // Create mapping from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
        // Currently not used due to predefined lookup arrays
        //const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);

        // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
        const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

        // Libint engine for computing shell quartet derivatives
        std::vector<libint2::Engine> eri_engines(nthreads);
        eri_engines[0] = libint2::Engine(libint2::Operator::coulomb, max_nprim, max_l, deriv_order);
        for (size_t i = 1; i != nthreads; ++i) {
            eri_engines[i] = eri_engines[0];
        }

        // Define HDF5 dataset name
        const H5std_string eri_dset_name("eri_deriv" + std::to_string(deriv_order));
        hsize_t file_dims[] = {nbf1, nbf2, nbf3, nbf4, nderivs_triu};
        DataSpace fspace(5, file_dims);
        // Create dataset for each integral type and write 0.0's into the file 
        DataSet* eri_dataset = new DataSet(file->createDataSet(eri_dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        hsize_t stride[5] = {1, 1, 1, 1, 1}; // stride and block can be used to 
        hsize_t block[5] = {1, 1, 1, 1, 1};  // add values to multiple places, useful if symmetry ever used.
        hsize_t zerostart[5] = {0, 0, 0, 0, 0};

#pragma omp parallel for collapse(4) num_threads(nthreads)
        for(auto s1 = 0; s1 != bs1.size(); ++s1) {
            for(auto s2 = 0; s2 != bs2.size(); ++s2) {
                for(auto s3 = 0; s3 != bs3.size(); ++s3) {
                    for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                        auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                        auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                        auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                        auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                        auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                        auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                        auto bf3 = shell2bf_3[s3];     // Index of first basis function in shell 3
                        auto atom3 = shell2atom_3[s3]; // Atom index of shell 3
                        auto n3 = bs3[s3].size();    // number of basis functions in shell 3
                        auto bf4 = shell2bf_4[s4];     // Index of first basis function in shell 4
                        auto atom4 = shell2atom_4[s4]; // Atom index of shell 4
                        auto n4 = bs4[s4].size();    // number of basis functions in shell 4

                        if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
                        std::vector<long> shell_atom_index_list{atom1, atom2, atom3, atom4};

                        size_t thread_id = 0;
#ifdef _OPENMP
                        thread_id = omp_get_thread_num();
#endif
                        eri_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                        const auto& eri_buffer = eri_engines[thread_id].results(); // will point to computed shell sets

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

// Writes all F12 ints up to `max_deriv_order` to disk.
// HDF5 File Name: f12_derivs.h5 
//      HDF5 Dataset names within the file:
//      f12_deriv1 
//          shape (nbf,nbf,nbf,nbf,n_unique_1st_derivs)
//      f12_deriv2
//          shape (nbf,nbf,nbf,nbf,n_unique_2nd_derivs)
//      f12_deriv3
//          shape (nbf,nbf,nbf,nbf,n_unique_3rd_derivs)
//      ...
void f12_deriv_disk(double beta, int max_deriv_order) { 
    std::cout << "Writing two-electron F12 integral derivative tensors up to order " << max_deriv_order << " to disk...";
    const H5std_string file_name("f12_derivs.h5");
    H5File* file = new H5File(file_name,H5F_ACC_TRUNC);
    double fillvalue = 0.0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);

    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));

    // Check to make sure you are not flooding the disk.
    long total_deriv_slices = 0;
    for (int i = 1; i <= max_deriv_order; i++){
        total_deriv_slices += how_many_derivs(natom, i);
    }
    double check = (nbf1 * nbf2 * nbf3 * nbf4 * total_deriv_slices * 8) * (1e-9);
    assert(check < 10 && "Total disk space required for ERI's exceeds 10 GB. Increase threshold and recompile to proceed.");

    auto cgtg_params = make_cgtg(beta);
    
    for (int deriv_order = 1; deriv_order <= max_deriv_order; deriv_order++){
        // Number of unique shell derivatives output by libint (number of indices in buffer)
        int nshell_derivs = how_many_derivs(4, deriv_order);
        // Number of unique nuclear derivatives of ERI's
        unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

        // Create mapping from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
        // Currently not used due to predefined lookup arrays
        //const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);

        // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
        const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

        // Libint engine for computing shell quartet derivatives
        std::vector<libint2::Engine> cgtg_engines(nthreads);
        cgtg_engines[0] = libint2::Engine(libint2::Operator::cgtg, max_nprim, max_l, deriv_order);
        cgtg_engines[0].set_params(cgtg_params);
        for (size_t i = 1; i != nthreads; ++i) {
            cgtg_engines[i] = cgtg_engines[0];
        }

        // Define HDF5 dataset name
        const H5std_string eri_dset_name("f12_deriv" + std::to_string(deriv_order));
        hsize_t file_dims[] = {nbf1, nbf2, nbf3, nbf4, nderivs_triu};
        DataSpace fspace(5, file_dims);
        // Create dataset for each integral type and write 0.0's into the file 
        DataSet* f12_dataset = new DataSet(file->createDataSet(eri_dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        hsize_t stride[5] = {1, 1, 1, 1, 1}; // stride and block can be used to 
        hsize_t block[5] = {1, 1, 1, 1, 1};  // add values to multiple places, useful if symmetry ever used.
        hsize_t zerostart[5] = {0, 0, 0, 0, 0};

#pragma omp parallel for collapse(4) num_threads(nthreads)
        for(auto s1 = 0; s1 != bs1.size(); ++s1) {
            for(auto s2 = 0; s2 != bs2.size(); ++s2) {
                for(auto s3 = 0; s3 != bs3.size(); ++s3) {
                    for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                        auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                        auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                        auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                        auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                        auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                        auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                        auto bf3 = shell2bf_3[s3];     // Index of first basis function in shell 3
                        auto atom3 = shell2atom_3[s3]; // Atom index of shell 3
                        auto n3 = bs3[s3].size();    // number of basis functions in shell 3
                        auto bf4 = shell2bf_4[s4];     // Index of first basis function in shell 4
                        auto atom4 = shell2atom_4[s4]; // Atom index of shell 4
                        auto n4 = bs4[s4].size();    // number of basis functions in shell 4

                        if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
                        std::vector<long> shell_atom_index_list{atom1, atom2, atom3, atom4};

                        size_t thread_id = 0;
#ifdef _OPENMP
                        thread_id = omp_get_thread_num();
#endif
                        cgtg_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                        const auto& f12_buffer = cgtg_engines[thread_id].results(); // will point to computed shell sets

                        // Define shell set slab, with extra dimension for unique derivatives, initialized with 0.0's
                        double f12_shellset_slab [n1][n2][n3][n4][nderivs_triu] = {};
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
                                auto f12_shellset = f12_buffer[buffer_indices[i]];
                                if (f12_shellset == nullptr) continue;
                                for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                                    for(auto f2 = 0; f2 != n2; ++f2) {
                                        for(auto f3 = 0; f3 != n3; ++f3) {
                                            for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                                f12_shellset_slab[f1][f2][f3][f4][nuc_idx] += f12_shellset[idx];
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
                        f12_dataset->write(f12_shellset_slab, PredType::NATIVE_DOUBLE, mspace, fspace);
                    }
                }
            }
        } // shell quartet loops
    // Close the dataset for this derivative order
    delete f12_dataset;
    } // deriv order loop 
// Close the file
delete file;
std::cout << " done" << std::endl;
} // f12_deriv_disk function

// Writes all F12 Squared ints up to `max_deriv_order` to disk.
// HDF5 File Name: f12_squared_derivs.h5 
//      HDF5 Dataset names within the file:
//      f12_squared_deriv1 
//          shape (nbf,nbf,nbf,nbf,n_unique_1st_derivs)
//      f12_squared_deriv2
//          shape (nbf,nbf,nbf,nbf,n_unique_2nd_derivs)
//      f12_squared_deriv3
//          shape (nbf,nbf,nbf,nbf,n_unique_3rd_derivs)
//      ...
void f12_squared_deriv_disk(double beta, int max_deriv_order) { 
    std::cout << "Writing two-electron F12 squared integral derivative tensors up to order " << max_deriv_order << " to disk...";
    const H5std_string file_name("f12_squared_derivs.h5");
    H5File* file = new H5File(file_name,H5F_ACC_TRUNC);
    double fillvalue = 0.0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);

    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));

    // Check to make sure you are not flooding the disk.
    long total_deriv_slices = 0;
    for (int i = 1; i <= max_deriv_order; i++){
        total_deriv_slices += how_many_derivs(natom, i);
    }
    double check = (nbf1 * nbf2 * nbf3 * nbf4 * total_deriv_slices * 8) * (1e-9);
    assert(check < 10 && "Total disk space required for ERI's exceeds 10 GB. Increase threshold and recompile to proceed.");

    auto cgtg_params = take_square(make_cgtg(beta));
    
    for (int deriv_order = 1; deriv_order <= max_deriv_order; deriv_order++){
        // Number of unique shell derivatives output by libint (number of indices in buffer)
        int nshell_derivs = how_many_derivs(4, deriv_order);
        // Number of unique nuclear derivatives of ERI's
        unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

        // Create mapping from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
        // Currently not used due to predefined lookup arrays
        //const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);

        // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
        const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

        // Libint engine for computing shell quartet derivatives
        std::vector<libint2::Engine> cgtg_squared_engines(nthreads);
        size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
        int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
        cgtg_squared_engines[0] = libint2::Engine(libint2::Operator::cgtg, max_nprim, max_l, deriv_order);
        cgtg_squared_engines[0].set_params(cgtg_params);
        for (size_t i = 1; i != nthreads; ++i) {
            cgtg_squared_engines[i] = cgtg_squared_engines[0];
        }

        // Define HDF5 dataset name
        const H5std_string eri_dset_name("f12_squared_deriv" + std::to_string(deriv_order));
        hsize_t file_dims[] = {nbf1, nbf2, nbf3, nbf4, nderivs_triu};
        DataSpace fspace(5, file_dims);
        // Create dataset for each integral type and write 0.0's into the file 
        DataSet* f12_squared_dataset = new DataSet(file->createDataSet(eri_dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        hsize_t stride[5] = {1, 1, 1, 1, 1}; // stride and block can be used to 
        hsize_t block[5] = {1, 1, 1, 1, 1};  // add values to multiple places, useful if symmetry ever used.
        hsize_t zerostart[5] = {0, 0, 0, 0, 0};

#pragma omp parallel for collapse(4) num_threads(nthreads)
        for(auto s1 = 0; s1 != bs1.size(); ++s1) {
            for(auto s2 = 0; s2 != bs2.size(); ++s2) {
                for(auto s3 = 0; s3 != bs3.size(); ++s3) {
                    for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                        auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                        auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                        auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                        auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                        auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                        auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                        auto bf3 = shell2bf_3[s3];     // Index of first basis function in shell 3
                        auto atom3 = shell2atom_3[s3]; // Atom index of shell 3
                        auto n3 = bs3[s3].size();    // number of basis functions in shell 3
                        auto bf4 = shell2bf_4[s4];     // Index of first basis function in shell 4
                        auto atom4 = shell2atom_4[s4]; // Atom index of shell 4
                        auto n4 = bs4[s4].size();    // number of basis functions in shell 4

                        if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
                        std::vector<long> shell_atom_index_list{atom1, atom2, atom3, atom4};

                        size_t thread_id = 0;
#ifdef _OPENMP
                        thread_id = omp_get_thread_num();
#endif
                        cgtg_squared_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                        const auto& f12_squared_buffer = cgtg_squared_engines[thread_id].results(); // will point to computed shell sets

                        // Define shell set slab, with extra dimension for unique derivatives, initialized with 0.0's
                        double f12_squared_shellset_slab [n1][n2][n3][n4][nderivs_triu] = {};
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
                                auto f12_squared_shellset = f12_squared_buffer[buffer_indices[i]];
                                if (f12_squared_shellset == nullptr) continue;
                                for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                                    for(auto f2 = 0; f2 != n2; ++f2) {
                                        for(auto f3 = 0; f3 != n3; ++f3) {
                                            for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                                f12_squared_shellset_slab[f1][f2][f3][f4][nuc_idx] += f12_squared_shellset[idx];
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
                        f12_squared_dataset->write(f12_squared_shellset_slab, PredType::NATIVE_DOUBLE, mspace, fspace);
                    }
                }
            }
        } // shell quartet loops
    // Close the dataset for this derivative order
    delete f12_squared_dataset;
    } // deriv order loop 
// Close the file
delete file;
std::cout << " done" << std::endl;
} // f12_squared_deriv_disk function

// Writes all F12G12 ints up to `max_deriv_order` to disk.
// HDF5 File Name: f12g12_derivs.h5 
//      HDF5 Dataset names within the file:
//      f12g12_deriv1 
//          shape (nbf,nbf,nbf,nbf,n_unique_1st_derivs)
//      f12g12_deriv2
//          shape (nbf,nbf,nbf,nbf,n_unique_2nd_derivs)
//      f12g12_deriv3
//          shape (nbf,nbf,nbf,nbf,n_unique_3rd_derivs)
//      ...
void f12g12_deriv_disk(double beta, int max_deriv_order) { 
    std::cout << "Writing two-electron F12G12 integral derivative tensors up to order " << max_deriv_order << " to disk...";
    const H5std_string file_name("f12g12_derivs.h5");
    H5File* file = new H5File(file_name,H5F_ACC_TRUNC);
    double fillvalue = 0.0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);

    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));

    // Check to make sure you are not flooding the disk.
    long total_deriv_slices = 0;
    for (int i = 1; i <= max_deriv_order; i++){
        total_deriv_slices += how_many_derivs(natom, i);
    }
    double check = (nbf1 * nbf2 * nbf3 * nbf4 * total_deriv_slices * 8) * (1e-9);
    assert(check < 10 && "Total disk space required for ERI's exceeds 10 GB. Increase threshold and recompile to proceed.");

    auto cgtg_params = make_cgtg(beta);
    
    for (int deriv_order = 1; deriv_order <= max_deriv_order; deriv_order++){
        // Number of unique shell derivatives output by libint (number of indices in buffer)
        int nshell_derivs = how_many_derivs(4, deriv_order);
        // Number of unique nuclear derivatives of ERI's
        unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

        // Create mapping from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
        // Currently not used due to predefined lookup arrays
        //const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);

        // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
        const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

        // Libint engine for computing shell quartet derivatives
        std::vector<libint2::Engine> cgtg_coulomb_engines(nthreads);
        cgtg_coulomb_engines[0] = libint2::Engine(libint2::Operator::cgtg_x_coulomb, max_nprim, max_l, deriv_order);
        cgtg_coulomb_engines[0].set_params(cgtg_params);
        for (size_t i = 1; i != nthreads; ++i) {
            cgtg_coulomb_engines[i] = cgtg_coulomb_engines[0];
        }

        // Define HDF5 dataset name
        const H5std_string eri_dset_name("f12g12_deriv" + std::to_string(deriv_order));
        hsize_t file_dims[] = {nbf1, nbf2, nbf3, nbf4, nderivs_triu};
        DataSpace fspace(5, file_dims);
        // Create dataset for each integral type and write 0.0's into the file 
        DataSet* f12g12_dataset = new DataSet(file->createDataSet(eri_dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        hsize_t stride[5] = {1, 1, 1, 1, 1}; // stride and block can be used to 
        hsize_t block[5] = {1, 1, 1, 1, 1};  // add values to multiple places, useful if symmetry ever used.
        hsize_t zerostart[5] = {0, 0, 0, 0, 0};

#pragma omp parallel for collapse(4) num_threads(nthreads)
        for(auto s1 = 0; s1 != bs1.size(); ++s1) {
            for(auto s2 = 0; s2 != bs2.size(); ++s2) {
                for(auto s3 = 0; s3 != bs3.size(); ++s3) {
                    for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                        auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                        auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                        auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                        auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                        auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                        auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                        auto bf3 = shell2bf_3[s3];     // Index of first basis function in shell 3
                        auto atom3 = shell2atom_3[s3]; // Atom index of shell 3
                        auto n3 = bs3[s3].size();    // number of basis functions in shell 3
                        auto bf4 = shell2bf_4[s4];     // Index of first basis function in shell 4
                        auto atom4 = shell2atom_4[s4]; // Atom index of shell 4
                        auto n4 = bs4[s4].size();    // number of basis functions in shell 4

                        if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
                        std::vector<long> shell_atom_index_list{atom1, atom2, atom3, atom4};

                        size_t thread_id = 0;
#ifdef _OPENMP
                        thread_id = omp_get_thread_num();
#endif
                        cgtg_coulomb_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                        const auto& f12g12_buffer = cgtg_coulomb_engines[thread_id].results(); // will point to computed shell sets

                        // Define shell set slab, with extra dimension for unique derivatives, initialized with 0.0's
                        double f12g12_shellset_slab [n1][n2][n3][n4][nderivs_triu] = {};
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
                                auto f12g12_shellset = f12g12_buffer[buffer_indices[i]];
                                if (f12g12_shellset == nullptr) continue;
                                for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                                    for(auto f2 = 0; f2 != n2; ++f2) {
                                        for(auto f3 = 0; f3 != n3; ++f3) {
                                            for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                                f12g12_shellset_slab[f1][f2][f3][f4][nuc_idx] += f12g12_shellset[idx];
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
                        f12g12_dataset->write(f12g12_shellset_slab, PredType::NATIVE_DOUBLE, mspace, fspace);
                    }
                }
            }
        } // shell quartet loops
    // Close the dataset for this derivative order
    delete f12g12_dataset;
    } // deriv order loop 
// Close the file
delete file;
std::cout << " done" << std::endl;
} // f12g12_deriv_disk function

// Writes all F12 Double Commutator ints up to `max_deriv_order` to disk.
// HDF5 File Name: f12_derivs.h5 
//      HDF5 Dataset names within the file:
//      f12_double_commutator_deriv1 
//          shape (nbf,nbf,nbf,nbf,n_unique_1st_derivs)
//      f12_double_commutator_deriv2
//          shape (nbf,nbf,nbf,nbf,n_unique_2nd_derivs)
//      f12_double_commutator_deriv3
//          shape (nbf,nbf,nbf,nbf,n_unique_3rd_derivs)
//      ...
void f12_double_commutator_deriv_disk(double beta, int max_deriv_order) { 
    std::cout << "Writing two-electron F12 Double Commutator integral derivative tensors up to order " << max_deriv_order << " to disk...";
    const H5std_string file_name("f12_double_commutator_derivs.h5");
    H5File* file = new H5File(file_name,H5F_ACC_TRUNC);
    double fillvalue = 0.0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);

    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));

    // Check to make sure you are not flooding the disk.
    long total_deriv_slices = 0;
    for (int i = 1; i <= max_deriv_order; i++){
        total_deriv_slices += how_many_derivs(natom, i);
    }
    double check = (nbf1 * nbf2 * nbf3 * nbf4 * total_deriv_slices * 8) * (1e-9);
    assert(check < 10 && "Total disk space required for ERI's exceeds 10 GB. Increase threshold and recompile to proceed.");

    auto cgtg_params = make_cgtg(beta);
    
    for (int deriv_order = 1; deriv_order <= max_deriv_order; deriv_order++){
        // Number of unique shell derivatives output by libint (number of indices in buffer)
        int nshell_derivs = how_many_derivs(4, deriv_order);
        // Number of unique nuclear derivatives of ERI's
        unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

        // Create mapping from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
        // Currently not used due to predefined lookup arrays
        //const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);

        // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
        const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

        // Libint engine for computing shell quartet derivatives
        std::vector<libint2::Engine> cgtg_del_engines(nthreads);
        // Returns Runtime Error: bad any_cast if shorthand version is used, may be an error on the Libint side since Psi4 works with this as well
        cgtg_del_engines[0] = libint2::Engine(libint2::Operator::delcgtg2, max_nprim, max_l, deriv_order, 0., cgtg_params, libint2::BraKet::xx_xx);
        for (size_t i = 1; i != nthreads; ++i) {
            cgtg_del_engines[i] = cgtg_del_engines[0];
        }

        // Define HDF5 dataset name
        const H5std_string eri_dset_name("f12_double_commutator_deriv" + std::to_string(deriv_order));
        hsize_t file_dims[] = {nbf1, nbf2, nbf3, nbf4, nderivs_triu};
        DataSpace fspace(5, file_dims);
        // Create dataset for each integral type and write 0.0's into the file 
        DataSet* f12_double_commutator_dataset = new DataSet(file->createDataSet(eri_dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        hsize_t stride[5] = {1, 1, 1, 1, 1}; // stride and block can be used to 
        hsize_t block[5] = {1, 1, 1, 1, 1};  // add values to multiple places, useful if symmetry ever used.
        hsize_t zerostart[5] = {0, 0, 0, 0, 0};

#pragma omp parallel for collapse(4) num_threads(nthreads)
        for(auto s1 = 0; s1 != bs1.size(); ++s1) {
            for(auto s2 = 0; s2 != bs2.size(); ++s2) {
                for(auto s3 = 0; s3 != bs3.size(); ++s3) {
                    for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                        auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                        auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                        auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                        auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                        auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                        auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                        auto bf3 = shell2bf_3[s3];     // Index of first basis function in shell 3
                        auto atom3 = shell2atom_3[s3]; // Atom index of shell 3
                        auto n3 = bs3[s3].size();    // number of basis functions in shell 3
                        auto bf4 = shell2bf_4[s4];     // Index of first basis function in shell 4
                        auto atom4 = shell2atom_4[s4]; // Atom index of shell 4
                        auto n4 = bs4[s4].size();    // number of basis functions in shell 4

                        if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
                        std::vector<long> shell_atom_index_list{atom1, atom2, atom3, atom4};

                        size_t thread_id = 0;
#ifdef _OPENMP
                        thread_id = omp_get_thread_num();
#endif
                        cgtg_del_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                        const auto& f12_double_commutator_buffer = cgtg_del_engines[thread_id].results(); // will point to computed shell sets

                        // Define shell set slab, with extra dimension for unique derivatives, initialized with 0.0's
                        double f12_double_commutator_shellset_slab [n1][n2][n3][n4][nderivs_triu] = {};
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
                                auto f12_double_commutator_shellset = f12_double_commutator_buffer[buffer_indices[i]];
                                if (f12_double_commutator_shellset == nullptr) continue;
                                for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                                    for(auto f2 = 0; f2 != n2; ++f2) {
                                        for(auto f3 = 0; f3 != n3; ++f3) {
                                            for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                                f12_double_commutator_shellset_slab[f1][f2][f3][f4][nuc_idx] += f12_double_commutator_shellset[idx];
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
                        f12_double_commutator_dataset->write(f12_double_commutator_shellset_slab, PredType::NATIVE_DOUBLE, mspace, fspace);
                    }
                }
            }
        } // shell quartet loops
    // Close the dataset for this derivative order
    delete f12_double_commutator_dataset;
    } // deriv order loop 
// Close the file
delete file;
std::cout << " done" << std::endl;
} // f12_double_commutator_deriv_disk function

// Computes a single 'deriv_order' derivative tensor of OEIs, keeps everything in core memory
std::vector<py::array> oei_deriv_core(int deriv_order) {
    // how many shell derivatives in the Libint buffer for overlap/kinetic integrals
    // how many shell and operator derivatives for potential integrals
    int nshell_derivs = how_many_derivs(2, deriv_order);
    int nshell_derivs_potential = how_many_derivs(2, deriv_order, natom);
    // how many unique cartesian nuclear derivatives (e.g., so we only save one of d^2/dx1dx2 and d^2/dx2dx1, etc)
    unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

    // Create mappings from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
    // Overlap and kinetic have different mappings than potential since potential has more elements in the buffer
    // Currently unused due to predefined lookup arrays
    //const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(6, deriv_order);
    // Potential integrals buffer is flattened upper triangle of (6 + NCART) dimensional deriv_order tensor
    const std::vector<std::vector<int>> potential_buffer_multidim_lookup = generate_multi_index_lookup(6 + ncart, deriv_order);

    // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
    const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

    // Define engines and buffers
    std::vector<libint2::Engine> s_engines(nthreads), t_engines(nthreads), v_engines(nthreads);
    size_t max_nprim = std::max(bs1.max_nprim(), bs2.max_nprim());
    int max_l = std::max(bs1.max_l(), bs2.max_l());
    s_engines[0] = libint2::Engine(libint2::Operator::overlap, max_nprim, max_l, deriv_order);
    t_engines[0] = libint2::Engine(libint2::Operator::kinetic, max_nprim, max_l, deriv_order);
    v_engines[0] = libint2::Engine(libint2::Operator::nuclear, max_nprim, max_l, deriv_order);
    v_engines[0].set_params(make_point_charges(atoms));
    for (size_t i = 1; i != nthreads; ++i) {
        s_engines[i] = s_engines[0];
        t_engines[i] = t_engines[0];
        v_engines[i] = v_engines[0];
    }

    size_t length = nbf1 * nbf2 * nderivs_triu;
    std::vector<double> S(length);
    std::vector<double> T(length);
    std::vector<double> V(length);

#pragma omp parallel for collapse(2) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
            auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
            auto n1 = bs1[s1].size();    // number of basis functions in shell 1
            auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
            auto n2 = bs2[s2].size();    // number of basis functions in shell 2
            std::vector<long> shell_atom_index_list{atom1, atom2};

            size_t thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            s_engines[thread_id].compute(bs1[s1], bs2[s2]); // Compute shell set
            t_engines[thread_id].compute(bs1[s1], bs2[s2]); // Compute shell set
            v_engines[thread_id].compute(bs1[s1], bs2[s2]); // Compute shell set
            const auto& overlap_buffer = s_engines[thread_id].results(); // will point to computed shell sets
            const auto& kinetic_buffer = t_engines[thread_id].results(); // will point to computed shell sets
            const auto& potential_buffer = v_engines[thread_id].results(); // will point to computed shell sets

            // Loop over every possible unique nuclear cartesian derivative index (flattened upper triangle)
            // For 1st derivatives of 2 atom system, this is 6. 2nd derivatives of 2 atom system: 21, etc
            for(int nuc_idx = 0; nuc_idx < nderivs_triu; ++nuc_idx) {
                size_t offset_nuc_idx = nuc_idx * nbf1 * nbf2;

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
                            int tmp = 3 * (i + 2) + desired_coord;
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
                //for (auto vec : index_combos)  {
                //    std::sort(vec.begin(), vec.end());
                //    int buf_idx = 0;
                //    auto it = lower_bound(buffer_multidim_lookup.begin(), buffer_multidim_lookup.end(), vec);
                //    if (it != buffer_multidim_lookup.end()) buf_idx = it - buffer_multidim_lookup.begin();
                //    buffer_indices.push_back(buf_idx);
                //}
                for (auto vec : index_combos)  {
                    if (deriv_order == 1) buffer_indices.push_back(buffer_index_oei1d[vec[0]]);
                    else if (deriv_order == 2) buffer_indices.push_back(buffer_index_oei2d[vec[0]][vec[1]]);
                    else if (deriv_order == 3) buffer_indices.push_back(buffer_index_oei3d[vec[0]][vec[1]][vec[2]]);
                    else if (deriv_order == 4) buffer_indices.push_back(buffer_index_oei4d[vec[0]][vec[1]][vec[2]][vec[3]]);
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
                            S[(bf1 + f1) * nbf2 + bf2 + f2 + offset_nuc_idx] += overlap_shellset[idx];
                            T[(bf1 + f1) * nbf2 + bf2 + f2 + offset_nuc_idx] += kinetic_shellset[idx];
                        }
                    }
                }
                // Potential
                for(auto i = 0; i < potential_buffer_indices.size(); ++i) {
                    auto potential_shellset = potential_buffer[potential_buffer_indices[i]];
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                            V[(bf1 + f1) * nbf2 + bf2 + f2 + offset_nuc_idx] += potential_shellset[idx];
                        }
                    }
                }
            } // Unique nuclear cartesian derivative indices loop
        }
    } // shell duet loops
    return {py::array(S.size(), S.data()), py::array(T.size(), T.data()), py::array(V.size(), V.data())}; // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042 there's a workaround
} // oei_deriv_core function

// Computes a single 'deriv_order' derivative tensor of electron repulsion integrals, keeps everything in core memory
py::array eri_deriv_core(int deriv_order) {
    // Number of unique shell derivatives output by libint (number of indices in buffer)
    int nshell_derivs = how_many_derivs(4, deriv_order);
    // Number of unique nuclear derivatives of ERI's
    unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

    // Create mapping from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
    // Currently unused due to predefined lookup arrays
    //const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);

    // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
    const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

    // Libint engine for computing shell quartet derivatives
    std::vector<libint2::Engine> eri_engines(nthreads);
    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
    eri_engines[0] = libint2::Engine(libint2::Operator::coulomb, max_nprim, max_l, deriv_order);
    for (size_t i = 1; i != nthreads; ++i) {
        eri_engines[i] = eri_engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4 * nderivs_triu;
    std::vector<double> result(length);

    // Begin shell quartet loops
#pragma omp parallel for collapse(4) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            for(auto s3 = 0; s3 != bs3.size(); ++s3) {
                for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                    auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                    auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                    auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                    auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                    auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                    auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                    auto bf3 = shell2bf_3[s3];     // Index of first basis function in shell 3
                    auto atom3 = shell2atom_3[s3]; // Atom index of shell 3
                    auto n3 = bs3[s3].size();    // number of basis functions in shell 3
                    auto bf4 = shell2bf_4[s4];     // Index of first basis function in shell 4
                    auto atom4 = shell2atom_4[s4]; // Atom index of shell 4
                    auto n4 = bs4[s4].size();    // number of basis functions in shell 4

                    if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
                    std::vector<long> shell_atom_index_list{atom1, atom2, atom3, atom4};

                    size_t thread_id = 0;
#ifdef _OPENMP
                    thread_id = omp_get_thread_num();
#endif
                    eri_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                    const auto& eri_buffer = eri_engines[thread_id].results(); // will point to computed shell sets

                    // Loop over every possible unique nuclear cartesian derivative index (flattened upper triangle)
                    for(int nuc_idx = 0; nuc_idx < nderivs_triu; ++nuc_idx) {
                        size_t offset_nuc_idx = nuc_idx * nbf1 * nbf2 * nbf3 * nbf4;

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
                                size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                                for(auto f2 = 0; f2 != n2; ++f2) {
                                    size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                                    for(auto f3 = 0; f3 != n3; ++f3) {
                                        size_t offset_3 = (bf3 + f3) * nbf4;
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

// Computes a single 'deriv_order' derivative tensor of contracted Gaussian-type geminal integrals, keeps everything in core memory
py::array f12_deriv_core(double beta, int deriv_order) {
    // Number of unique shell derivatives output by libint (number of indices in buffer)
    int nshell_derivs = how_many_derivs(4, deriv_order);
    // Number of unique nuclear derivatives of ERI's
    unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

    // Create mapping from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
    // Currently unused due to predefined lookup arrays
    //const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);

    // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
    const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

    // Libint engine for computing shell quartet derivatives
    auto cgtg_params = make_cgtg(beta);
    std::vector<libint2::Engine> cgtg_engines(nthreads);
    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
    cgtg_engines[0] = libint2::Engine(libint2::Operator::cgtg, max_nprim, max_l, deriv_order);
    cgtg_engines[0].set_params(cgtg_params);
    for (size_t i = 1; i != nthreads; ++i) {
        cgtg_engines[i] = cgtg_engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4 * nderivs_triu;
    std::vector<double> result(length);

    // Begin shell quartet loops
#pragma omp parallel for collapse(4) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            for(auto s3 = 0; s3 != bs3.size(); ++s3) {
                for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                    auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                    auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                    auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                    auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                    auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                    auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                    auto bf3 = shell2bf_3[s3];     // Index of first basis function in shell 3
                    auto atom3 = shell2atom_3[s3]; // Atom index of shell 3
                    auto n3 = bs3[s3].size();    // number of basis functions in shell 3
                    auto bf4 = shell2bf_4[s4];     // Index of first basis function in shell 4
                    auto atom4 = shell2atom_4[s4]; // Atom index of shell 4
                    auto n4 = bs4[s4].size();    // number of basis functions in shell 4

                    if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
                    std::vector<long> shell_atom_index_list{atom1, atom2, atom3, atom4};

                    size_t thread_id = 0;
#ifdef _OPENMP
                    thread_id = omp_get_thread_num();
#endif
                    cgtg_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                    const auto& f12_buffer = cgtg_engines[thread_id].results(); // will point to computed shell sets

                    // Loop over every possible unique nuclear cartesian derivative index (flattened upper triangle)
                    for(int nuc_idx = 0; nuc_idx < nderivs_triu; ++nuc_idx) {
                        size_t offset_nuc_idx = nuc_idx * nbf1 * nbf2 * nbf3 * nbf4;

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
                            auto f12_shellset = f12_buffer[buffer_indices[i]];
                            if (f12_shellset == nullptr) continue;
                            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                                size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                                for(auto f2 = 0; f2 != n2; ++f2) {
                                    size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                                    for(auto f3 = 0; f3 != n3; ++f3) {
                                        size_t offset_3 = (bf3 + f3) * nbf4;
                                        for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                            size_t offset_4 = bf4 + f4;
                                            result[offset_1 + offset_2 + offset_3 + offset_4 + offset_nuc_idx] += f12_shellset[idx];
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
} // f12_deriv_core function

// Computes a single 'deriv_order' derivative tensor of squared contracted Gaussian-type geminal integrals, keeps everything in core memory
py::array f12_squared_deriv_core(double beta, int deriv_order) {
    // Number of unique shell derivatives output by libint (number of indices in buffer)
    int nshell_derivs = how_many_derivs(4, deriv_order);
    // Number of unique nuclear derivatives of ERI's
    unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

    // Create mapping from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
    // Currently unused due to predefined lookup arrays
    //const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);

    // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
    const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

    // Libint engine for computing shell quartet derivatives
    auto cgtg_params = take_square(make_cgtg(beta));
    std::vector<libint2::Engine> cgtg_squared_engines(nthreads);
    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
    cgtg_squared_engines[0] = libint2::Engine(libint2::Operator::cgtg, max_nprim, max_l, deriv_order);
    cgtg_squared_engines[0].set_params(cgtg_params);
    for (size_t i = 1; i != nthreads; ++i) {
        cgtg_squared_engines[i] = cgtg_squared_engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4 * nderivs_triu;
    std::vector<double> result(length);

    // Begin shell quartet loops
#pragma omp parallel for collapse(4) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            for(auto s3 = 0; s3 != bs3.size(); ++s3) {
                for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                    auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                    auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                    auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                    auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                    auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                    auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                    auto bf3 = shell2bf_3[s3];     // Index of first basis function in shell 3
                    auto atom3 = shell2atom_3[s3]; // Atom index of shell 3
                    auto n3 = bs3[s3].size();    // number of basis functions in shell 3
                    auto bf4 = shell2bf_4[s4];     // Index of first basis function in shell 4
                    auto atom4 = shell2atom_4[s4]; // Atom index of shell 4
                    auto n4 = bs4[s4].size();    // number of basis functions in shell 4

                    if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
                    std::vector<long> shell_atom_index_list{atom1, atom2, atom3, atom4};

                    size_t thread_id = 0;
#ifdef _OPENMP
                    thread_id = omp_get_thread_num();
#endif
                    cgtg_squared_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                    const auto& f12_squared_buffer = cgtg_squared_engines[thread_id].results(); // will point to computed shell sets

                    // Loop over every possible unique nuclear cartesian derivative index (flattened upper triangle)
                    for(int nuc_idx = 0; nuc_idx < nderivs_triu; ++nuc_idx) {
                        size_t offset_nuc_idx = nuc_idx * nbf1 * nbf2 * nbf3 * nbf4;

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
                            auto f12_squared_shellset = f12_squared_buffer[buffer_indices[i]];
                            if (f12_squared_shellset == nullptr) continue;
                            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                                size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                                for(auto f2 = 0; f2 != n2; ++f2) {
                                    size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                                    for(auto f3 = 0; f3 != n3; ++f3) {
                                        size_t offset_3 = (bf3 + f3) * nbf4;
                                        for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                            size_t offset_4 = bf4 + f4;
                                            result[offset_1 + offset_2 + offset_3 + offset_4 + offset_nuc_idx] += f12_squared_shellset[idx];
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
} // f12_squared_deriv_core function

// Computes a single 'deriv_order' derivative tensor of contracted Gaussian-type geminal times Coulomb replusion integrals, keeps everything in core memory
py::array f12g12_deriv_core(double beta, int deriv_order) {
    // Number of unique shell derivatives output by libint (number of indices in buffer)
    int nshell_derivs = how_many_derivs(4, deriv_order);
    // Number of unique nuclear derivatives of ERI's
    unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

    // Create mapping from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
    // Currently unused due to predefined lookup arrays
    //const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);

    // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
    const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

    // Libint engine for computing shell quartet derivatives
    auto cgtg_params = make_cgtg(beta);
    std::vector<libint2::Engine> cgtg_coulomb_engines(nthreads);
    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
    cgtg_coulomb_engines[0] = libint2::Engine(libint2::Operator::cgtg_x_coulomb, max_nprim, max_l, deriv_order);
    cgtg_coulomb_engines[0].set_params(cgtg_params);
    for (size_t i = 1; i != nthreads; ++i) {
        cgtg_coulomb_engines[i] = cgtg_coulomb_engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4 * nderivs_triu;
    std::vector<double> result(length);

    // Begin shell quartet loops
#pragma omp parallel for collapse(4) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            for(auto s3 = 0; s3 != bs3.size(); ++s3) {
                for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                    auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                    auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                    auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                    auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                    auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                    auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                    auto bf3 = shell2bf_3[s3];     // Index of first basis function in shell 3
                    auto atom3 = shell2atom_3[s3]; // Atom index of shell 3
                    auto n3 = bs3[s3].size();    // number of basis functions in shell 3
                    auto bf4 = shell2bf_4[s4];     // Index of first basis function in shell 4
                    auto atom4 = shell2atom_4[s4]; // Atom index of shell 4
                    auto n4 = bs4[s4].size();    // number of basis functions in shell 4

                    if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
                    std::vector<long> shell_atom_index_list{atom1, atom2, atom3, atom4};

                    size_t thread_id = 0;
#ifdef _OPENMP
                    thread_id = omp_get_thread_num();
#endif
                    cgtg_coulomb_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                    const auto& f12g12_buffer = cgtg_coulomb_engines[thread_id].results(); // will point to computed shell sets

                    // Loop over every possible unique nuclear cartesian derivative index (flattened upper triangle)
                    for(int nuc_idx = 0; nuc_idx < nderivs_triu; ++nuc_idx) {
                        size_t offset_nuc_idx = nuc_idx * nbf1 * nbf2 * nbf3 * nbf4;

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
                            auto f12g12_shellset = f12g12_buffer[buffer_indices[i]];
                            if (f12g12_shellset == nullptr) continue;
                            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                                size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                                for(auto f2 = 0; f2 != n2; ++f2) {
                                    size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                                    for(auto f3 = 0; f3 != n3; ++f3) {
                                        size_t offset_3 = (bf3 + f3) * nbf4;
                                        for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                            size_t offset_4 = bf4 + f4;
                                            result[offset_1 + offset_2 + offset_3 + offset_4 + offset_nuc_idx] += f12g12_shellset[idx];
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
} // f12g12_deriv_core function

// Computes a single 'deriv_order' derivative tensor of gradient norm of contracted Gaussian-type geminal integrals, keeps everything in core memory
py::array f12_double_commutator_deriv_core(double beta, int deriv_order) {
    // Number of unique shell derivatives output by libint (number of indices in buffer)
    int nshell_derivs = how_many_derivs(4, deriv_order);
    // Number of unique nuclear derivatives of ERI's
    unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

    // Create mapping from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
    // Currently unused due to predefined lookup arrays
    //const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);

    // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
    const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

    // Libint engine for computing shell quartet derivatives
    auto cgtg_params = make_cgtg(beta);
    std::vector<libint2::Engine> cgtg_del_engines(nthreads);
    size_t max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()), std::max(bs3.max_nprim(), bs4.max_nprim()));
    int max_l = std::max(std::max(bs1.max_l(), bs2.max_l()), std::max(bs3.max_l(), bs4.max_l()));
    // Returns Runtime Error: bad any_cast if shorthand version is used, may be an error on the Libint side since Psi4 works with this as well
    cgtg_del_engines[0] = libint2::Engine(libint2::Operator::delcgtg2, max_nprim, max_l, deriv_order, 0., cgtg_params, libint2::BraKet::xx_xx);
    for (size_t i = 1; i != nthreads; ++i) {
        cgtg_del_engines[i] = cgtg_del_engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4 * nderivs_triu;
    std::vector<double> result(length);

    // Begin shell quartet loops
#pragma omp parallel for collapse(4) num_threads(nthreads)
    for(auto s1 = 0; s1 != bs1.size(); ++s1) {
        for(auto s2 = 0; s2 != bs2.size(); ++s2) {
            for(auto s3 = 0; s3 != bs3.size(); ++s3) {
                for(auto s4 = 0; s4 != bs4.size(); ++s4) {
                    auto bf1 = shell2bf_1[s1];     // Index of first basis function in shell 1
                    auto atom1 = shell2atom_1[s1]; // Atom index of shell 1
                    auto n1 = bs1[s1].size();    // number of basis functions in shell 1
                    auto bf2 = shell2bf_2[s2];     // Index of first basis function in shell 2
                    auto atom2 = shell2atom_2[s2]; // Atom index of shell 2
                    auto n2 = bs2[s2].size();    // number of basis functions in shell 2
                    auto bf3 = shell2bf_3[s3];     // Index of first basis function in shell 3
                    auto atom3 = shell2atom_3[s3]; // Atom index of shell 3
                    auto n3 = bs3[s3].size();    // number of basis functions in shell 3
                    auto bf4 = shell2bf_4[s4];     // Index of first basis function in shell 4
                    auto atom4 = shell2atom_4[s4]; // Atom index of shell 4
                    auto n4 = bs4[s4].size();    // number of basis functions in shell 4

                    if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
                    std::vector<long> shell_atom_index_list{atom1, atom2, atom3, atom4};

                    size_t thread_id = 0;
#ifdef _OPENMP
                    thread_id = omp_get_thread_num();
#endif
                    cgtg_del_engines[thread_id].compute(bs1[s1], bs2[s2], bs3[s3], bs4[s4]); // Compute shell set
                    const auto& f12_double_commutator_buffer = cgtg_del_engines[thread_id].results(); // will point to computed shell sets

                    // Loop over every possible unique nuclear cartesian derivative index (flattened upper triangle)
                    for(int nuc_idx = 0; nuc_idx < nderivs_triu; ++nuc_idx) {
                        size_t offset_nuc_idx = nuc_idx * nbf1 * nbf2 * nbf3 * nbf4;

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
                            auto f12_double_commutator_shellset = f12_double_commutator_buffer[buffer_indices[i]];
                            if (f12_double_commutator_shellset == nullptr) continue;
                            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                                size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                                for(auto f2 = 0; f2 != n2; ++f2) {
                                    size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                                    for(auto f3 = 0; f3 != n3; ++f3) {
                                        size_t offset_3 = (bf3 + f3) * nbf4;
                                        for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                            size_t offset_4 = bf4 + f4;
                                            result[offset_1 + offset_2 + offset_3 + offset_4 + offset_nuc_idx] += f12_double_commutator_shellset[idx];
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
} // f12_double_commutator_deriv_core function

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
    m.def("f12", &f12, "Computes contracted Gaussian-type geminal integrals with libint");
    m.def("f12_squared", &f12_squared, "Computes sqaured contracted Gaussian-type geminal integrals with libint");
    m.def("f12g12", &f12g12, "Computes contracted Gaussian-type geminal times Coulomb repulsion integrals with libint");
    m.def("f12_double_commutator", &f12_double_commutator, "Computes gradient norm of contracted Gaussian-type geminal integrals with libint");
    m.def("overlap_deriv", &overlap_deriv, "Computes overlap integral nuclear derivatives with libint");
    m.def("kinetic_deriv", &kinetic_deriv, "Computes kinetic integral nuclear derivatives with libint");
    m.def("potential_deriv", &potential_deriv, "Computes potential integral nuclear derivatives with libint");
    m.def("eri_deriv", &eri_deriv, "Computes electron repulsion integral nuclear derivatives with libint");
    m.def("f12_deriv", &f12_deriv, "Computes contracted Gaussian-type geminal integral nuclear derivatives with libint");
    m.def("f12_squared_deriv", &f12_squared_deriv, "Computes sqaured contracted Gaussian-type geminal integral nuclear derivatives with libint");
    m.def("f12g12_deriv", &f12g12_deriv, "Computes contracted Gaussian-type geminal times Coulomb repulsion integral nuclear derivatives with libint");
    m.def("f12_double_commutator_deriv", &f12_double_commutator_deriv, "Computes gradient norm of contracted Gaussian-type geminal integral nuclear derivatives with libint");
    m.def("oei_deriv_disk", &oei_deriv_disk, "Computes overlap, kinetic, and potential integral derivative tensors from 1st order up to nth order and writes them to disk with HDF5");
    m.def("eri_deriv_disk", &eri_deriv_disk, "Computes coulomb integral nuclear derivative tensors from 1st order up to nth order and writes them to disk with HDF5");
    m.def("f12_deriv_disk", &f12_deriv_disk, "Computes contracted Gaussian-type geminal integral nuclear derivative tensors from 1st order up to nth order and writes them to disk with HDF5");
    m.def("f12_squared_deriv_disk", &f12_squared_deriv_disk, "Computes sqaured contracted Gaussian-type geminal integral nuclear derivative tensors from 1st order up to nth order and writes them to disk with HDF5");
    m.def("f12g12_deriv_disk", &f12g12_deriv_disk, "Computes contracted Gaussian-type geminal times Coulomb repulsion integral nuclear derivative tensors from 1st order up to nth order and writes them to disk with HDF5");
    m.def("f12_double_commutator_deriv_disk", &f12_double_commutator_deriv_disk, "Computes gradient norm of contracted Gaussian-type geminal integral nuclear derivative tensors from 1st order up to nth order and writes them to disk with HDF5");
    m.def("oei_deriv_core", &oei_deriv_core, "Computes a single OEI integral derivative tensor, in memory.");
    m.def("eri_deriv_core", &eri_deriv_core, "Computes a single coulomb integral nuclear derivative tensor, in memory.");
    m.def("f12_deriv_core", &f12_deriv_core, "Computes a single contracted Gaussian-type geminal integral nuclear derivative tensor, in memory.");
    m.def("f12_squared_deriv_core", &f12_squared_deriv_core, "Computes a single sqaured contracted Gaussian-type geminal integral nuclear derivative tensor, in memory.");
    m.def("f12g12_deriv_core", &f12g12_deriv_core, "Computes a single contracted Gaussian-type geminal times Coulomb repulsion integral nuclear derivative tensor, in memory.");
    m.def("f12_double_commutator_deriv_core", &f12_double_commutator_deriv_core, "Computes a single gradient norm of contracted Gaussian-type geminal integral nuclear derivative tensor, in memory.");
    //TODO partial derivative impl's
    //m.def("eri_partial_deriv_disk", &eri_partial_deriv_disk, "Computes a subset of the full coulomb integral nuclear derivative tensor and writes them to disk with HDF5");
     m.attr("LIBINT2_MAX_DERIV_ORDER") = LIBINT2_MAX_DERIV_ORDER;
}

