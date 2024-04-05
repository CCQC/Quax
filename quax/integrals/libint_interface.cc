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

// TODO support spherical harmonic gaussians, implement symmetry considerations, support 5th, 6th derivs

namespace py = pybind11;
using namespace H5;

/*Global variable, OpenMP lock*/
omp_lock_t  lock;

std::vector<libint2::Atom> atoms;
unsigned int natom;
unsigned int ncart;
libint2::BasisSet bs1, bs2, bs3, bs4;
unsigned int nbf1, nbf2, nbf3, nbf4;
std::vector<size_t> shell2bf_1, shell2bf_2, shell2bf_3, shell2bf_4;
std::vector<long> shell2atom_1, shell2atom_2, shell2atom_3, shell2atom_4;
size_t max_nprim;
int max_l;
int nthreads = 1;
double threshold;
double max_engine_precision;

// Creates atom objects from xyz file path
std::vector<libint2::Atom> get_atoms(std::string xyzfilename) 
{
    std::ifstream input_file(xyzfilename);
    std::vector<libint2::Atom> atoms = libint2::read_dotxyz(input_file);
    return atoms;
}

// Creates a combined basis set
libint2::BasisSet make_ao_cabs(std::string obs_name, libint2::BasisSet cabs) {
    // Create OBS
    obs_name.erase(obs_name.end() - 5, obs_name.end());
    auto obs = libint2::BasisSet(obs_name, atoms);
    obs.set_pure(false); // use cartesian gaussians

    auto obs_idx = obs.atom2shell(atoms);
    auto cabs_idx = cabs.atom2shell(atoms);

    std::vector<std::vector<libint2::Shell>> el_bases(36); // Only consider atoms up to Kr
    for (size_t i = 0; i < atoms.size(); i++) {
        if (el_bases[atoms[i].atomic_number].empty()) {
            std::vector<libint2::Shell> tmp;

            for(long int& idx : obs_idx[i]) {
                tmp.push_back(obs[idx]);
            }
            for(long int& idx : cabs_idx[i]) {
                tmp.push_back(cabs[idx]);
            }

            stable_sort(tmp.begin(), tmp.end(), [](const auto& a, const auto& b) -> bool
            {
                return a.contr[0].l < b.contr[0].l;
            });

            el_bases[atoms[i].atomic_number] = tmp;
        }
    }

    // Create CABS, union of orbital and auxiliary basis AOs
    cabs = libint2::BasisSet(atoms, el_bases);
    cabs.set_pure(false);
    return cabs;
}

// Must call initialize before computing ints 
void initialize(std::string xyzfilename, std::string basis1, std::string basis2,
                std::string basis3, std::string basis4, double ints_tol) {
    libint2::initialize();
    atoms = get_atoms(xyzfilename);
    natom = atoms.size();
    ncart = natom * 3;

    threshold = ints_tol;

    // Move harddrive load of basis and xyz to happen only once
    bs1 = libint2::BasisSet(basis1, atoms);
    bs1.set_pure(false); // use cartesian gaussians
    if (basis1.find("-cabs", 10) != std::string::npos) {
        bs1 = make_ao_cabs(basis1, bs1);
    }

    bs2 = libint2::BasisSet(basis2, atoms);
    bs2.set_pure(false); // use cartesian gaussians
    if (basis2.find("-cabs", 10) != std::string::npos) {
        bs2 = make_ao_cabs(basis2, bs2);
    }

    bs3 = libint2::BasisSet(basis3, atoms);
    bs3.set_pure(false); // use cartesian gaussians
    if (basis3.find("-cabs", 10) != std::string::npos) {
        bs3 = make_ao_cabs(basis3, bs3);
    }

    bs4 = libint2::BasisSet(basis4, atoms);
    bs4.set_pure(false); // use cartesian gaussians
    if (basis4.find("-cabs", 10) != std::string::npos) {
        bs4 = make_ao_cabs(basis4, bs4);
    }

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

    max_nprim = std::max(std::max(bs1.max_nprim(), bs2.max_nprim()),
                         std::max(bs3.max_nprim(), bs4.max_nprim()));
    max_l = std::max(std::max(bs1.max_l(), bs2.max_l()),
                     std::max(bs3.max_l(), bs4.max_l()));
    max_engine_precision = std::log(std::numeric_limits<double>::epsilon() * threshold);

    // Get number of OMP threads
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
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
// For example, if molecule has two atoms, A and B, and we want nuclear derivative d^2/dAz dBz,
// represented by deriv_vec = [0,0,1,0,0,1], and we are looping over 4 shells in ERI's,
// and the four shells are atoms (0,0,1,1), then possible indices 
// of the 0-11 shell cartesian component indices are {2,5} for d/dAz and {8,11} for d/dBz.
// So the vector passed to cartesian_product is { {{2,5},{8,11}}, and all combinations of elements
// from first and second subvectors are produced, and the total nuclear derivative of the shell
// is obtained by summing all of these pieces together.
// These resulting indices are converted to flattened Libint buffer indices using the generate_*_lookup functions,
// explained below.
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

// Computes non-negligible shell pair list for one-electron integrals
std::vector<std::pair<int, int>> build_shellpairs(libint2::BasisSet A, libint2::BasisSet B) {
    const auto A_equiv_B = (A == B);

    // construct the 2-electron repulsion integrals engine
    std::vector<libint2::Engine> engines(nthreads);
    engines[0] = libint2::Engine(libint2::Operator::overlap, max_nprim, max_l);
    engines[0].set_precision(0.);
    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }

    std::vector<std::vector<std::pair<int, int>>> threads_sp_list(nthreads);
    double threshold_sq = threshold * threshold;

    #pragma omp parallel num_threads(nthreads)
    {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        auto &engine = engines[thread_id];
        const auto &buf = engine.results();

        // loop over permutationally-unique set of shells
        for (auto s1 = 0l, s12 = 0l; s1 != A.size(); ++s1) {
            auto n1 = A[s1].size();

            auto s2_max = A_equiv_B ? s1 : B.size() - 1;
            for (auto s2 = 0; s2 <= s2_max; ++s2, ++s12) {
                if (s12 % nthreads != thread_id) continue;

                auto on_same_center = (A[s1].O == B[s2].O);
                bool significant = on_same_center;
                if (!on_same_center) {
                    auto n2 = B[s2].size();
                    engines[thread_id].compute(A[s1], B[s2]);
                    double normsq = std::inner_product(buf[0], buf[0] + n1 * n2, buf[0], 0.0); // Frobenius Norm
                    significant = (normsq >= threshold_sq);
                }

                if (significant)
                    threads_sp_list[thread_id].push_back(std::make_pair(s1, s2));
            }
        }
    }  // end of compute

    for (int thread = 1; thread < nthreads; ++thread) {
        for (const auto &pair : threads_sp_list[thread]) {
            threads_sp_list[0].push_back(pair);
        }
    }

    return threads_sp_list[0];
}

// Schwarz-Screening of two-electron integrals
std::tuple<std::vector<std::pair<int, int>>, std::vector<double>> schwarz_screening(libint2::BasisSet A, libint2::BasisSet B){

    const auto A_equiv_B = (A == B);

    // construct the 2-electron repulsion integrals engine
    std::vector<libint2::Engine> engines(nthreads);
    engines[0] = libint2::Engine(libint2::Operator::coulomb, max_nprim, max_l);
    engines[0].set_precision(0.);
    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }

    std::vector<double> shell_pair_values(A.size() * B.size());
    double max_integral = 0.0;

    // loop over permutationally-unique set of shells
    #pragma omp parallel num_threads(nthreads)
    {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif

        // loop over permutationally-unique set of shells
        for (auto s1 = 0l, s12 = 0l; s1 != A.size(); ++s1) {
            auto n1 = A[s1].size();

            auto s2_max = A_equiv_B ? s1 : B.size() - 1;
            for (auto s2 = 0; s2 <= s2_max; ++s2, ++s12) {
                if (s12 % nthreads != thread_id) continue;

                auto n2 = B[s2].size();

                engines[thread_id].compute(A[s1], B[s2], A[s1], B[s2]);
                const double * buffer = const_cast<double *>(engines[thread_id].results()[0]);

                if (buffer == nullptr) continue;

                double shell_max_val = 0.0;
                for (int f1 = 0; f1 != n1; f1++) {
                    for (int f2 = 0; f2 != n2; f2++) {
                        shell_max_val =
                            std::max(shell_max_val, std::fabs(buffer[f1 * (n1 * n2 * n2 + n2) + f2 * (n1 * n2 + 1)]));
                    }
                }

                max_integral = std::max(max_integral, shell_max_val);

                if (A_equiv_B) {
                    shell_pair_values[s1 * B.size() + s2] = shell_pair_values[s2 * A.size() + s1] = shell_max_val;
                } else {
                    shell_pair_values[s1 * B.size() + s2] = shell_max_val;
                }
            }
        }
    }

    double threshold_sq = threshold * threshold;
    double threshold_sq_over_max = threshold_sq / max_integral;

    std::vector<std::pair<int, int>> shell_pairs;

    for (auto s1 = 0l, s12 = 0l; s1 != A.size(); ++s1) {
        auto s2_max = A_equiv_B ? s1 : B.size() - 1;
        for (auto s2 = 0; s2 <= s2_max; ++s2, ++s12) {
            if (shell_pair_values[s1 * B.size() + s2] >= threshold_sq_over_max)
                shell_pairs.push_back(std::make_pair(s1, s2));
        }
    }

    return std::make_tuple(shell_pairs, shell_pair_values);
}

// Compute one-electron integral
py::array compute_1e_int(std::string type) {
    // Shell pairs after screening
    const auto bs1_equiv_bs2 = (bs1 == bs2);
    auto shellpairs = build_shellpairs(bs1, bs2);

    // Integral engine
    std::vector<libint2::Engine> engines(nthreads);
    
    if (type == "overlap") {
        engines[0] = libint2::Engine(libint2::Operator::overlap, max_nprim, max_l);
    } else if (type == "kinetic") {
        engines[0] = libint2::Engine(libint2::Operator::kinetic, max_nprim, max_l);
    } else if (type == "potential") {
        engines[0] = libint2::Engine(libint2::Operator::nuclear, max_nprim, max_l);
        engines[0].set_params(make_point_charges(atoms));
    } else {
       throw std::invalid_argument("type must be overlap, kinetic, or potential");
    }

    engines[0].set_precision(max_engine_precision);
    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }

    size_t length = nbf1 * nbf2;
    std::vector<double> result(length); // vector to store integral array

#pragma omp parallel for num_threads(nthreads)
    for (const auto &pair : shellpairs) {
        int p1 = pair.first;
        int p2 = pair.second;

        const auto &s1 = bs1[p1];
        const auto &s2 = bs2[p2];
        auto n1 = bs1[p1].size(); // number of basis functions in first shell
        auto n2 = bs2[p2].size(); // number of basis functions in first shell
        auto bf1 = shell2bf_1[p1];  // first basis function in first shell
        auto bf2 = shell2bf_2[p2];  // first basis function in second shell

        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        engines[thread_id].compute(s1, s2); // Compute shell set
        const auto& buf_vec = engines[thread_id].results(); // will point to computed shell sets

        auto ints_shellset = buf_vec[0];    // Location of the computed integrals
        if (ints_shellset == nullptr)
            continue;  // nullptr returned if the entire shell-set was screened out

        // Loop over shell block, keeping a total count idx for the size of shell set
        if (bs1_equiv_bs2 && p1 != p2) {
            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                    result[(bf1 + f1) * nbf2 + bf2 + f2] = ints_shellset[idx];
                    result[(bf2 + f2) * nbf1 + bf1 + f1] = ints_shellset[idx];
                }
            }
        } else {
            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                    result[(bf1 + f1) * nbf2 + bf2 + f2] = ints_shellset[idx];
                }
            }
        }
    }
    return py::array(result.size(), result.data()); 
}

// Compute one-electron dipole integrals
std::vector<py::array> compute_dipole_ints() {
    // Shell pairs after screening
    const auto bs1_equiv_bs2 = (bs1 == bs2);
    auto shellpairs = build_shellpairs(bs1, bs2);

    // Integral engine
    std::vector<libint2::Engine> engines(nthreads);

    // COM generator
    std::array<double,3> COM = {0.000, 0.000, 0.000};

    // Will compute overlap + electric dipole moments
    engines[0] = libint2::Engine(libint2::Operator::emultipole1, max_nprim, max_l);
    engines[0].set_params(COM); // with COM as the multipole origin
    engines[0].set_precision(max_engine_precision);
    engines[0].prescale_by(-1);
    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }

    size_t length = nbf1 * nbf2;
    std::vector<double> Mu_X(length); // Mu_X Vector
    std::vector<double> Mu_Y(length); // Mu_Y Vector
    std::vector<double> Mu_Z(length); // Mu_Z Vector

#pragma omp parallel for num_threads(nthreads)
    for (const auto &pair : shellpairs) {
        int p1 = pair.first;
        int p2 = pair.second;

        const auto &s1 = bs1[p1];
        const auto &s2 = bs2[p2];
        auto n1 = bs1[p1].size(); // number of basis functions in first shell
        auto n2 = bs2[p2].size(); // number of basis functions in first shell
        auto bf1 = shell2bf_1[p1];  // first basis function in first shell
        auto bf2 = shell2bf_2[p2];  // first basis function in second shell

        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        engines[thread_id].compute(s1, s2); // Compute shell set
        const auto& buf_vec = engines[thread_id].results(); // will point to computed shell sets
        auto mu_x_shellset = buf_vec[1];
        auto mu_y_shellset = buf_vec[2];
        auto mu_z_shellset = buf_vec[3];

        if (mu_x_shellset == nullptr && mu_y_shellset == nullptr && mu_z_shellset == nullptr)
            continue;  // nullptr returned if the entire shell-set was screened out

        // Loop over shell block, keeping a total count idx for the size of shell set
        if (bs1_equiv_bs2 && p1 != p2) {
            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                    Mu_X[(bf1 + f1) * nbf2 + bf2 + f2] = mu_x_shellset[idx];
                    Mu_X[(bf2 + f2) * nbf1 + bf1 + f1] = mu_x_shellset[idx];
                    Mu_Y[(bf1 + f1) * nbf2 + bf2 + f2] = mu_y_shellset[idx];
                    Mu_Y[(bf2 + f2) * nbf1 + bf1 + f1] = mu_y_shellset[idx];
                    Mu_Z[(bf1 + f1) * nbf2 + bf2 + f2] = mu_z_shellset[idx];
                    Mu_Z[(bf2 + f2) * nbf1 + bf1 + f1] = mu_z_shellset[idx];
                }
            }
        } else {
            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                    Mu_X[(bf1 + f1) * nbf2 + bf2 + f2] = mu_x_shellset[idx];
                    Mu_Y[(bf1 + f1) * nbf2 + bf2 + f2] = mu_y_shellset[idx];
                    Mu_Z[(bf1 + f1) * nbf2 + bf2 + f2] = mu_z_shellset[idx];
                }
            }
        }
    }
    return {py::array(Mu_X.size(), Mu_X.data()), py::array(Mu_Y.size(), Mu_Y.data()),
            py::array(Mu_Z.size(), Mu_Z.data())};
}

// Computes two-electron integrals
py::array compute_2e_int(std::string type, double beta) {
    // Shell screening
    std::vector<std::pair<int, int>> shellpairs_bra, shellpairs_ket;
    std::vector<double> schwarz_bra, schwarz_ket;
    const auto bs1_equiv_bs2 = (bs1 == bs2);
    const auto bs3_equiv_bs4 = (bs3 == bs4);
    std::tie(shellpairs_bra, schwarz_bra) = schwarz_screening(bs1, bs2);
    std::tie(shellpairs_ket, schwarz_ket) = schwarz_screening(bs3, bs4);
    auto threshold_sq = threshold * threshold;

    // workaround for data copying: perhaps pass an empty numpy array, then populate it in C++?
    // avoids last line, which copies
    std::vector<libint2::Engine> engines(nthreads);

    if (type == "eri") {
        engines[0] = libint2::Engine(libint2::Operator::coulomb, max_nprim, max_l);
    } else if (type == "f12") {
        auto cgtg_params = make_cgtg(beta);
        engines[0] = libint2::Engine(libint2::Operator::cgtg, max_nprim, max_l);
        engines[0].set_params(cgtg_params);
    } else if (type == "f12g12") {
        auto cgtg_params = make_cgtg(beta);
        engines[0] = libint2::Engine(libint2::Operator::cgtg_x_coulomb, max_nprim, max_l);
        engines[0].set_params(cgtg_params);
    } else if (type == "f12_squared") {
        auto cgtg_params = take_square(make_cgtg(beta));
        engines[0] = libint2::Engine(libint2::Operator::cgtg, max_nprim, max_l);
        engines[0].set_params(cgtg_params);
    } else if (type == "f12_double_commutator") {
        auto cgtg_params = make_cgtg(beta);
        engines[0] = libint2::Engine(libint2::Operator::delcgtg2, max_nprim, max_l, 0,
                                            std::numeric_limits<libint2::scalar_type>::epsilon(),
                                            cgtg_params, libint2::BraKet::xx_xx);
    } else {
        throw std::invalid_argument("type must be eri, f12, f12g12, f12_squared, or f12_double_commutator");
    }

    engines[0].set_precision(max_engine_precision);
    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4;
    std::vector<double> result(length);
    
#pragma omp parallel for num_threads(nthreads)
    for (const auto &pair : shellpairs_bra) {
        int p1 = pair.first;
        int p2 = pair.second;

        const auto &s1 = bs1[p1];
        const auto &s2 = bs2[p2];
        auto n1 = bs1[p1].size(); // number of basis functions in first shell
        auto n2 = bs2[p2].size(); // number of basis functions in first shell
        auto bf1 = shell2bf_1[p1];  // first basis function in first shell
        auto bf2 = shell2bf_2[p2];  // first basis function in second shell

        for (const auto &pair : shellpairs_ket) {
            int p3 = pair.first;
            int p4 = pair.second;

            const auto &s3 = bs3[p3];
            const auto &s4 = bs4[p4];
            auto n3 = bs3[p3].size(); // number of basis functions in first shell
            auto n4 = bs4[p4].size(); // number of basis functions in first shell
            auto bf3 = shell2bf_3[p3];  // first basis function in first shell
            auto bf4 = shell2bf_4[p4];  // first basis function in second shell

            // Perform schwarz screening
            if (schwarz_bra[p1 * bs2.size() + p2] * schwarz_ket[p3 * bs4.size() + p4] < threshold_sq) continue;

            int thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            engines[thread_id].compute(s1, s2, s3, s4); // Compute shell set
            const auto& buf_vec = engines[thread_id].results(); // will point to computed shell sets

            auto ints_shellset = buf_vec[0];    // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out

            auto full = false;
            if (bs1_equiv_bs2 && p1 != p2 && bs3_equiv_bs4 && p3 != p4) {
                // Loop over shell block, keeping a total count idx for the size of shell set
                for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                    size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                    size_t offset_1_T = (bf1 + f1) * nbf3 * nbf4;
                    for(auto f2 = 0; f2 != n2; ++f2) {
                        size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                        size_t offset_2_T = (bf2 + f2) * nbf1 * nbf3 * nbf4;
                        for(auto f3 = 0; f3 != n3; ++f3) {
                            size_t offset_3 = (bf3 + f3) * nbf4;
                            size_t offset_3_T = bf3 + f3;
                            for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                size_t offset_4 = bf4 + f4;
                                size_t offset_4_T = (bf4 + f4) * nbf3;
                                result[offset_1 + offset_2 + offset_3 + offset_4] = 
                                    result[offset_1_T + offset_2_T + offset_3_T + offset_4_T] = ints_shellset[idx];
                            }
                        }
                    }
                }
                full = true;
            } 
            if (bs1_equiv_bs2 && p1 != p2) {
                // Loop over shell block, keeping a total count idx for the size of shell set
                for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                    size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                    size_t offset_1_T = (bf1 + f1) * nbf3 * nbf4;
                    for(auto f2 = 0; f2 != n2; ++f2) {
                        size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                        size_t offset_2_T = (bf2 + f2) * nbf1 * nbf3 * nbf4;
                        for(auto f3 = 0; f3 != n3; ++f3) {
                            size_t offset_3 = (bf3 + f3) * nbf4;
                            for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                size_t offset_4 = bf4 + f4;
                                result[offset_1 + offset_2 + offset_3 + offset_4] =
                                    result[offset_1_T + offset_2_T + offset_3 + offset_4] = ints_shellset[idx];
                            }
                        }
                    }
                }
                full = true;
            } 
            if (bs3_equiv_bs4 && p3 != p4) {
                // Loop over shell block, keeping a total count idx for the size of shell set
                for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                    size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                    for(auto f2 = 0; f2 != n2; ++f2) {
                        size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                        for(auto f3 = 0; f3 != n3; ++f3) {
                            size_t offset_3 = (bf3 + f3) * nbf4;
                            size_t offset_3_T = bf3 + f3;
                            for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                size_t offset_4 = bf4 + f4;
                                size_t offset_4_T = (bf4 + f4) * nbf3;
                                result[offset_1 + offset_2 + offset_3 + offset_4] =
                                    result[offset_1 + offset_2 + offset_3_T + offset_4_T] = ints_shellset[idx];
                            }
                        }
                    }
                }
                full = true;
            } 
            if (full == false) {
                // Loop over shell block, keeping a total count idx for the size of shell set
                for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                    size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                    for(auto f2 = 0; f2 != n2; ++f2) {
                        size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                        for(auto f3 = 0; f3 != n3; ++f3) {
                            size_t offset_3 = (bf3 + f3) * nbf4;
                            for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                size_t offset_4 = bf4 + f4;
                                result[offset_1 + offset_2 + offset_3 + offset_4] = ints_shellset[idx];
                            }
                        }
                    }
                }
            }
        }
    }
    return py::array(result.size(), result.data());
    // This apparently copies data, but it should be fine right?
    // https://github.com/pybind/pybind11/issues/1042 there's a workaround
}

// Computes nuclear derivatives of one-electron integrals 
py::array compute_1e_deriv(std::string type, std::vector<int> deriv_vec) {
    assert(ncart == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");
    // Get order of differentiation
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    // Create mappings from 1d buffer index (flattened upper triangle shell derivative index)
    // to multidimensional shell derivative index
    // Potential integrals buffer is flattened upper triangle of (6 + NCART) dimensional deriv_order tensor
    int d1_buf_idx = (type == "potential") ? 6 + ncart : 6;
    const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(d1_buf_idx, deriv_order);

    // Shell pairs after screening
    const auto bs1_equiv_bs2 = (bs1 == bs2);
    auto shellpairs = build_shellpairs(bs1, bs2);

    // One-electron integral derivative engine
    std::vector<libint2::Engine> engines(nthreads);

    if (type == "overlap") {
        engines[0] = libint2::Engine(libint2::Operator::overlap, max_nprim, max_l, deriv_order);
    } else if (type == "kinetic") {
        engines[0] = libint2::Engine(libint2::Operator::kinetic, max_nprim, max_l, deriv_order);
    } else if (type == "potential") {
        engines[0] = libint2::Engine(libint2::Operator::nuclear, max_nprim, max_l, deriv_order);
        engines[0].set_params(make_point_charges(atoms));
    } else {
       throw std::invalid_argument("type must be overlap, kinetic, or potential");
    }

    engines[0].set_precision(max_engine_precision);
    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }

    // Get size of derivative array and allocate
    size_t length = nbf1 * nbf2;
    std::vector<double> result(length);

#pragma omp parallel for num_threads(nthreads)
    for (const auto &pair : shellpairs) {
        int p1 = pair.first;
        int p2 = pair.second;

        const auto &s1 = bs1[p1];
        const auto &s2 = bs2[p2];
        auto n1 = bs1[p1].size(); // number of basis functions in first shell
        auto n2 = bs2[p2].size(); // number of basis functions in first shell
        auto bf1 = shell2bf_1[p1];  // first basis function in first shell
        auto bf2 = shell2bf_2[p2];  // first basis function in second shell
        auto atom1 = shell2atom_1[p1]; // Atom index of shell 1
        auto atom2 = shell2atom_2[p2]; // Atom index of shell 2

        // If the atoms are the same we ignore it as the derivatives will be zero.
        if (atom1 == atom2 && type != "potential") continue;
        // Create list of atom indices corresponding to each shell. Libint uses longs, so we will too.
        std::vector<long> shell_atom_index_list{atom1, atom2};

        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        engines[thread_id].compute(s1, s2); // Compute shell set
        const auto& buf_vec = engines[thread_id].results(); // will point to computed shell sets

        // For every desired atom derivative, check shell and nuclear indices for a match,
        // add it to subvector for that derivative
        // Add in the coordinate index 0,1,2 (x,y,z) in desired coordinates and offset the index appropriately.
        std::vector<std::vector<int>> indices(deriv_order, std::vector<int> (0,0));
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

            if (type == "potential") {
                for (int i = 0; i < natom; i++){
                    if (i == desired_atom_idx) {
                        int tmp = 3 * (i + 2) + desired_coordinates[j];
                        indices[j].push_back(tmp);
                    }
                }
            }
        }
        
        // Now indices is a vector of vectors, where each subvector is your choices
        // for the first derivative operator, second, third, etc
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

        // Loop over every buffer index and accumulate for every shell set.
        if (bs1_equiv_bs2 && p1 != p2) {
            for(auto i = 0; i < buffer_indices.size(); ++i) {
                auto ints_shellset = buf_vec[buffer_indices[i]];
                if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                    for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                        result[(bf1 + f1) * nbf2 + bf2 + f2] += ints_shellset[idx];
                        result[(bf2 + f2) * nbf1 + bf1 + f1] += ints_shellset[idx];
                    }
                }
            }
        } else {
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

// Computes nuclear derivatives of dipole integrals
std::vector<py::array> compute_dipole_derivs(std::vector<int> deriv_vec) {
    assert(ncart == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");
    // Get order of differentiation
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    // Create mappings from 1d buffer index (flattened upper triangle shell derivative index)
    // to multidimensional shell derivative index
    const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(6, deriv_order);

    // Shell pairs after screening
    const auto bs1_equiv_bs2 = (bs1 == bs2);
    auto shellpairs = build_shellpairs(bs1, bs2);

    // Integral engine
    std::vector<libint2::Engine> engines(nthreads);

    // COM generator
    std::array<double,3> COM = {0.000, 0.000, 0.000};

    // Will compute overlap + electric dipole moments
    engines[0] = libint2::Engine(libint2::Operator::emultipole1, max_nprim, max_l, deriv_order);
    engines[0].set_params(COM); // with COM as the multipole origin
    engines[0].set_precision(max_engine_precision);
    engines[0].prescale_by(-1);
    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }

    size_t length = nbf1 * nbf2;
    std::vector<double> Mu_X(length); // Mu_X Vector
    std::vector<double> Mu_Y(length); // Mu_Y Vector
    std::vector<double> Mu_Z(length); // Mu_Z Vector

#pragma omp parallel for num_threads(nthreads)
    for (const auto &pair : shellpairs) {
        int p1 = pair.first;
        int p2 = pair.second;

        const auto &s1 = bs1[p1];
        const auto &s2 = bs2[p2];
        auto n1 = bs1[p1].size(); // number of basis functions in first shell
        auto n2 = bs2[p2].size(); // number of basis functions in first shell
        auto bf1 = shell2bf_1[p1];  // first basis function in first shell
        auto bf2 = shell2bf_2[p2];  // first basis function in second shell
        auto atom1 = shell2atom_1[p1]; // Atom index of shell 1
        auto atom2 = shell2atom_2[p2]; // Atom index of shell 2

        // Create list of atom indices corresponding to each shell. Libint uses longs, so we will too.
        std::vector<long> shell_atom_index_list{atom1, atom2};

        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        engines[thread_id].compute(s1, s2); // Compute shell set
        const auto& buf_vec = engines[thread_id].results(); // will point to computed shell sets

        // For every desired atom derivative, check shell and nuclear indices for a match,
        // add it to subvector for that derivative
        // Add in the coordinate index 0,1,2 (x,y,z) in desired coordinates and offset the index appropriately.
        std::vector<std::vector<int>> indices(deriv_order, std::vector<int> (0,0));
        for (int j = 0; j < desired_atom_indices.size(); j++){
            int desired_atom_idx = desired_atom_indices[j];
            // Shell indices
            for (int i = 0; i < 2; i++){
                int atom_idx = shell_atom_index_list[i];
                if (atom_idx == desired_atom_idx) {
                    int tmp = 3 * i + desired_coordinates[j];
                    indices[j].push_back(tmp);
                    continue; // Avoid adding same atom and coord twice
                }
            }
        }

        // Now indices is a vector of vectors, where each subvector is your choices
        // for the first derivative operator, second, third, etc
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
            buffer_indices.push_back(buf_idx * 4);
        }

        // Loop over every buffer index and accumulate for every shell set.
        if (bs1_equiv_bs2 && p1 != p2) {
            for(auto i = 0; i < buffer_indices.size(); ++i) {
                auto mu_x_shellset = buf_vec[buffer_indices[i] + 1];
                auto mu_y_shellset = buf_vec[buffer_indices[i] + 2];
                auto mu_z_shellset = buf_vec[buffer_indices[i] + 3];
                if (mu_x_shellset == nullptr && mu_y_shellset == nullptr && mu_z_shellset == nullptr)
                    continue;  // nullptr returned if the entire shell-set was screened out
                for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                    for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                        Mu_X[(bf1 + f1) * nbf2 + bf2 + f2] += mu_x_shellset[idx];
                        Mu_X[(bf2 + f2) * nbf1 + bf1 + f1] += mu_x_shellset[idx];
                        Mu_Y[(bf1 + f1) * nbf2 + bf2 + f2] += mu_y_shellset[idx];
                        Mu_Y[(bf2 + f2) * nbf1 + bf1 + f1] += mu_y_shellset[idx];
                        Mu_Z[(bf1 + f1) * nbf2 + bf2 + f2] += mu_z_shellset[idx];
                        Mu_Z[(bf2 + f2) * nbf1 + bf1 + f1] += mu_z_shellset[idx];
                    }
                }
            }
        } else {
            for(auto i = 0; i < buffer_indices.size(); ++i) {
                auto mu_x_shellset = buf_vec[buffer_indices[i] + 1];
                auto mu_y_shellset = buf_vec[buffer_indices[i] + 2];
                auto mu_z_shellset = buf_vec[buffer_indices[i] + 3];
                if (mu_x_shellset == nullptr && mu_y_shellset == nullptr && mu_z_shellset == nullptr)
                    continue;  // nullptr returned if the entire shell-set was screened out
                for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                    for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                        Mu_X[(bf1 + f1) * nbf2 + bf2 + f2] += mu_x_shellset[idx];
                        Mu_Y[(bf1 + f1) * nbf2 + bf2 + f2] += mu_y_shellset[idx];
                        Mu_Z[(bf1 + f1) * nbf2 + bf2 + f2] += mu_z_shellset[idx];
                    }
                }
            }
        }
    }
    return {py::array(Mu_X.size(), Mu_X.data()), py::array(Mu_Y.size(), Mu_Y.data()),
            py::array(Mu_Z.size(), Mu_Z.data())};
}

// Computes nuclear derivatives of two-electron integrals
py::array compute_2e_deriv(std::string type, double beta, std::vector<int> deriv_vec) {
    assert(ncart == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");
    // Get order of differentiation
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    // Create mapping from 1d buffer index (flattened upper triangle shell derivative index)
    // to multidimensional shell derivative index
    const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);

    // Shell screening
    std::vector<std::pair<int, int>> shellpairs_bra, shellpairs_ket;
    std::vector<double> schwarz_bra, schwarz_ket;
    const auto bs1_equiv_bs2 = (bs1 == bs2);
    const auto bs3_equiv_bs4 = (bs3 == bs4);
    std::tie(shellpairs_bra, schwarz_bra) = schwarz_screening(bs1, bs2);
    std::tie(shellpairs_ket, schwarz_ket) = schwarz_screening(bs3, bs4);
    auto threshold_sq = threshold * threshold;

    // ERI derivative integral engine
    std::vector<libint2::Engine> engines(nthreads);

    if (type == "eri") {
        engines[0] = libint2::Engine(libint2::Operator::coulomb, max_nprim, max_l, deriv_order);
    } else if (type == "f12") {
        auto cgtg_params = make_cgtg(beta);
        engines[0] = libint2::Engine(libint2::Operator::cgtg, max_nprim, max_l, deriv_order);
        engines[0].set_params(cgtg_params);
    } else if (type == "f12g12") {
        auto cgtg_params = make_cgtg(beta);
        engines[0] = libint2::Engine(libint2::Operator::cgtg_x_coulomb, max_nprim, max_l, deriv_order);
        engines[0].set_params(cgtg_params);
    } else if (type == "f12_squared") {
        auto cgtg_params = take_square(make_cgtg(beta));
        engines[0] = libint2::Engine(libint2::Operator::cgtg, max_nprim, max_l, deriv_order);
        engines[0].set_params(cgtg_params);
    } else if (type == "f12_double_commutator") {
        auto cgtg_params = make_cgtg(beta);
        engines[0] = libint2::Engine(libint2::Operator::delcgtg2, max_nprim, max_l, deriv_order,
                                            std::numeric_limits<libint2::scalar_type>::epsilon(),
                                            cgtg_params, libint2::BraKet::xx_xx);
    } else {
        throw std::invalid_argument("type must be eri, f12, f12g12, f12_squared, or f12_double_commutator");
    }

    engines[0].set_precision(max_engine_precision);
    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4;
    std::vector<double> result(length);

#pragma omp parallel for num_threads(nthreads)
    for (const auto &pair : shellpairs_bra) {
        int p1 = pair.first;
        int p2 = pair.second;

        const auto &s1 = bs1[p1];
        const auto &s2 = bs2[p2];
        auto n1 = bs1[p1].size(); // number of basis functions in first shell
        auto n2 = bs2[p2].size(); // number of basis functions in second shell
        auto bf1 = shell2bf_1[p1];  // first basis function in first shell
        auto bf2 = shell2bf_2[p2];  // first basis function in second shell
        auto atom1 = shell2atom_1[p1]; // Atom index of shell 1
        auto atom2 = shell2atom_2[p2]; // Atom index of shell 2

        for (const auto &pair : shellpairs_ket) {
            int p3 = pair.first;
            int p4 = pair.second;

            const auto &s3 = bs3[p3];
            const auto &s4 = bs4[p4];
            auto n3 = bs3[p3].size(); // number of basis functions in third shell
            auto n4 = bs4[p4].size(); // number of basis functions in fourth shell
            auto bf3 = shell2bf_3[p3];  // first basis function in third shell
            auto bf4 = shell2bf_4[p4];  // first basis function in fourth shell
            auto atom3 = shell2atom_3[p3]; // Atom index of shell 3
            auto atom4 = shell2atom_4[p4]; // Atom index of shell 4

            // Perform schwarz screening
            if (schwarz_bra[p1 * bs2.size() + p2] * schwarz_ket[p3 * bs4.size() + p4] < threshold_sq) continue;

            // If the atoms are the same we ignore it as the derivatives will be zero.
            if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
            // Ensure all desired_atoms correspond to at least one shell atom to
            // ensure desired derivative exists. else, skip this shell quartet.
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
            
            // For every desired atom derivative, check shell indices for a match,
            // add it to subvector for that derivative
            // Add in the coordinate index 0,1,2 (x,y,z) in desired coordinates and offset the index appropriately.
            std::vector<std::vector<int>> indices(deriv_order, std::vector<int> (0,0));
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
            
            // Now indices is a vector of vectors, where each subvector is your choices
            // for the first derivative operator, second, third, etc
            // and the total number of subvectors is the order of differentiation
            // Now we want all combinations where we pick exactly one index from each subvector.
            // This is achievable through a cartesian product 
            std::vector<std::vector<int>> index_combos = cartesian_product(indices);
            std::vector<int> buffer_indices;

            // Binary search to find 1d buffer index from multidimensional shell derivative index in `index_combos`
            for (auto vec : index_combos)  {
                std::sort(vec.begin(), vec.end());
                int buf_idx = 0;
                // buffer_multidim_lookup
                auto it = lower_bound(buffer_multidim_lookup.begin(), buffer_multidim_lookup.end(), vec);
                if (it != buffer_multidim_lookup.end()) buf_idx = it - buffer_multidim_lookup.begin();
                buffer_indices.push_back(buf_idx);
            }

            // If we made it this far, the shell derivative we want is contained in the buffer. 
            int thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            engines[thread_id].compute(s1, s2, s3, s4); // Compute shell set
            const auto& buf_vec = engines[thread_id].results(); // will point to computed shell sets
            
            auto full = false;
            if (bs1_equiv_bs2 && p1 != p2 && bs3_equiv_bs4 && p3 != p4) {
                for(auto i = 0; i < buffer_indices.size(); ++i) {
                    auto ints_shellset = buf_vec[buffer_indices[i]];
                    if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                        size_t offset_1_T = (bf1 + f1) * nbf3 * nbf4;
                        for(auto f2 = 0; f2 != n2; ++f2) {
                            size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                            size_t offset_2_T = (bf2 + f2) * nbf1 * nbf3 * nbf4;
                            for(auto f3 = 0; f3 != n3; ++f3) {
                                size_t offset_3 = (bf3 + f3) * nbf4;
                                size_t offset_3_T = bf3 + f3;
                                for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                    size_t offset_4 = bf4 + f4;
                                    size_t offset_4_T = (bf4 + f4) * nbf3;
                                    result[offset_1 + offset_2 + offset_3 + offset_4] = 
                                        result[offset_1_T + offset_2_T + offset_3_T + offset_4_T] += ints_shellset[idx];
                                }
                            }
                        }
                    }
                }
                full = true;
            }
            if (bs1_equiv_bs2 && p1 != p2) {
                for(auto i = 0; i < buffer_indices.size(); ++i) {
                    auto ints_shellset = buf_vec[buffer_indices[i]];
                    if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                        size_t offset_1_T = (bf1 + f1) * nbf3 * nbf4;
                        for(auto f2 = 0; f2 != n2; ++f2) {
                            size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                            size_t offset_2_T = (bf2 + f2) * nbf1 * nbf3 * nbf4;
                            for(auto f3 = 0; f3 != n3; ++f3) {
                                size_t offset_3 = (bf3 + f3) * nbf4;
                                for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                    size_t offset_4 = bf4 + f4;
                                    result[offset_1 + offset_2 + offset_3 + offset_4] =
                                        result[offset_1_T + offset_2_T + offset_3 + offset_4] += ints_shellset[idx];
                                }
                            }
                        }
                    }
                }
                full = true;
            }
            if (bs3_equiv_bs4 && p3 != p4) {
                for(auto i = 0; i < buffer_indices.size(); ++i) {
                    auto ints_shellset = buf_vec[buffer_indices[i]];
                    if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                    // Loop over shell block, keeping a total count idx for the size of shell set
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                        for(auto f2 = 0; f2 != n2; ++f2) {
                            size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                            for(auto f3 = 0; f3 != n3; ++f3) {
                                size_t offset_3 = (bf3 + f3) * nbf4;
                                size_t offset_3_T = bf3 + f3;
                                for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                    size_t offset_4 = bf4 + f4;
                                    size_t offset_4_T = (bf4 + f4) * nbf3;
                                    result[offset_1 + offset_2 + offset_3 + offset_4] =
                                        result[offset_1 + offset_2 + offset_3_T + offset_4_T] += ints_shellset[idx];
                                }
                            }
                        }
                    }
                }
                full = true;
            }
            if (full == false) {
                for(auto i = 0; i < buffer_indices.size(); ++i) {
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
    // This is not the bottleneck
    return py::array(result.size(), result.data()); 
    // This apparently copies data, but it should be fine right?
    // https://github.com/pybind/pybind11/issues/1042 there's a workaround
}

// Write OEI derivatives up to `max_deriv_order` to disk
// HDF5 File Name: oei_derivs.h5 
//      HDF5 Dataset names within the file:
//      oei_nbf1_nbf2_deriv1 
//          shape (nbf,nbf,n_unique_1st_derivs)
//      oei_nbf1_nbf2_deriv2 
//          shape (nbf,nbf,n_unique_2nd_derivs)
//      oei_nbf1_nbf2_deriv3 
//          shape (nbf,nbf,n_unique_3rd_derivs)
//      ...
// The number of unique derivatives is essentially equal to the size of the
// generalized upper triangle of the derivative tensor.
void compute_1e_deriv_disk(std::string type, int max_deriv_order) {
    std::cout << "Writing one-electron " << type << " integral derivative tensors up to order " 
                                         << max_deriv_order << " to disk...";
    long total_deriv_slices = 0;
    for (int i = 1; i <= max_deriv_order; i++){
        total_deriv_slices += how_many_derivs(natom, i);
    }

    double check = (nbf1 * nbf2 * total_deriv_slices * 8) * (1e-9);
    assert(check < 10 && "Total disk space required for ERI's exceeds 10 GB. Increase threshold and recompile to proceed.");

    // Shell pairs after screening
    const auto bs1_equiv_bs2 = (bs1 == bs2);
    auto shellpairs = build_shellpairs(bs1, bs2);

    // Create H5 File and prepare to fill with 0.0's
    const H5std_string file_name("oei_derivs.h5");
    H5File* file = new H5File(file_name,H5F_ACC_TRUNC);
    double fillvalue = 0.0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);

    for (int deriv_order = 1; deriv_order <= max_deriv_order; deriv_order++){
        // how many unique cartesian nuclear derivatives (e.g., so we only save one of d^2/dx1dx2 and d^2/dx2dx1, etc)
        unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

        // Create mappings from 1d buffer index (flattened upper triangle shell derivative index)
        // to multidimensional shell derivative index
        // Potential integrals buffer is flattened upper triangle of (6 + NCART) dimensional deriv_order tensor
        int d1_buf_idx = (type == "potential") ? 6 + ncart : 6;
        const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(d1_buf_idx, deriv_order);

        // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index)
        // to multidimensional index
        const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

        // Define engines and buffers
        std::vector<libint2::Engine> engines(nthreads), t_engines(nthreads), v_engines(nthreads);
        
        if (type == "overlap") {
            engines[0] = libint2::Engine(libint2::Operator::overlap, max_nprim, max_l, deriv_order);
        } else if (type == "kinetic") {
            engines[0] = libint2::Engine(libint2::Operator::kinetic, max_nprim, max_l, deriv_order);
        } else if (type == "potential") {
            engines[0] = libint2::Engine(libint2::Operator::nuclear, max_nprim, max_l, deriv_order);
            engines[0].set_params(make_point_charges(atoms));
        } else {
           throw std::invalid_argument("type must be overlap, kinetic, or potential");
        }

        engines[0].set_precision(max_engine_precision);
        for (size_t i = 1; i != nthreads; ++i) {
            engines[i] = engines[0];
        }

        // Define HDF5 dataset names
        const H5std_string dset_name(type + "_" + std::to_string(nbf1) + "_" + std::to_string(nbf2) 
                                             + "_deriv" + std::to_string(deriv_order));

        // Define rank and dimensions of data that will be written to the file
        hsize_t file_dims[] = {nbf1, nbf2, nderivs_triu};
        DataSpace fspace(3, file_dims);
        // Create dataset for each integral type and write 0.0's into the file 
        DataSet* dataset = new DataSet(file->createDataSet(dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        hsize_t stride[3] = {1, 1, 1}; // stride and block can be used to 
        hsize_t block[3] = {1, 1, 1};  // add values to multiple places, useful if symmetry ever used.
        hsize_t zerostart[3] = {0, 0, 0};

        /* Initialize lock */
        omp_init_lock(&lock);

#pragma omp parallel for num_threads(nthreads)
        for (const auto &pair : shellpairs) {
            int p1 = pair.first;
            int p2 = pair.second;

            const auto &s1 = bs1[p1];
            const auto &s2 = bs2[p2];
            auto n1 = bs1[p1].size(); // number of basis functions in first shell
            auto n2 = bs2[p2].size(); // number of basis functions in first shell
            auto bf1 = shell2bf_1[p1];  // first basis function in first shell
            auto bf2 = shell2bf_2[p2];  // first basis function in second shell
            auto atom1 = shell2atom_1[p1]; // Atom index of shell 1
            auto atom2 = shell2atom_2[p2]; // Atom index of shell 2
            std::vector<long> shell_atom_index_list{atom1, atom2};

            int thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            engines[thread_id].compute(s1, s2); // Compute shell set
            const auto& buffer = engines[thread_id].results(); // will point to computed shell sets

            // Define shell set slabs
            double shellset_slab_12 [n1][n2][nderivs_triu] = {};
            double shellset_slab_21 [n2][n1][nderivs_triu] = {};

            // Loop over every possible unique nuclear cartesian derivative index (flattened upper triangle)
            // For 1st derivatives of 2 atom system, this is 6. 2nd derivatives of 2 atom system: 21, etc
            for(int nuc_idx = 0; nuc_idx < nderivs_triu; ++nuc_idx) {
                // Look up multidimensional cartesian derivative index
                auto multi_cart_idx = cart_multidim_lookup[nuc_idx];
                // Create a vector of vectors called `indices`, where each subvector
                // is your possible choices for the first derivative operator, second, third, etc
                // and the total number of subvectors is order of differentiation
                // What follows fills these indices
                std::vector<std::vector<int>> indices(deriv_order, std::vector<int> (0,0));

                // Loop over each cartesian coordinate index which we are differentiating wrt
                // for this nuclear cartesian derivative index and check to see if it is present
                // in the shell duet, and where it is present in the potential operator
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
                    // Now for potentials only, loop over each atom in molecule, and if this derivative
                    // differentiates wrt that atom, we also need to collect that index.
                    if (type == "potential") {
                        for (int i = 0; i < natom; i++){
                            if (i == desired_atom_idx) {
                                int tmp = 3 * (i + 2) + desired_coord;
                                indices[j].push_back(tmp);
                            }
                        }
                    }
                }

                // Now indices is a vector of vectors, where each subvector is your choices
                // for the first derivative operator, second, third, etc
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
                if (bs1_equiv_bs2 && p1 != p2){
                    // Loop over shell block for each buffer index which contributes to this derivative
                    for(auto i = 0; i < buffer_indices.size(); ++i) {
                        auto shellset = buffer[buffer_indices[i]];
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                                shellset_slab_12[f1][f2][nuc_idx] =
                                    shellset_slab_21[f2][f1][nuc_idx] += shellset[idx];
                            }
                        }
                    }
                } else {
                    // Loop over shell block for each buffer index which contributes to this derivative
                    for(auto i = 0; i < buffer_indices.size(); ++i) {
                        auto shellset = buffer[buffer_indices[i]];
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                                shellset_slab_12[f1][f2][nuc_idx] += shellset[idx];
                            }
                        }
                    }
                }
            } // Unique nuclear cartesian derivative indices loop

            /* Serialize HDF dataset writing using OpenMP lock */
            omp_set_lock(&lock);

            // Now write this shell set slab to HDF5 file
            // Create file space hyperslab, defining where to write data to in file
            hsize_t count[3] = {n1, n2, nderivs_triu};
            hsize_t start[3] = {bf1, bf2, 0};
            fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
            // Create dataspace defining for memory dataset to write to file
            hsize_t mem_dims[] = {n1, n2, nderivs_triu};
            DataSpace mspace(3, mem_dims);
            mspace.selectHyperslab(H5S_SELECT_SET, count, zerostart, stride, block);
            // Write buffer data 'shellset_slab' with data type double from
            // memory dataspace `mspace` to file dataspace `fspace`
            dataset->write(shellset_slab_12, PredType::NATIVE_DOUBLE, mspace, fspace);

            if (bs1_equiv_bs2 && p1 != p2) {
                // Now write this shell set slab to HDF5 file
                // Create file space hyperslab, defining where to write data to in file
                hsize_t count_T[3] = {n2, n1, nderivs_triu};
                hsize_t start_T[3] = {bf2, bf1, 0};
                fspace.selectHyperslab(H5S_SELECT_SET, count_T, start_T, stride, block);
                // Create dataspace defining for memory dataset to write to file
                hsize_t mem_dims_T[] = {n2, n1, nderivs_triu};
                DataSpace mspace_T(3, mem_dims_T);
                mspace_T.selectHyperslab(H5S_SELECT_SET, count_T, zerostart, stride, block);
                // Write buffer data 'shellset_slab' with data type double from memory dataspace `mspace` to file dataspace `fspace`
                dataset->write(shellset_slab_21, PredType::NATIVE_DOUBLE, mspace_T, fspace);
            }

            /* Release lock */
            omp_unset_lock(&lock);
            
        } // shell duet loops
        // Delete datasets for this derivative order
        delete dataset;
    } // deriv order loop

    /* Finished lock mechanism, destroy it */
    omp_destroy_lock(&lock);
    // close the file
    delete file;
    std::cout << " done" << std::endl;
} // compute_1e_deriv_disk 

// Write dipole derivatives up to `max_deriv_order` to disk
// HDF5 File Name: dipole_derivs.h5 
//      HDF5 Dataset names within the file:
//      dipole_nbf1_nbf2_deriv1 
//          shape (nbf,nbf,n_unique_1st_derivs)
//      dipole_nbf1_nbf2_deriv2 
//          shape (nbf,nbf,n_unique_2nd_derivs)
//      dipole_nbf1_nbf2_deriv3 
//          shape (nbf,nbf,n_unique_3rd_derivs)
//      ...
// The number of unique derivatives is essentially equal to the size of the
// generalized upper triangle of the derivative tensor.
void compute_dipole_deriv_disk(int max_deriv_order) {
    std::cout << "Writing dipole integral derivative tensors up to order " << max_deriv_order << " to disk...";
    long total_deriv_slices = 0;
    for (int i = 1; i <= max_deriv_order; i++){
        total_deriv_slices += how_many_derivs(natom, i);
    }

    // Shell pairs after screening
    auto shellpairs = build_shellpairs(bs1, bs2);

    // Create H5 File and prepare to fill with 0.0's
    const H5std_string file_name("dipole_derivs.h5");
    H5File* file = new H5File(file_name,H5F_ACC_TRUNC);
    double fillvalue = 0.0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);

    for (int deriv_order = 1; deriv_order <= max_deriv_order; deriv_order++){
        // how many unique cartesian nuclear derivatives (e.g., so we only save one of d^2/dx1dx2 and d^2/dx2dx1, etc)
        unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

        // Create mappings from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
        // Overlap and kinetic have different mappings than potential since potential has more elements in the buffer 
        const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(6, deriv_order);
        
        // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
        const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

        // Define engines and buffers
        std::vector<libint2::Engine> engines(nthreads);

        // COM generator
        std::array<double,3> COM = {0.000, 0.000, 0.000};

        // Will compute overlap + electric dipole moments
        engines[0] = libint2::Engine(libint2::Operator::emultipole1, max_nprim, max_l, deriv_order);
        engines[0].set_params(COM); // with COM as the multipole origin
        engines[0].set_precision(max_engine_precision);
        engines[0].prescale_by(-1);
        for (size_t i = 1; i != nthreads; ++i) {
            engines[i] = engines[0];
        }

        // Define HDF5 dataset names
        const H5std_string Mu_X_dset_name("mu_x_" + std::to_string(nbf1) + "_" + std::to_string(nbf2) 
                                                  + "_deriv" + std::to_string(deriv_order));
        const H5std_string Mu_Y_dset_name("mu_y_" + std::to_string(nbf1) + "_" + std::to_string(nbf2) 
                                                  + "_deriv" + std::to_string(deriv_order));
        const H5std_string Mu_Z_dset_name("mu_z_" + std::to_string(nbf1) + "_" + std::to_string(nbf2) 
                                                  + "_deriv" + std::to_string(deriv_order));

        // Define rank and dimensions of data that will be written to the file
        hsize_t file_dims[] = {nbf1, nbf2, nderivs_triu};
        DataSpace fspace(3, file_dims);
        // Create dataset for each integral type and write 0.0's into the file 
        DataSet* Mu_X_dataset = new DataSet(file->createDataSet(Mu_X_dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        DataSet* Mu_Y_dataset = new DataSet(file->createDataSet(Mu_Y_dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        DataSet* Mu_Z_dataset = new DataSet(file->createDataSet(Mu_Z_dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        hsize_t stride[3] = {1, 1, 1}; // stride and block can be used to 
        hsize_t block[3] = {1, 1, 1};  // add values to multiple places, useful if symmetry ever used.
        hsize_t zerostart[3] = {0, 0, 0};

        /* Initialize lock */
        omp_init_lock(&lock);

#pragma omp parallel for num_threads(nthreads)
        for (const auto &pair : shellpairs) {
            int p1 = pair.first;
            int p2 = pair.second;

            const auto &s1 = bs1[p1];
            const auto &s2 = bs2[p2];
            auto n1 = bs1[p1].size(); // number of basis functions in first shell
            auto n2 = bs2[p2].size(); // number of basis functions in first shell
            auto bf1 = shell2bf_1[p1];  // first basis function in first shell
            auto bf2 = shell2bf_2[p2];  // first basis function in second shell
            auto atom1 = shell2atom_1[p1]; // Atom index of shell 1
            auto atom2 = shell2atom_2[p2]; // Atom index of shell 2
            std::vector<long> shell_atom_index_list{atom1, atom2};

            int thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            engines[thread_id].compute(s1, s2); // Compute shell set
            const auto& buf_vec = engines[thread_id].results(); // will point to computed shell sets

            // Define shell set slabs
            double Mu_X_shellset_slab_12 [n1][n2][nderivs_triu] = {};
            double Mu_Y_shellset_slab_12 [n1][n2][nderivs_triu] = {};
            double Mu_Z_shellset_slab_12 [n1][n2][nderivs_triu] = {};
            double Mu_X_shellset_slab_21 [n2][n1][nderivs_triu] = {};
            double Mu_Y_shellset_slab_21 [n2][n1][nderivs_triu] = {};
            double Mu_Z_shellset_slab_21 [n2][n1][nderivs_triu] = {};

            // Loop over every possible unique nuclear cartesian derivative index (flattened upper triangle)
            // For 1st derivatives of 2 atom system, this is 6. 2nd derivatives of 2 atom system: 21, etc
            for(int nuc_idx = 0; nuc_idx < nderivs_triu; ++nuc_idx) {
                // Look up multidimensional cartesian derivative index
                auto multi_cart_idx = cart_multidim_lookup[nuc_idx];
                // For overlap/kinetic and potential sepearately, create a vector of vectors called `indices`, where each subvector
                // is your possible choices for the first derivative operator, second, third, etc and the total number of subvectors is order of differentiation
                // What follows fills these indices
                std::vector<std::vector<int>> indices(deriv_order, std::vector<int> (0,0));

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
                        }
                    }
                }

                // Now indices is a vector of vectors, where each subvector is your choices for the first derivative operator, second, third, etc
                // and the total number of subvectors is the order of differentiation
                // Now we want all combinations where we pick exactly one index from each subvector.
                // This is achievable through a cartesian product
                std::vector<std::vector<int>> index_combos = cartesian_product(indices);
                std::vector<int> buffer_indices;
                // Overlap/Kinetic integrals: collect needed buffer indices which we need to sum for this nuclear cartesian derivative
                for (auto vec : index_combos)  {
                    std::sort(vec.begin(), vec.end());
                    int buf_idx = 0;
                    auto it = lower_bound(buffer_multidim_lookup.begin(), buffer_multidim_lookup.end(), vec);
                    if (it != buffer_multidim_lookup.end()) buf_idx = it - buffer_multidim_lookup.begin();
                    buffer_indices.push_back(buf_idx * 4);
                }

                // Loop over shell block for each buffer index which contributes to this derivative
                if (p1 != p2) {
                    for(auto i = 0; i < buffer_indices.size(); ++i) {
                        auto mu_x_shellset = buf_vec[buffer_indices[i] + 1];
                        auto mu_y_shellset = buf_vec[buffer_indices[i] + 2];
                        auto mu_z_shellset = buf_vec[buffer_indices[i] + 3];
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                                Mu_X_shellset_slab_12[f1][f2][nuc_idx] =
                                    Mu_X_shellset_slab_21[f2][f1][nuc_idx] += mu_x_shellset[idx];
                                Mu_Y_shellset_slab_12[f1][f2][nuc_idx] =
                                    Mu_Y_shellset_slab_21[f2][f1][nuc_idx] += mu_y_shellset[idx];
                                Mu_Z_shellset_slab_12[f1][f2][nuc_idx] =
                                    Mu_Z_shellset_slab_21[f2][f1][nuc_idx] += mu_z_shellset[idx];
                            }
                        }
                    }
                } else { 
                    for(auto i = 0; i < buffer_indices.size(); ++i) {
                        auto mu_x_shellset = buf_vec[buffer_indices[i] + 1];
                        auto mu_y_shellset = buf_vec[buffer_indices[i] + 2];
                        auto mu_z_shellset = buf_vec[buffer_indices[i] + 3];
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                                Mu_X_shellset_slab_12[f1][f2][nuc_idx] += mu_x_shellset[idx];
                                Mu_Y_shellset_slab_12[f1][f2][nuc_idx] += mu_y_shellset[idx];
                                Mu_Z_shellset_slab_12[f1][f2][nuc_idx] += mu_z_shellset[idx];
                            }
                        }
                    }
                }
            } // Unique nuclear cartesian derivative indices loop

            /* Serialize HDF dataset writing using OpenMP lock */
            omp_set_lock(&lock);

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
            Mu_X_dataset->write(Mu_X_shellset_slab_12, PredType::NATIVE_DOUBLE, mspace, fspace);
            Mu_Y_dataset->write(Mu_Y_shellset_slab_12, PredType::NATIVE_DOUBLE, mspace, fspace);
            Mu_Z_dataset->write(Mu_Z_shellset_slab_12, PredType::NATIVE_DOUBLE, mspace, fspace);

            if (p1 != p2) {
                // Now write this shell set slab to HDF5 file
                // Create file space hyperslab, defining where to write data to in file
                hsize_t count_T[3] = {n2, n1, nderivs_triu};
                hsize_t start_T[3] = {bf2, bf1, 0};
                fspace.selectHyperslab(H5S_SELECT_SET, count_T, start_T, stride, block);
                // Create dataspace defining for memory dataset to write to file
                hsize_t mem_dims_T[] = {n2, n1, nderivs_triu};
                DataSpace mspace_T(3, mem_dims_T);
                mspace_T.selectHyperslab(H5S_SELECT_SET, count_T, zerostart, stride, block);
                // Write buffer data 'shellset_slab' with data type double from memory dataspace `mspace` to file dataspace `fspace`
                Mu_X_dataset->write(Mu_X_shellset_slab_21, PredType::NATIVE_DOUBLE, mspace_T, fspace);
                Mu_Y_dataset->write(Mu_Y_shellset_slab_21, PredType::NATIVE_DOUBLE, mspace_T, fspace);
                Mu_Z_dataset->write(Mu_Z_shellset_slab_21, PredType::NATIVE_DOUBLE, mspace_T, fspace);
            }

            /* Release lock */
            omp_unset_lock(&lock);

        } // shell duet loops
        // Delete datasets for this derivative order
        delete Mu_X_dataset;
        delete Mu_Y_dataset;
        delete Mu_Z_dataset;
    } // deriv order loop

    /* Finished lock mechanism, destroy it */
    omp_destroy_lock(&lock);
    // close the file
    delete file;
    std::cout << " done" << std::endl;
} // compute_dipole_deriv_disk 


// Writes TEI derivatives up to `max_deriv_order` to disk.
// HDF5 File Name: tei_derivs.h5 
//      HDF5 Dataset names within the file:
//      tei_nbf1_nbf2_nbf3_nbf4_deriv1 
//          shape (nbf,nbf,nbf,nbf,n_unique_1st_derivs)
//      tei_nbf1_nbf2_nbf3_nbf4_deriv2
//          shape (nbf,nbf,nbf,nbf,n_unique_2nd_derivs)
//      tei_nbf1_nbf2_nbf3_nbf4_deriv3
//          shape (nbf,nbf,nbf,nbf,n_unique_3rd_derivs)
//      ...
void compute_2e_deriv_disk(std::string type, double beta, int max_deriv_order) { 
    std::cout << "Writing two-electron " << type << " integral derivative tensors up to order " 
                                         << max_deriv_order << " to disk...";
    // Check to make sure you are not flooding the disk.
    long total_deriv_slices = 0;
    for (int i = 1; i <= max_deriv_order; i++){
        total_deriv_slices += how_many_derivs(natom, i);
    }
    double check = (nbf1 * nbf2 * nbf3 * nbf4 * total_deriv_slices * 8) * (1e-9);
    assert(check < 50 && "Total disk space required for ERI's exceeds 50 GB. Increase threshold and recompile to proceed.");

    // Shell screening
    std::vector<std::pair<int, int>> shellpairs_bra, shellpairs_ket;
    std::vector<double> schwarz_bra, schwarz_ket;
    const auto bs1_equiv_bs2 = (bs1 == bs2);
    const auto bs3_equiv_bs4 = (bs3 == bs4);
    std::tie(shellpairs_bra, schwarz_bra) = schwarz_screening(bs1, bs2);
    std::tie(shellpairs_ket, schwarz_ket) = schwarz_screening(bs3, bs4);
    auto threshold_sq = threshold * threshold;
    
    // Create H5 File and prepare to fill with 0.0's                                         
    const H5std_string file_name(type + "_derivs.h5");
    H5File* file = new H5File(file_name,H5F_ACC_TRUNC);
    double fillvalue = 0.0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);

    for (int deriv_order = 1; deriv_order <= max_deriv_order; deriv_order++){
        // Number of unique nuclear derivatives of ERI's
        unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

        // Create mapping from 1d buffer index (flattened upper triangle shell derivative index)
        // to multidimensional shell derivative index
        const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);

        // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index)
        // to multidimensional index
        const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

        // Libint engine for computing shell quartet derivatives
        std::vector<libint2::Engine> engines(nthreads);

        if (type == "eri") {
            engines[0] = libint2::Engine(libint2::Operator::coulomb, max_nprim, max_l, deriv_order);
        } else if (type == "f12") {
            auto cgtg_params = make_cgtg(beta);
            engines[0] = libint2::Engine(libint2::Operator::cgtg, max_nprim, max_l, deriv_order);
            engines[0].set_params(cgtg_params);
        } else if (type == "f12g12") {
            auto cgtg_params = make_cgtg(beta);
            engines[0] = libint2::Engine(libint2::Operator::cgtg_x_coulomb, max_nprim, max_l, deriv_order);
            engines[0].set_params(cgtg_params);
        } else if (type == "f12_squared") {
            auto cgtg_params = take_square(make_cgtg(beta));
            engines[0] = libint2::Engine(libint2::Operator::cgtg, max_nprim, max_l, deriv_order);
            engines[0].set_params(cgtg_params);
        } else if (type == "f12_double_commutator") {
            auto cgtg_params = make_cgtg(beta);
            engines[0] = libint2::Engine(libint2::Operator::delcgtg2, max_nprim, max_l, deriv_order,
                                                std::numeric_limits<libint2::scalar_type>::epsilon(),
                                                cgtg_params, libint2::BraKet::xx_xx);
        } else {
            throw std::invalid_argument("type must be eri, f12, f12g12, f12_squared, or f12_double_commutator");
        }

        engines[0].set_precision(max_engine_precision);
        for (size_t i = 1; i != nthreads; ++i) {
            engines[i] = engines[0];
        }

        // Define HDF5 dataset name
        const H5std_string dset_name(type + "_" + std::to_string(nbf1) + "_" + std::to_string(nbf2)
                                         + "_" + std::to_string(nbf3) + "_" + std::to_string(nbf4)
                                         + "_deriv" + std::to_string(deriv_order));
        hsize_t file_dims[] = {nbf1, nbf2, nbf3, nbf4, nderivs_triu};
        DataSpace fspace(5, file_dims);
        // Create dataset for each integral type and write 0.0's into the file 
        DataSet* dataset = new DataSet(file->createDataSet(dset_name, PredType::NATIVE_DOUBLE, fspace, plist));
        hsize_t stride[5] = {1, 1, 1, 1, 1}; // stride and block can be used to 
        hsize_t block[5] = {1, 1, 1, 1, 1};  // add values to multiple places, useful if symmetry ever used.
        hsize_t zerostart[5] = {0, 0, 0, 0, 0};

        /* Initialize lock */
        omp_init_lock(&lock);

#pragma omp parallel for num_threads(nthreads)
        for (const auto &pair : shellpairs_bra) {
            int p1 = pair.first;
            int p2 = pair.second;

            const auto &s1 = bs1[p1];
            const auto &s2 = bs2[p2];
            auto n1 = bs1[p1].size(); // number of basis functions in first shell
            auto n2 = bs2[p2].size(); // number of basis functions in second shell
            auto bf1 = shell2bf_1[p1];  // first basis function in first shell
            auto bf2 = shell2bf_2[p2];  // first basis function in second shell
            auto atom1 = shell2atom_1[p1]; // Atom index of shell 1
            auto atom2 = shell2atom_2[p2]; // Atom index of shell 2

            for (const auto &pair : shellpairs_ket) {
                int p3 = pair.first;
                int p4 = pair.second;

                const auto &s3 = bs3[p3];
                const auto &s4 = bs4[p4];
                auto n3 = bs3[p3].size(); // number of basis functions in third shell
                auto n4 = bs4[p4].size(); // number of basis functions in fourth shell
                auto bf3 = shell2bf_3[p3];  // first basis function in third shell
                auto bf4 = shell2bf_4[p4];  // first basis function in fourth shell
                auto atom3 = shell2atom_3[p3]; // Atom index of shell 3
                auto atom4 = shell2atom_4[p4]; // Atom index of shell 4

                // Perform schwarz screening
                if (schwarz_bra[p1 * bs2.size() + p2] * schwarz_ket[p3 * bs4.size() + p4] < threshold_sq) continue;

                // If the atoms are the same we ignore it as the derivatives will be zero.
                if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
                std::vector<long> shell_atom_index_list{atom1, atom2, atom3, atom4};

                int thread_id = 0;
#ifdef _OPENMP
                thread_id = omp_get_thread_num();
#endif
                engines[thread_id].compute(s1, s2, s3, s4); // Compute shell set
                const auto& buf_vec = engines[thread_id].results(); // will point to computed shell sets

                // Define shell set slab, with extra dimension for unique derivatives, initialized with 0.0's
                double ints_shellset_slab_1234 [n1][n2][n3][n4][nderivs_triu] = {};
                double ints_shellset_slab_2143 [n2][n1][n4][n3][nderivs_triu] = {};
                double ints_shellset_slab_2134 [n2][n1][n3][n4][nderivs_triu] = {};
                double ints_shellset_slab_1243 [n1][n2][n4][n3][nderivs_triu] = {};

                // Loop over every possible unique nuclear cartesian derivative index (flattened upper triangle)
                for(int nuc_idx = 0; nuc_idx < nderivs_triu; ++nuc_idx) {
                    // Look up multidimensional cartesian derivative index
                    auto multi_cart_idx = cart_multidim_lookup[nuc_idx];

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

                    // Now indices is a vector of vectors, where each subvector is your choices
                    // for the first derivative operator, second, third, etc
                    // and the total number of subvectors is the order of differentiation
                    // Now we want all combinations where we pick exactly one index from each subvector.
                    // This is achievable through a cartesian product 
                    std::vector<std::vector<int>> index_combos = cartesian_product(indices);
                    std::vector<int> buffer_indices;

                    // Binary search to find 1d buffer index from multidimensional shell derivative index in `index_combos`
                    for (auto vec : index_combos)  {
                        std::sort(vec.begin(), vec.end());
                        int buf_idx = 0;
                        // buffer_multidim_lookup
                        auto it = lower_bound(buffer_multidim_lookup.begin(), buffer_multidim_lookup.end(), vec);
                        if (it != buffer_multidim_lookup.end()) buf_idx = it - buffer_multidim_lookup.begin();
                        buffer_indices.push_back(buf_idx);
                    }

                    auto full = false;
                    // Loop over shell block, keeping a total count idx for the size of shell set
                    if (bs1_equiv_bs2 && p1 != p2 && bs3_equiv_bs4 && p3 != p4) {
                        for(auto i = 0; i < buffer_indices.size(); ++i) {
                            auto ints_shellset = buf_vec[buffer_indices[i]];
                            if (ints_shellset == nullptr) continue;
                            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                                for(auto f2 = 0; f2 != n2; ++f2) {
                                    for(auto f3 = 0; f3 != n3; ++f3) {
                                        for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                            ints_shellset_slab_1234[f1][f2][f3][f4][nuc_idx] =
                                                ints_shellset_slab_2143[f2][f1][f4][f3][nuc_idx] += ints_shellset[idx];
                                        }
                                    }
                                }
                            }
                        }
                        full = true;
                    }
                    if (bs1_equiv_bs2 && p1 != p2) {
                        for(auto i = 0; i < buffer_indices.size(); ++i) {
                            auto ints_shellset = buf_vec[buffer_indices[i]];
                            if (ints_shellset == nullptr) continue;
                            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                                for(auto f2 = 0; f2 != n2; ++f2) {
                                    for(auto f3 = 0; f3 != n3; ++f3) {
                                        for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                            ints_shellset_slab_1234[f1][f2][f3][f4][nuc_idx] =
                                                ints_shellset_slab_2134[f2][f1][f3][f4][nuc_idx] += ints_shellset[idx];
                                        }
                                    }
                                }
                            }
                        }
                        full = true;
                    }
                    if (bs3_equiv_bs4 && p3 != p4) {
                        for(auto i = 0; i < buffer_indices.size(); ++i) {
                            auto ints_shellset = buf_vec[buffer_indices[i]];
                            if (ints_shellset == nullptr) continue;
                            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                                for(auto f2 = 0; f2 != n2; ++f2) {
                                    for(auto f3 = 0; f3 != n3; ++f3) {
                                        for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                            ints_shellset_slab_1234[f1][f2][f3][f4][nuc_idx] =
                                                ints_shellset_slab_1243[f1][f2][f4][f3][nuc_idx] += ints_shellset[idx];
                                        }
                                    }
                                }
                            }
                        }
                        full = true;
                    }
                    if (full == false) {
                        for(auto i = 0; i < buffer_indices.size(); ++i) {
                            auto ints_shellset = buf_vec[buffer_indices[i]];
                            if (ints_shellset == nullptr) continue;
                            for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                                for(auto f2 = 0; f2 != n2; ++f2) {
                                    for(auto f3 = 0; f3 != n3; ++f3) {
                                        for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                            ints_shellset_slab_1234[f1][f2][f3][f4][nuc_idx] += ints_shellset[idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                } // For every nuc_idx 0, nderivs_triu

                /* Serialize HDF dataset writing using OpenMP lock */
                omp_set_lock(&lock);

                // Now write this shell set slab to HDF5 file
                hsize_t count[5] = {n1, n2, n3, n4, nderivs_triu};
                hsize_t start[5] = {bf1, bf2, bf3, bf4, 0};
                fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
                // Create dataspace defining for memory dataset to write to file
                hsize_t mem_dims[] = {n1, n2, n3, n4, nderivs_triu};
                DataSpace mspace(5, mem_dims);
                mspace.selectHyperslab(H5S_SELECT_SET, count, zerostart, stride, block);
                // Write buffer data 'shellset_slab' with data type double from
                // memory dataspace `mspace` to file dataspace `fspace`
                dataset->write(ints_shellset_slab_1234, PredType::NATIVE_DOUBLE, mspace, fspace);

                if (bs1_equiv_bs2 && p1 != p2 && bs3_equiv_bs4 && p3 != p4) {
                    // Now write this shell set slab to HDF5 file
                    hsize_t count_T[5] = {n2, n1, n4, n3, nderivs_triu};
                    hsize_t start_T[5] = {bf2, bf1, bf4, bf3, 0};
                    fspace.selectHyperslab(H5S_SELECT_SET, count_T, start_T, stride, block);
                    // Create dataspace defining for memory dataset to write to file
                    hsize_t mem_dims_T[] = {n2, n1, n4, n3, nderivs_triu};
                    DataSpace mspace_T(5, mem_dims_T);
                    mspace_T.selectHyperslab(H5S_SELECT_SET, count_T, zerostart, stride, block);
                    // Write buffer data 'shellset_slab' with data type double from
                    // memory dataspace `mspace` to file dataspace `fspace`
                    dataset->write(ints_shellset_slab_2143, PredType::NATIVE_DOUBLE, mspace_T, fspace);
                }

                if (bs1_equiv_bs2 && p1 != p2) {
                    // Now write this shell set slab to HDF5 file
                    hsize_t count_T[5] = {n2, n1, n3, n4, nderivs_triu};
                    hsize_t start_T[5] = {bf2, bf1, bf3, bf4, 0};
                    fspace.selectHyperslab(H5S_SELECT_SET, count_T, start_T, stride, block);
                    // Create dataspace defining for memory dataset to write to file
                    hsize_t mem_dims_T[] = {n2, n1, n3, n4, nderivs_triu};
                    DataSpace mspace_T(5, mem_dims_T);
                    mspace_T.selectHyperslab(H5S_SELECT_SET, count_T, zerostart, stride, block);
                    // Write buffer data 'shellset_slab' with data type double from
                    // memory dataspace `mspace` to file dataspace `fspace`
                    dataset->write(ints_shellset_slab_2134, PredType::NATIVE_DOUBLE, mspace_T, fspace);
                }

                if (bs3_equiv_bs4 && p3 != p4) {
                    // Now write this shell set slab to HDF5 file
                    hsize_t count_T[5] = {n1, n2, n4, n3, nderivs_triu};
                    hsize_t start_T[5] = {bf1, bf2, bf4, bf3, 0};
                    fspace.selectHyperslab(H5S_SELECT_SET, count_T, start_T, stride, block);
                    // Create dataspace defining for memory dataset to write to file
                    hsize_t mem_dims_T[] = {n1, n2, n4, n3, nderivs_triu};
                    DataSpace mspace_T(5, mem_dims_T);
                    mspace_T.selectHyperslab(H5S_SELECT_SET, count_T, zerostart, stride, block);
                    // Write buffer data 'shellset_slab' with data type double from
                    // memory dataspace `mspace` to file dataspace `fspace`
                    dataset->write(ints_shellset_slab_1243, PredType::NATIVE_DOUBLE, mspace_T, fspace);
                }

                /* Release lock */
                omp_unset_lock(&lock);
            }
        } // shell quartet loops
        // Close the dataset for this derivative order
        delete dataset;
    } // deriv order loop

    /* Finished lock mechanism, destroy it */
    omp_destroy_lock(&lock);
    // Close the file
    delete file;
    std::cout << " done" << std::endl;
} // compute_2e_deriv_disk function

// The following function writes all overlap, kinetic, and potential derivatives up to `max_deriv_order` to disk
// HDF5 File Name: oei_derivs.h5 
//      HDF5 Dataset names within the file:
//      overlap_nbf1_nbf2_deriv1 
//          shape (nbf,nbf,n_unique_1st_derivs)
//      overlap_nbf1_nbf2_deriv2 
//          shape (nbf,nbf,n_unique_2nd_derivs)
//      overlap_nbf1_nbf2_deriv3 
//          shape (nbf,nbf,n_unique_3rd_derivs)
//      ...
//      kinetic_nbf1_nbf2_deriv1 
//          shape (nbf,nbf,n_unique_1st_derivs)
//      kinetic_nbf1_nbf2_deriv2 
//          shape (nbf,nbf,n_unique_2nd_derivs)
//      kinetic_nbf1_nbf2_deriv3 
//          shape (nbf,nbf,n_unique_3rd_derivs)
//      ...
//      potential_nbf1_nbf2_deriv1 
//          shape (nbf,nbf,n_unique_1st_derivs)
//      potential_nbf1_nbf2_deriv2 
//          shape (nbf,nbf,n_unique_2nd_derivs)
//      potential_nbf1_nbf2_deriv3 
//          shape (nbf,nbf,n_unique_3rd_derivs)
// The number of unique derivatives is essentially equal to the size of the generalized upper triangle of the derivative tensor.
void oei_deriv_disk(int max_deriv_order) {
    std::cout << "Writing one-electron integral derivative tensors up to order " << max_deriv_order << " to disk...";
    long total_deriv_slices = 0;
    for (int i = 1; i <= max_deriv_order; i++){
        total_deriv_slices += how_many_derivs(natom, i);
    }

    // Shell pairs after screening
    auto shellpairs = build_shellpairs(bs1, bs2);

    // Create H5 File and prepare to fill with 0.0's
    const H5std_string file_name("oei_derivs.h5");
    H5File* file = new H5File(file_name,H5F_ACC_TRUNC);
    double fillvalue = 0.0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);

    for (int deriv_order = 1; deriv_order <= max_deriv_order; deriv_order++){
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

        s_engines[0].set_precision(max_engine_precision);
        t_engines[0].set_precision(max_engine_precision);
        v_engines[0].set_precision(max_engine_precision);
        for (size_t i = 1; i != nthreads; ++i) {
            s_engines[i] = s_engines[0];
            t_engines[i] = t_engines[0];
            v_engines[i] = v_engines[0];
        }

        // Define HDF5 dataset names
        const H5std_string overlap_dset_name("overlap_" + std::to_string(nbf1) + "_" + std::to_string(nbf2) 
                                              + "_deriv" + std::to_string(deriv_order));
        const H5std_string kinetic_dset_name("kinetic_" + std::to_string(nbf1) + "_" + std::to_string(nbf2) 
                                              + "_deriv" + std::to_string(deriv_order));
        const H5std_string potential_dset_name("potential_" + std::to_string(nbf1) + "_" + std::to_string(nbf2) 
                                                + "_deriv" + std::to_string(deriv_order));

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

        /* Initialize lock */
        omp_init_lock(&lock);

#pragma omp parallel for num_threads(nthreads)
        for (const auto &pair : shellpairs) {
            int p1 = pair.first;
            int p2 = pair.second;

            const auto &s1 = bs1[p1];
            const auto &s2 = bs2[p2];
            auto n1 = bs1[p1].size(); // number of basis functions in first shell
            auto n2 = bs2[p2].size(); // number of basis functions in first shell
            auto bf1 = shell2bf_1[p1];  // first basis function in first shell
            auto bf2 = shell2bf_2[p2];  // first basis function in second shell
            auto atom1 = shell2atom_1[p1]; // Atom index of shell 1
            auto atom2 = shell2atom_2[p2]; // Atom index of shell 2
            std::vector<long> shell_atom_index_list{atom1, atom2};

            int thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            s_engines[thread_id].compute(s1, s2); // Compute shell set
            t_engines[thread_id].compute(s1, s2); // Compute shell set
            v_engines[thread_id].compute(s1, s2); // Compute shell set
            const auto& overlap_buffer = s_engines[thread_id].results(); // will point to computed shell sets
            const auto& kinetic_buffer = t_engines[thread_id].results(); // will point to computed shell sets
            const auto& potential_buffer = v_engines[thread_id].results(); // will point to computed shell sets

            // Define shell set slabs
            double overlap_shellset_slab_12 [n1][n2][nderivs_triu] = {};
            double kinetic_shellset_slab_12 [n1][n2][nderivs_triu] = {};
            double potential_shellset_slab_12 [n1][n2][nderivs_triu] = {};
            double overlap_shellset_slab_21 [n2][n1][nderivs_triu] = {};
            double kinetic_shellset_slab_21 [n2][n1][nderivs_triu] = {};
            double potential_shellset_slab_21 [n2][n1][nderivs_triu] = {};

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
                if (p1 != p2) {
                    // Overlap and Kinetic
                    for(auto i = 0; i < buffer_indices.size(); ++i) {
                        auto overlap_shellset = overlap_buffer[buffer_indices[i]];
                        auto kinetic_shellset = kinetic_buffer[buffer_indices[i]];
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                                overlap_shellset_slab_12[f1][f2][nuc_idx] =
                                    overlap_shellset_slab_21[f2][f1][nuc_idx] += overlap_shellset[idx];
                                kinetic_shellset_slab_12[f1][f2][nuc_idx] =
                                    kinetic_shellset_slab_21[f2][f1][nuc_idx] += kinetic_shellset[idx];
                            }
                        }
                    }
                    // Potential
                    for(auto i = 0; i < potential_buffer_indices.size(); ++i) {
                        auto potential_shellset = potential_buffer[potential_buffer_indices[i]];
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                                potential_shellset_slab_12[f1][f2][nuc_idx] =
                                    potential_shellset_slab_21[f2][f1][nuc_idx] += potential_shellset[idx];
                            }
                        }
                    }
                } else { 
                    // Overlap and Kinetic
                    for(auto i = 0; i < buffer_indices.size(); ++i) {
                        auto overlap_shellset = overlap_buffer[buffer_indices[i]];
                        auto kinetic_shellset = kinetic_buffer[buffer_indices[i]];
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                                overlap_shellset_slab_12[f1][f2][nuc_idx] += overlap_shellset[idx];
                                kinetic_shellset_slab_12[f1][f2][nuc_idx] += kinetic_shellset[idx];
                            }
                        }
                    }
                    // Potential
                    for(auto i = 0; i < potential_buffer_indices.size(); ++i) {
                        auto potential_shellset = potential_buffer[potential_buffer_indices[i]];
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                                potential_shellset_slab_12[f1][f2][nuc_idx] += potential_shellset[idx];
                            }
                        }
                    }
                }
            } // Unique nuclear cartesian derivative indices loop

            /* Serialize HDF dataset writing using OpenMP lock */
            omp_set_lock(&lock);

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
            overlap_dataset->write(overlap_shellset_slab_12, PredType::NATIVE_DOUBLE, mspace, fspace);
            kinetic_dataset->write(kinetic_shellset_slab_12, PredType::NATIVE_DOUBLE, mspace, fspace);
            potential_dataset->write(potential_shellset_slab_12, PredType::NATIVE_DOUBLE, mspace, fspace);

            if (p1 != p2) {
                // Now write this shell set slab to HDF5 file
                // Create file space hyperslab, defining where to write data to in file
                hsize_t count_T[3] = {n2, n1, nderivs_triu};
                hsize_t start_T[3] = {bf2, bf1, 0};
                fspace.selectHyperslab(H5S_SELECT_SET, count_T, start_T, stride, block);
                // Create dataspace defining for memory dataset to write to file
                hsize_t mem_dims_T[] = {n2, n1, nderivs_triu};
                DataSpace mspace_T(3, mem_dims_T);
                mspace_T.selectHyperslab(H5S_SELECT_SET, count_T, zerostart, stride, block);
                // Write buffer data 'shellset_slab' with data type double from memory dataspace `mspace` to file dataspace `fspace`
                overlap_dataset->write(overlap_shellset_slab_21, PredType::NATIVE_DOUBLE, mspace_T, fspace);
                kinetic_dataset->write(kinetic_shellset_slab_21, PredType::NATIVE_DOUBLE, mspace_T, fspace);
                potential_dataset->write(potential_shellset_slab_21, PredType::NATIVE_DOUBLE, mspace_T, fspace);
            }

            /* Release lock */
            omp_unset_lock(&lock);

        } // shell duet loops
        // Delete datasets for this derivative order
        delete overlap_dataset;
        delete kinetic_dataset;
        delete potential_dataset;
    } // deriv order loop

    /* Finished lock mechanism, destroy it */
    omp_destroy_lock(&lock);
    // close the file
    delete file;
    std::cout << " done" << std::endl;
} //oei_deriv_disk 

// Computes a single 'deriv_order' derivative tensor of OEIs, keeps everything in core memory
std::vector<py::array> oei_deriv_core(int deriv_order) {
    // Shell pairs after screening
    auto shellpairs = build_shellpairs(bs1, bs2);

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

    s_engines[0].set_precision(max_engine_precision);
    t_engines[0].set_precision(max_engine_precision);
    v_engines[0].set_precision(max_engine_precision);
    for (size_t i = 1; i != nthreads; ++i) {
        s_engines[i] = s_engines[0];
        t_engines[i] = t_engines[0];
        v_engines[i] = v_engines[0];
    }

    size_t length = nbf1 * nbf2 * nderivs_triu;
    std::vector<double> S(length);
    std::vector<double> T(length);
    std::vector<double> V(length);

#pragma omp parallel for num_threads(nthreads)
    for (const auto &pair : shellpairs) {
        int p1 = pair.first;
        int p2 = pair.second;

        const auto &s1 = bs1[p1];
        const auto &s2 = bs2[p2];
        auto n1 = bs1[p1].size(); // number of basis functions in first shell
        auto n2 = bs2[p2].size(); // number of basis functions in first shell
        auto bf1 = shell2bf_1[p1];  // first basis function in first shell
        auto bf2 = shell2bf_2[p2];  // first basis function in second shell
        auto atom1 = shell2atom_1[p1]; // Atom index of shell 1
        auto atom2 = shell2atom_2[p2]; // Atom index of shell 2
        std::vector<long> shell_atom_index_list{atom1, atom2};

        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        s_engines[thread_id].compute(s1, s2); // Compute shell set
        t_engines[thread_id].compute(s1, s2); // Compute shell set
        v_engines[thread_id].compute(s1, s2); // Compute shell set
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
            if (p1 != p2) {
                for(auto i = 0; i < buffer_indices.size(); ++i) {
                    auto overlap_shellset = overlap_buffer[buffer_indices[i]];
                    auto kinetic_shellset = kinetic_buffer[buffer_indices[i]];
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                            S[(bf1 + f1) * nbf2 + bf2 + f2 + offset_nuc_idx] =
                                S[(bf2 + f2) * nbf1 + bf1 + f1 + offset_nuc_idx] += overlap_shellset[idx];
                            T[(bf1 + f1) * nbf2 + bf2 + f2 + offset_nuc_idx] =
                                T[(bf2 + f2) * nbf1 + bf1 + f1 + offset_nuc_idx] += kinetic_shellset[idx];
                        }
                    }
                }
            } else {
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
            }
            // Potential
            if (p1 != p2) {
                for(auto i = 0; i < potential_buffer_indices.size(); ++i) {
                    auto potential_shellset = potential_buffer[potential_buffer_indices[i]];
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                            V[(bf1 + f1) * nbf2 + bf2 + f2 + offset_nuc_idx] =
                                V[(bf2 + f2) * nbf1 + bf1 + f1 + offset_nuc_idx] += potential_shellset[idx];
                        }
                    }
                }
            } else {
                for(auto i = 0; i < potential_buffer_indices.size(); ++i) {
                    auto potential_shellset = potential_buffer[potential_buffer_indices[i]];
                    for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                        for(auto f2 = 0; f2 != n2; ++f2, ++idx) {
                            V[(bf1 + f1) * nbf2 + bf2 + f2 + offset_nuc_idx] += potential_shellset[idx];
                        }
                    }
                }
            }
        } // Unique nuclear cartesian derivative indices loop
    } // shell duet loops
    return {py::array(S.size(), S.data()), py::array(T.size(), T.data()), py::array(V.size(), V.data())}; // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042 there's a workaround
} // oei_deriv_core function

// Computes a single 'deriv_order' derivative tensor of electron repulsion integrals, keeps everything in core memory
py::array eri_deriv_core(int deriv_order) {
    // Number of unique nuclear derivatives of ERI's
    unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

    // Create mapping from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional shell derivative index
    const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);

    // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
    const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(ncart, deriv_order);

    // Shell screening assumes bs1 == bs2 == bs3 == bs4 for Hartree-Fock
    std::vector<std::pair<int, int>> shellpairs;
    std::vector<double> schwarz;
    std::tie(shellpairs, schwarz) = schwarz_screening(bs1, bs2);
    auto threshold_sq = threshold * threshold;

    // Libint engine for computing shell quartet derivatives
    std::vector<libint2::Engine> engines(nthreads);
    engines[0] = libint2::Engine(libint2::Operator::coulomb, max_nprim, max_l, deriv_order);

    engines[0].set_precision(max_engine_precision);
    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }

    size_t length = nbf1 * nbf2 * nbf3 * nbf4 * nderivs_triu;
    std::vector<double> result(length);

#pragma omp parallel for num_threads(nthreads)
    for (const auto &pair : shellpairs) {
        int p1 = pair.first;
        int p2 = pair.second;

        const auto &s1 = bs1[p1];
        const auto &s2 = bs2[p2];
        auto n1 = bs1[p1].size(); // number of basis functions in first shell
        auto n2 = bs2[p2].size(); // number of basis functions in second shell
        auto bf1 = shell2bf_1[p1];  // first basis function in first shell
        auto bf2 = shell2bf_2[p2];  // first basis function in second shell
        auto atom1 = shell2atom_1[p1]; // Atom index of shell 1
        auto atom2 = shell2atom_2[p2]; // Atom index of shell 2

        for (const auto &pair : shellpairs) {
            int p3 = pair.first;
            int p4 = pair.second;

            const auto &s3 = bs3[p3];
            const auto &s4 = bs4[p4];
            auto n3 = bs3[p3].size(); // number of basis functions in third shell
            auto n4 = bs4[p4].size(); // number of basis functions in fourth shell
            auto bf3 = shell2bf_3[p3];  // first basis function in third shell
            auto bf4 = shell2bf_4[p4];  // first basis function in fourth shell
            auto atom3 = shell2atom_3[p3]; // Atom index of shell 3
            auto atom4 = shell2atom_4[p4]; // Atom index of shell 4

            // Perform schwarz screening
            if (schwarz[p1 * bs2.size() + p2] * schwarz[p3 * bs4.size() + p4] < threshold_sq) continue;

            // If the atoms are the same we ignore it as the derivatives will be zero.
            if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
            std::vector<long> shell_atom_index_list{atom1, atom2, atom3, atom4};

            int thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            engines[thread_id].compute(s1, s2, s3, s4); // Compute shell set
            const auto& buf_vec = engines[thread_id].results(); // will point to computed shell sets

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
                for (auto vec : index_combos)  {
                    std::sort(vec.begin(), vec.end());
                    int buf_idx = 0;
                    // buffer_multidim_lookup
                    auto it = lower_bound(buffer_multidim_lookup.begin(), buffer_multidim_lookup.end(), vec);
                    if (it != buffer_multidim_lookup.end()) buf_idx = it - buffer_multidim_lookup.begin();
                    buffer_indices.push_back(buf_idx);
                }

                auto full = false;
                if (p1 != p2 && p3 != p4) {
                    for(auto i = 0; i < buffer_indices.size(); ++i) {
                        auto ints_shellset = buf_vec[buffer_indices[i]];
                        if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                            size_t offset_1_T = (bf1 + f1) * nbf3 * nbf4;
                            for(auto f2 = 0; f2 != n2; ++f2) {
                                size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                                size_t offset_2_T = (bf2 + f2) * nbf1 * nbf3 * nbf4;
                                for(auto f3 = 0; f3 != n3; ++f3) {
                                    size_t offset_3 = (bf3 + f3) * nbf4;
                                    size_t offset_3_T = bf3 + f3;
                                    for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                        size_t offset_4 = bf4 + f4;
                                        size_t offset_4_T = (bf4 + f4) * nbf3;
                                        result[offset_1 + offset_2 + offset_3 + offset_4 + offset_nuc_idx] = 
                                            result[offset_1_T + offset_2_T + offset_3_T + offset_4_T  + offset_nuc_idx] += ints_shellset[idx];
                                    }
                                }
                            }
                        }
                    }
                    full = true;
                }
                if (p1 != p2) {
                    for(auto i = 0; i < buffer_indices.size(); ++i) {
                        auto ints_shellset = buf_vec[buffer_indices[i]];
                        if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                            size_t offset_1_T = (bf1 + f1) * nbf3 * nbf4;
                            for(auto f2 = 0; f2 != n2; ++f2) {
                                size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                                size_t offset_2_T = (bf2 + f2) * nbf1 * nbf3 * nbf4;
                                for(auto f3 = 0; f3 != n3; ++f3) {
                                    size_t offset_3 = (bf3 + f3) * nbf4;
                                    for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                        size_t offset_4 = bf4 + f4;
                                        result[offset_1 + offset_2 + offset_3 + offset_4  + offset_nuc_idx] =
                                            result[offset_1_T + offset_2_T + offset_3 + offset_4  + offset_nuc_idx] += ints_shellset[idx];
                                    }
                                }
                            }
                        }
                    }
                    full = true;
                }
                if (p3 != p4) {
                    for(auto i = 0; i < buffer_indices.size(); ++i) {
                        auto ints_shellset = buf_vec[buffer_indices[i]];
                        if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                        // Loop over shell block, keeping a total count idx for the size of shell set
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                            for(auto f2 = 0; f2 != n2; ++f2) {
                                size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                                for(auto f3 = 0; f3 != n3; ++f3) {
                                    size_t offset_3 = (bf3 + f3) * nbf4;
                                    size_t offset_3_T = bf3 + f3;
                                    for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                        size_t offset_4 = bf4 + f4;
                                        size_t offset_4_T = (bf4 + f4) * nbf3;
                                        result[offset_1 + offset_2 + offset_3 + offset_4  + offset_nuc_idx] =
                                            result[offset_1 + offset_2 + offset_3_T + offset_4_T  + offset_nuc_idx] += ints_shellset[idx];
                                    }
                                }
                            }
                        }
                    }
                    full = true;
                }
                if (full == false) {
                    for(auto i = 0; i < buffer_indices.size(); ++i) {
                        auto ints_shellset = buf_vec[buffer_indices[i]];
                        if (ints_shellset == nullptr) continue;  // nullptr returned if shell-set screened out
                        for(auto f1 = 0, idx = 0; f1 != n1; ++f1) {
                            size_t offset_1 = (bf1 + f1) * nbf2 * nbf3 * nbf4;
                            for(auto f2 = 0; f2 != n2; ++f2) {
                                size_t offset_2 = (bf2 + f2) * nbf3 * nbf4;
                                for(auto f3 = 0; f3 != n3; ++f3) {
                                    size_t offset_3 = (bf3 + f3) * nbf4;
                                    for(auto f4 = 0; f4 != n4; ++f4, ++idx) {
                                        result[offset_1 + offset_2 + offset_3 + bf4 + f4  + offset_nuc_idx] += ints_shellset[idx];
                                    }
                                }
                            }
                        }
                    }
                }
            } // For every nuc_idx 0, nderivs_triu
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
    m.def("generate_multi_index_lookup", &generate_multi_index_lookup, "Flattened upper triangular map");
    m.def("finalize", &finalize, "Kills libint");
    m.def("compute_1e_int", &compute_1e_int, "Computes one-electron integrals with libint");
    m.def("compute_dipole_ints", &compute_dipole_ints, "Computes electric (Cartesian) dipole integrals with libint");
    m.def("compute_2e_int", &compute_2e_int, "Computes two-electron integrals with libint");
    m.def("compute_1e_deriv", &compute_1e_deriv, "Computes one-electron integral nuclear derivatives with libint");
    m.def("compute_dipole_derivs", &compute_dipole_derivs, "Computes electric (Cartesian) dipole nuclear integrals with libint");
    m.def("compute_2e_deriv", &compute_2e_deriv, "Computes two-electron integral nuclear derivatives with libint");
    m.def("compute_1e_deriv_disk", &compute_1e_deriv_disk, "Computes one-electron nuclear derivative tensors from 1st order up to nth order and writes them to disk with HDF5");
    m.def("compute_dipole_deriv_disk", &compute_dipole_deriv_disk, "Computes dipole nuclear derivative tensors from 1st order up to nth order and writes them to disk with HDF5");
    m.def("compute_2e_deriv_disk", &compute_2e_deriv_disk, "Computes coulomb integral nuclear derivative tensors from 1st order up to nth order and writes them to disk with HDF5");
    m.def("oei_deriv_disk", &oei_deriv_disk, "Computes overlap, kinetic, and potential integral derivative tensors from 1st order up to nth order and writes them to disk with HDF5");
    m.def("oei_deriv_core", &oei_deriv_core, "Computes a single OEI integral derivative tensor, in memory.");
    m.def("eri_deriv_core", &eri_deriv_core, "Computes a single coulomb integral nuclear derivative tensor, in memory.");
    //TODO partial derivative impl's
    //m.def("compute_2e_deriv_core", &compute_2e_partial_deriv_core, "Computes a single contracted Gaussian-type geminal integral nuclear derivative tensor, in memory.");
    //m.def("compute_2e_partial_deriv_disk", &compute_2e_partial_deriv_disk, "Computes a subset of the full coulomb integral nuclear derivative tensor and writes them to disk with HDF5");
     m.attr("LIBINT2_MAX_DERIV_ORDER") = LIBINT2_MAX_DERIV_ORDER;
}
