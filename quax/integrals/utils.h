// Utility functions for libint_interface

// Creates atom objects from xyz file path
std::vector<libint2::Atom> get_atoms(std::string xyzfilename) 
{
    std::ifstream input_file(xyzfilename);
    std::vector<libint2::Atom> atoms = libint2::read_dotxyz(input_file);
    return atoms;
}

// Creates a combined basis set
libint2::BasisSet make_ao_cabs(std::vector<libint2::Atom> atoms, 
                               std::string obs_name, libint2::BasisSet cabs)
{
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