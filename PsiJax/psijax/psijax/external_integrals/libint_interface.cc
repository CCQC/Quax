#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdlib.h>
#include <libint2.hpp>

namespace py = pybind11;

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

// Creates atom objects from xyz file path
std::vector<libint2::Atom> get_atoms(std::string xyzfilename) 
{
    std::ifstream input_file(xyzfilename);
    std::vector<libint2::Atom> atoms = libint2::read_dotxyz(input_file);
    return atoms;
}

// Compute overlap integrals
py::array overlap(std::string xyzfilename, std::string basis_name) {
    libint2::initialize();
    // Load geometry and basis set
    std::vector<libint2::Atom> atoms = get_atoms(xyzfilename);
    libint2::BasisSet obs(basis_name, atoms);
    obs.set_pure(false); // use cartesian gaussians

    // Overlap integral engine
    libint2::Engine s_engine(libint2::Operator::overlap,obs.max_nprim(),obs.max_l());
    const auto& buf_vec = s_engine.results(); // will point to computed shell sets

    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
    int nbf = obs.nbf();
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
    libint2::finalize();
    return py::array(result.size(), result.data()); 
}

// Compute kinetic energy integrals
py::array kinetic(std::string xyzfilename, std::string basis_name) {
    libint2::initialize();
    // Load geometry and basis set
    std::vector<libint2::Atom> atoms = get_atoms(xyzfilename);
    libint2::BasisSet obs(basis_name, atoms);
    obs.set_pure(false); // use cartesian gaussians

    // Kinetic energy integral engine
    libint2::Engine t_engine(libint2::Operator::kinetic,obs.max_nprim(),obs.max_l());
    const auto& buf_vec = t_engine.results(); // will point to computed shell sets

    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
    int nbf = obs.nbf();
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
                    // idx = x + (y * width) where x = bf2 + f2 and y = bf1 + f1 
                    result[ (bf1 + f1) * nbf + bf2 + f2 ] = ints_shellset[idx];
                }
            }
        }
    }
    libint2::finalize();
    return py::array(result.size(), result.data());
}

// Compute nuclear-electron potential energy integrals
py::array potential(std::string xyzfilename, std::string basis_name) {
    libint2::initialize();
    // Load geometry and basis set
    std::vector<libint2::Atom> atoms = get_atoms(xyzfilename);
    libint2::BasisSet obs(basis_name, atoms);
    obs.set_pure(false); // use cartesian gaussians

    // Potential integral engine
    libint2::Engine v_engine(libint2::Operator::nuclear,obs.max_nprim(),obs.max_l());
    v_engine.set_params(make_point_charges(atoms));
    const auto& buf_vec = v_engine.results(); // will point to computed shell sets

    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
    int nbf = obs.nbf();
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
    libint2::finalize();
    return py::array(result.size(), result.data());
}

// Computes electron repulsion integrals
py::array eri(std::string xyzfilename, std::string basis_name) {
    // workaround for data copying: perhaps pass an empty numpy array, then populate it in C++? avoids last line, which copies
    libint2::initialize();

    // Load basis set and geometry. TODO this assumes units are angstroms... 
    std::vector<libint2::Atom> atoms = get_atoms(xyzfilename);
    libint2::BasisSet obs(basis_name, atoms);
    obs.set_pure(false); // use cartesian gaussians
    int nbf = obs.nbf();
    libint2::Engine eri_engine(libint2::Operator::coulomb,obs.max_nprim(),obs.max_l());

    size_t length = nbf * nbf * nbf * nbf;
    std::vector<double> result(length);

    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
    const auto& buf_vec = eri_engine.results(); // will point to computed shell sets
    
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
    libint2::finalize();
    return py::array(result.size(), result.data()); // This apparently copies data, but it should be fine right? https://github.com/pybind/pybind11/issues/1042 there's a workaround
}

// Computes nuclear derivatives of overlap integrals
py::array overlap_deriv(std::string xyzfilename, std::string basis_name, std::vector<int> deriv_vec) {
    libint2::initialize();
    // Get order of differentiation
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Default: first derivatives, but if deriv_order=2 then use 2nd derivative buffer index lookup
    static const std::vector<int> buffer_index_lookup1 = {0,1,2,3,4,5};
    static const std::vector<std::vector<int>> buffer_index_lookup2 = {
         {0, 1,  2,  3,  4,  5}, 
         {1, 6,  7,  8,  9, 10},
         {2, 7, 11, 12, 13, 14},
         {3, 8, 12, 15, 16, 17},
         {4, 9, 13, 16, 18, 19},
         {5,10, 14, 17, 19, 20},};
    // TODO 
    // if deriv_order == 3
    // if deriv_order == 4

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    // Load basis set and geometry.
    std::vector<libint2::Atom> atoms = get_atoms(xyzfilename);
    libint2::BasisSet obs(basis_name, atoms);

    assert(3 * atoms.size() == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");

    obs.set_pure(false); // use cartesian gaussians
    int nbf = obs.nbf();

    // Overlap derivative integral engine
    libint2::Engine s_engine(libint2::Operator::overlap,obs.max_nprim(),obs.max_l(),deriv_order);

    // Get size of overlap derivative array and allocate 
    size_t length = nbf * nbf;
    std::vector<double> result(length);

    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
    const auto shell2atom = obs.shell2atom(atoms); // maps shell index to atom index
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
    libint2::finalize();
    return py::array(result.size(), result.data()); 
}

// Computes nuclear derivatives of kinetic energy integrals
py::array kinetic_deriv(std::string xyzfilename, std::string basis_name, std::vector<int> deriv_vec) {
    libint2::initialize();
    // Get order of differentiation
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Default: first derivatives, but if deriv_order=2 then use 2nd derivative buffer index lookup
    static const std::vector<int> buffer_index_lookup1 = {0,1,2,3,4,5};
    static const std::vector<std::vector<int>> buffer_index_lookup2 = {
         {0, 1,  2,  3,  4,  5}, 
         {1, 6,  7,  8,  9, 10},
         {2, 7, 11, 12, 13, 14},
         {3, 8, 12, 15, 16, 17},
         {4, 9, 13, 16, 18, 19},
         {5,10, 14, 17, 19, 20},};

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    // Load basis set and geometry.
    std::vector<libint2::Atom> atoms = get_atoms(xyzfilename);
    libint2::BasisSet obs(basis_name, atoms);
    obs.set_pure(false); // use cartesian gaussians
    assert(3 * atoms.size() == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");

    // Overlap derivative integral engine
    libint2::Engine t_engine(libint2::Operator::kinetic,obs.max_nprim(),obs.max_l(),deriv_order);
    const auto& buf_vec = t_engine.results(); // will point to computed shell sets

    // Get size of kinetic derivative array and allocate 
    int nbf = obs.nbf();
    size_t length = nbf * nbf;
    std::vector<double> result(length);
    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
    const auto shell2atom = obs.shell2atom(atoms); // maps shell index to atom index

    
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
    libint2::finalize();
    return py::array(result.size(), result.data()); 
}

// Computes nuclear derivatives of potential energy integrals 
py::array potential_deriv(std::string xyzfilename, std::string basis_name, std::vector<int> deriv_vec) {
    libint2::initialize();
    // Get order of differentiation
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Default: first derivatives, but if deriv_order=2 then use 2nd derivative buffer index lookup
    static const std::vector<int> buffer_index_lookup1 = {0,1,2,3,4,5};
    static const std::vector<std::vector<int>> buffer_index_lookup2 = {
         {0, 1,  2,  3,  4,  5}, 
         {1, 6,  7,  8,  9, 10},
         {2, 7, 11, 12, 13, 14},
         {3, 8, 12, 15, 16, 17},
         {4, 9, 13, 16, 18, 19},
         {5,10, 14, 17, 19, 20},};

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    // Load basis set and geometry.
    std::vector<libint2::Atom> atoms = get_atoms(xyzfilename);
    libint2::BasisSet obs(basis_name, atoms);
    obs.set_pure(false); // use cartesian gaussians

    assert(3 * atoms.size() == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");

    // Overlap derivative integral engine
    libint2::Engine v_engine(libint2::Operator::nuclear,obs.max_nprim(),obs.max_l(),deriv_order);
    v_engine.set_params(libint2::make_point_charges(atoms));
    const auto& buf_vec = v_engine.results(); // will point to computed shell sets

    // Get size of kinetic derivative array and allocate 
    int nbf = obs.nbf();
    int ncart = atoms.size() * 3;
    size_t length = nbf * nbf;
    std::vector<double> result(length);
    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
    const auto shell2atom = obs.shell2atom(atoms); // maps shell index to atom index
    
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

            // For potential integrals, we compute all derivatives wrt the two shell centers have the 2 shell centers making 6 coordinates
            // and every single nuclear derivative. These components must be summed to obtain 
            // overall nuclear derivative.
            // Collect primary shell derivatives 
            // The number of subectors in "indices" is equal to the order of differentiation 
            // The length of each subvector is equal to the number of shell centers which correspond to a desired (differentiated) atom according to deriv_vec
            std::vector<std::vector<int>> indices;
            std::vector<int> tmp;
            for (int j=0; j < desired_atom_indices.size(); j++){
                for (int i=0; i<2; i++){
                    int atom_idx = shell_atom_index_list[i];
                    int desired_atom_idx = desired_atom_indices[j];
                    if (atom_idx == desired_atom_idx) { 
                        tmp.push_back(3 * i + desired_coordinates[j]);
                    }
                }
                if (tmp.size() > 0) {
                    indices.push_back(tmp);
                }
                tmp.clear(); // wipe the temporary vector 
            }

            std::vector<int> buffer_indices;
            // Sometimes no shell set centers are being differentiated, so skip to nuclear part
            if (indices.size() != 0){
                // Loop over combinations of shell derivatives and look up buffer index
                // This obtains buffer indices corresponding to shell derivatives
                // TODO this seg faults.
                if (deriv_order == 1){
                  for (int i=0; i < indices[0].size(); i++){
                    int idx = indices[0][i]; 
                    buffer_indices.push_back(buffer_index_lookup1[idx]);
                  }
                }

                else if (deriv_order == 2){
                  for (int i=0; i < indices[0].size(); i++){
                    for (int j=0; j < indices[1].size(); j++){
                      int idx1 = indices[0][i]; 
                      int idx2 = indices[1][j]; 
                      buffer_indices.push_back(buffer_index_lookup2[idx1][idx2]);
                    }
                  }
                }
            }

            // Compute integrals
            v_engine.compute(obs[s1], obs[s2]); 

            // Now add in buffer indices corresponding to nuclear derivatives, contained at the end of the buffer 
            // There is only one of them.
            // Creates flattened 1d index of (ncart,ncart...ncart) array  
            // which effectively looks up nuclaer derivative operator index
            int tmp_idx = 0; 
            for (int i = 0; i < desired_atom_indices.size(); i++){
                tmp_idx *= ncart;
                tmp_idx += (3 * desired_atom_indices[i] + desired_coordinates[i]); 
            }

            // I have no idea how libint is packing these nuclear coordinates into the buffer.
            // I assume its just (All shell derivs) then (ncart^deriv_order nuclear derivs)
            // However H2 deriv1 has 18 elements in buffer... weird!
            // For now, lets convert tmp_idx to be starting from the last ncart^deriv_order indices
            buffer_indices.push_back(buf_vec.size() - pow(ncart,deriv_order) + tmp_idx);

            //printf("Buffer indices \n"); 
            //for(auto i=0; i<buffer_indices.size(); ++i) {
            //    printf("%ld ", buffer_indices[i]);
            //}

            //// TEMP Print all computed integrals 
            //printf("\nAtoms %ld %ld \n", atom1, atom2);
            //for(auto i=0; i<buf_vec.size(); ++i) {
            //  auto ints_shellset = buf_vec[i]; //TODO check for nullptr here?
            //  for(auto f1=0, idx=0; f1!=n1; ++f1) {
            //    for(auto f2=0; f2!=n2; ++f2, ++idx) {
            //        printf("%lf \n", ints_shellset[idx]);
            //    }
            //  }
            //}

            // Loop over every buffer index and accumulate for every shell set.
            for(auto i=0; i<buffer_indices.size(); ++i) {
              auto ints_shellset = buf_vec[buffer_indices[i]]; //TODO check for nullptr here?
              for(auto f1=0, idx=0; f1!=n1; ++f1) {
                for(auto f2=0; f2!=n2; ++f2, ++idx) {
                  result[(bf1 + f1) * nbf + bf2 + f2] += ints_shellset[idx]; 
                }
              }
            }
        }
    }
    libint2::finalize();
    return py::array(result.size(), result.data()); 
}

// Computes nuclear derivatives of electron repulsion integrals
py::array eri_deriv(std::string xyzfilename, std::string basis_name, std::vector<int> deriv_vec) {
    libint2::initialize();
    int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);
    // Lookup arrays for mapping shell derivative index to buffer index 
    // TODO move somewhere, change deriv orders to support 3rd, 4th
    assert(deriv_order <= 2);
    static const std::vector<int> buffer_index_lookup1 = {0,1,2,3,4,5,6,7,8,9,10,11};
    static const std::vector<std::vector<int>> buffer_index_lookup2 = {
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

    // Convert deriv_vec to set of atom indices and their cartesian components which we are differentiating wrt
    std::vector<int> desired_atom_indices;
    std::vector<int> desired_coordinates;
    process_deriv_vec(deriv_vec, &desired_atom_indices, &desired_coordinates);

    // Load basis set and geometry.
    std::vector<libint2::Atom> atoms = get_atoms(xyzfilename);
    // Make sure the number of atoms match the size of deriv_vec
    assert(3 * atoms.size() == deriv_vec.size() && "Derivative vector incorrect size for this molecule.");

    libint2::BasisSet obs(basis_name, atoms);
    obs.set_pure(false); // use cartesian gaussians

    // ERI derivative integral engine
    libint2::Engine eri_engine(libint2::Operator::coulomb,obs.max_nprim(),obs.max_l(),deriv_order);
    const auto& buf_vec = eri_engine.results(); // will point to computed shell sets

    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
    const auto shell2atom = obs.shell2atom(atoms); // maps shell index to atom index
    int nbf = obs.nbf();
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

                    // Collect primary shell derivatives 
                    // The number of subectors in "indices" is equal to the order of differentiation 
                    // The length of each subvector is equal to the number of shell centers which correspond to a desired (differentiated) atom according to deriv_vec
                    std::vector<std::vector<int>> indices;
                    std::vector<int> tmp;
                    for (int j=0; j < desired_atom_indices.size(); j++){
                        for (int i=0; i<4; i++){
                            int atom_idx = shell_atom_index_list[i];
                            int desired_atom_idx = desired_atom_indices[j];
                            if (atom_idx == desired_atom_idx) { 
                                tmp.push_back(3 * i + desired_coordinates[j]);
                            }
                        }
                        if (tmp.size() > 0) {
                            indices.push_back(tmp);
                        }
                        tmp.clear(); // wipe the temporary vector 
                    }
                    if (indices.size() == 0) continue; //TODO this is new, sometimes with larger molecules there is none

                    // Alternative to below: effectively takes the cartesian product of every subvector of 'indices' 
                    // would be general and avoid loop blocks for each case below

                    // Loop over combinations of shell derivatives and look up buffer index
                    std::vector<int> buffer_indices;
                    if (deriv_order == 1){
                      for (int i=0; i < indices[0].size(); i++){
                        int idx = indices[0][i]; 
                        buffer_indices.push_back(buffer_index_lookup1[idx]);
                      }
                    }

                    else if (deriv_order == 2){
                      for (int i=0; i < indices[0].size(); i++){
                        for (int j=0; j < indices[1].size(); j++){
                          int idx1 = indices[0][i]; 
                          int idx2 = indices[1][j]; 
                          buffer_indices.push_back(buffer_index_lookup2[idx1][idx2]);
                        }
                      }
                    }
                    
                    // This effectively checks that this shell quartet contains the desired derivative 
                    if (buffer_indices.size() == 0) continue;

                    // If we made it this far, the shell derivative we want is contained in the buffer. 
                    eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]); // Compute shell set, fills buf_vec

                    // This won't loop if none of the atoms in the shell quartet
                    for(auto i=0; i<buffer_indices.size(); ++i) {
                        auto ints_shellset = buf_vec[buffer_indices[i]];
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

