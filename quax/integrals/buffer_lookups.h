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
