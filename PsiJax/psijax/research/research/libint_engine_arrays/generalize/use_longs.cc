#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>

// Combinations with repitition. 
// Combinations of size 'k' from 'n' elements stored in vector 'inp'.
// Requires instantiating vector 'out' and vector of vectors 'result', which stores every combination.
void cwr_recursion(std::vector<long int> inp, 
                          std::vector<long int> &out, 
                          std::vector<std::vector<long int>> &result, 
                          long int k, long int i, long int n)
{
    // base case: if combination size is k, add to result 
    if (out.size() == k){
        result.push_back(out);
        return;
    }
    for (long int j = i; j < n; j++){
        out.push_back(inp[j]);
        cwr_recursion(inp, out, result, k, j, n);
        // backtrack - remove current element from solution
        out.pop_back();
    }
}

// How many shell set derivatives
// k is number of centers, n is deriv order. Assumes coordinate-independent operator.
long int how_many_derivs(long int k, long int n) {
    long int val = 1;
    long int factorial = 1;
    for (long int i=0; i < n; i++) {
        val *= (3 * k + i);
        factorial *= i + 1;
    }
    val /= factorial;
    return val;
}

// Cartesian product of a series of vectors
std::vector<std::vector<long int>> cartesian_product (const std::vector<std::vector<long int>>& v) {
    std::vector<std::vector<long int>> s = {{}};
    for (const auto& u : v) {
        std::vector<std::vector<long int>> r;
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

// Create array which is of size (nderivs, deriv_order)
// which when indexed with a 1d buffer index (the flattened generalized upper triangle index)
// it maps it to the corresponding to multidimensional index tuple (i,j,k,...) where i<=j<=k<=...
// The multidimensional indices each correspond to a partial derivative operator, 
// and the value of each index indicates which parameter is being differentiated.
// For example, if there are 6 differentiable parameters, and deriv_order = 2,
// calling generate_multi_index_lookup(6,2) yields 'combos' such that combos[13] yields the pair [2,4]
// which matches the value and index pair in the below array:
// 0  1  2  3  4  5
//    6  7  8  9 10
//      11 12 13 14
//         15 16 17
//            18 19
//               20
// for 4center deriv_order=1 it is of shape (12, 1)
// for 4center deriv_order=2 it is of shape (78, 2)
// for 4center deriv_order=3 it is of shape (364, 3)
// for 4center deriv_order=4 it is of shape (1365, 4)
std::vector<std::vector<long int>> generate_multi_index_lookup(long int nparams, long int deriv_order) {
    using namespace std;
    // Generate vector of indices 0 through nparams-1
    vector<long int> inp;
    for (long int i = 0; i < nparams; i++) {
        inp.push_back(i);
    }
    // Generate all possible combinations with repitition. 
    // These are upper triangle indices, and the length of them is the total number of derivatives
    vector<long int> out;
    vector<vector<long int>> combos;
    cwr_recursion(inp, out, combos, deriv_order, 0, nparams);
    return combos;
}

std::vector<std::vector<std::vector<std::vector<long int>>>> generate_deriv_index_map(long int deriv_order, long int ncenters) {
    using namespace std;
    // Number of possible derivatives
    long int nderivs = how_many_derivs(ncenters, deriv_order); // e.g. for 4 center: 12,78,364,1365
    // Number of differentiable parameters in a shell set (assumes 3 cartesian components each center)
    long int nparams = ncenters * 3;

    // The BraKet type determines what the permutations are.
    vector<long int> swap_braket_perm;
    vector<long int> swap_bra_perm;
    vector<long int> swap_ket_perm;
    vector<vector<long int>> swap_combos;
    // TODO switch to checking BraKet types
    if (ncenters == 4){
        swap_braket_perm = {6,7,8,9,10,11,0,1,2,3,4,5};
        swap_bra_perm = {3,4,5,0,1,2,6,7,8,9,10,11};
        swap_ket_perm = {0,1,2,3,4,5,9,10,11,6,7,8};
        // All possible on/off combinations of swap_braket, swap_bra, and swap_ket
        swap_combos = {{0, 0, 0}, 
                       {0, 0, 1},
                       {0, 1, 0},
                       {0, 1, 1},
                       {1, 0, 0},
                       {1, 0, 1},
                       {1, 1, 0},
                       {1, 1, 1}};
    }
    // TODO switch to checking BraKet types
    // TODO should I add BraKet::xx_xs?
    if (ncenters == 3){
        swap_braket_perm = {0,1,2,3,4,5,6,7,8};
        swap_bra_perm = {0,1,2,3,4,5,6,7,8};
        swap_ket_perm = {0,1,2,6,7,8,3,4,5};
        swap_combos = {{0,0,0}, {0,0,1}};
    }
    // Initialize mapDerivIndex. First three dimensions are either 0 or 1
    // acting as a bool for whether to swap braket, swap bra, or swap ket.
    // Last index is the long integer map.
    // Note that BraKet::xx_xx uses the whole thing, BraKet::xs_xx, only need [0][0][:][:] slice
    // and BraKet::xx_sx, only need [0][:][0][:] slice
    vector<vector<vector<vector<long int>>>> mapDerivIndex(2, vector<vector<vector<long int>>> 
                                                     (2, vector<vector<long int>> 
                                                     (2, vector<long int>
                                                     (nderivs, 0))));


    // Get lookup which maps flattened upper triangle index to the multidimensional index 
    // in terms of full array axes. Each axis of this multidimensional array represents
    // a different partial derivative of the shell set
    vector<vector<long int>> lookup = generate_multi_index_lookup(nparams, deriv_order);
    
    // Now loop over every combination of swap_* 
    for (long int swap = 0; swap < swap_combos.size(); swap++){
        long int swap_braket = swap_combos[swap][0];
        long int swap_bra = swap_combos[swap][1];
        long int swap_ket = swap_combos[swap][2];
        // For every single derivative index, lookup its multidimensional index
        // and apply the current permutation rules for this Braket::*
        for (long int i = 0; i < nderivs; i++){
            vector<long int> multi_idx = lookup[i]; 
            vector<long int> permuted_indices;
            for (long int j=0; j < multi_idx.size(); j++){
                long int idx = multi_idx[j];
                if (swap_braket == 1) idx = swap_braket_perm[idx];
                if (swap_bra == 1) idx = swap_bra_perm[idx];
                if (swap_ket == 1) idx = swap_ket_perm[idx];
                permuted_indices.push_back(idx);
            }
            // Sort permuted indices long into order of upper triangle indices, i <= j <= k...
            sort(permuted_indices.begin(), permuted_indices.end()); 
             
            // Now we need to map back to the 1d index.
            // There are two easy ways to do this.
            // The first is to construct a (deriv_order)-dimensional array, where each element is a 1d buffer index.
            // Then we can just index the array with 'permuted_indices' to find the 1d index
            // This is very fast, but has large memory requirement at higher orders for storing the array.
            // Instead, we can loop through 'lookup' until 'permuted_indices' matches an entry, and then that entry is the desired 1d buffer index.
            // This requires many more loops, but is cheaper memory wise.
            //long int new_idx = 0;
            //for (long int k=0; k < nderivs; k++){ 
            //    if (permuted_indices == lookup[k]){ 
            //        new_idx = k;
            //        break;
            //    }
            //}

            // Using lower_bound wayyy faster than naive loop checking for equality. .
            long int new_idx = 0;
            auto it = lower_bound(lookup.begin(), lookup.end(), permuted_indices);
            if (it != lookup.end()) new_idx = it - lookup.begin();

            mapDerivIndex[swap_braket][swap_bra][swap_ket][i] = new_idx;
        }
    }
    return mapDerivIndex;
}


int main()
{
    // Test combinations with replacement
    std::vector<long int> inp {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    long int k = 2;
    long int n = inp.size();
    std::vector<long int> out;
    std::vector<std::vector<long int>> result;
    cwr_recursion(inp, out, result, k, 0, n);
    // printf int result
    //for (long int i=0;i<result.size();i++){
    //    printf intf(" \n ");
    //    for (long int j=0; j<result[i].size(); j++){
    //        printf intf("%d ", result[i][j]);
    //    }
    //}

    // Test generate_deriv_index_map. 
    // excellent. This is blazing fast.
    long int ncenters1 = 4;
    long int deriv_order1 = 1;
    std::vector<std::vector<std::vector<std::vector<long int>>>> mapDerivIndex1 = generate_deriv_index_map(deriv_order1, ncenters1);

    long int ncenters2 = 4;
    long int deriv_order2 = 2;
    std::vector<std::vector<std::vector<std::vector<long int>>>> mapDerivIndex2 = generate_deriv_index_map(deriv_order2, ncenters2);

    long int ncenters3 = 4;
    long int deriv_order3 = 3;
    std::vector<std::vector<std::vector<std::vector<long int>>>> mapDerivIndex3 = generate_deriv_index_map(deriv_order3, ncenters3);

    long int ncenters4 = 4;
    long int deriv_order4 = 4;
    std::vector<std::vector<std::vector<std::vector<long int>>>> mapDerivIndex4 = generate_deriv_index_map(deriv_order4, ncenters4);

    long int ncenters5 = 4;
    long int deriv_order5 = 5;
    std::vector<std::vector<std::vector<std::vector<long int>>>> mapDerivIndex5 = generate_deriv_index_map(deriv_order5, ncenters5);

    long int ncenters6 = 4;
    long int deriv_order6 = 6;
    std::vector<std::vector<std::vector<std::vector<long int>>>> mapDerivIndex6 = generate_deriv_index_map(deriv_order6, ncenters6);

    long int ncenters7 = 3;
    long int deriv_order7 = 1;
    std::vector<std::vector<std::vector<std::vector<long int>>>> mapDerivIndex7 = generate_deriv_index_map(deriv_order7, ncenters7);

    long int ncenters8 = 3;
    long int deriv_order8 = 2;
    std::vector<std::vector<std::vector<std::vector<long int>>>> mapDerivIndex8 = generate_deriv_index_map(deriv_order8, ncenters8);

    long int ncenters9 = 3;
    long int deriv_order9 = 3;
    std::vector<std::vector<std::vector<std::vector<long int>>>> mapDerivIndex9 = generate_deriv_index_map(deriv_order9, ncenters9);

    long int ncenters10 = 3;
    long int deriv_order10 = 4;
    std::vector<std::vector<std::vector<std::vector<long int>>>> mapDerivIndex10 = generate_deriv_index_map(deriv_order10, ncenters10);

    long int ncenters11 = 3;
    long int deriv_order11 = 5;
    std::vector<std::vector<std::vector<std::vector<long int>>>> mapDerivIndex11 = generate_deriv_index_map(deriv_order11, ncenters11);

    long int ncenters12 = 3;
    long int deriv_order12 = 6;
    std::vector<std::vector<std::vector<std::vector<long int>>>> mapDerivIndex12 = generate_deriv_index_map(deriv_order12, ncenters12);

    //long int ncenters8 = 4;
    //long int deriv_order8 = 7;
    //std::vector<std::vector<std::vector<std::vector<long int>>>> mapDerivIndex8 = generate_deriv_index_map(deriv_order8, ncenters8);


    //for (long int i=0;i<2;i++){
    //  printf("\n");
    //  for (long int j=0;j<2;j++){
    //    printf("\n");
    //    for (long int k=0;k<2;k++){
    //      printf("\n");
    //      for (long int l=0;l< mapDerivIndex8[i][j][k].size();l++){
    //        printf("%d ", mapDerivIndex8[i][j][k][l]);
    //      }
    //    }
    //  }
    //}
    //printf("\n\n");
     
    // Practice making arrays of vectors?

    // This is allegedly a array of vector<long int>
    //std::vector<long int> test[2];
    // This is allegedly a 2x2x2 array each element being a vector<long int>
    //std::vector<long int>  test[2][2][2];
    //std::fill(test.begin(), test.end(), std::vector<long int>(78));
    //test[0][0][0].push_back(1);
    //printf intf("\n %d \n", test[0][0][0][0]);
	return 0;
}
