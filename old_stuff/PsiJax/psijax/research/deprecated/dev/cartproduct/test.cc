#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Cartesian product 
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


int main()
{
    // If you have a vector of integers and want k combinations with replacement, no repeats
    int deriv_order = 2;
    std::vector<std::vector<int>> indices; 
    for (int i=0;i<deriv_order; i++){
        std::vector<int> new_vec; 
        indices.push_back(new_vec);
    }
        
    for (int i=0;i<indices.size(); i++){
      for (int j=0;j<indices[i].size(); j++){
        printf("%d ", indices[i][j]);
      }
    }


    //std::vector<std::vector<int>> inp {{2, 5, 14, 14}, {2, 5}, {10,11},};
    std::vector<std::vector<int>> inp {{2, 5, 14}};
    std::vector<std::vector<int>> index_combos = cartesian_product(inp);
    // Test: print result
    for (int i=0;i<index_combos.size();i++){
        printf(" \n ");
        for (int j=0; j<index_combos[i].size(); j++){
            printf("%d ", index_combos[i][j]);
        }
    }
	return 0;
}


