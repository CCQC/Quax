#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void printVector(vector<int> const &out)
{
	for (auto it = out.begin(); it != out.end(); it++)
		cout << *it << " ";
	cout << '\n';
}

// Function to store all distinct combinations of length k 
// in a vector of vectors
// Function to print all distinct combinations of length k where
// repetition of elements is allowed
void recur(std::vector<int> arr, std::vector<int> &out, std::vector<std::vector<int>> &result, int k, int i, int n)
{
	// base case: if combination size is k, add to result 
	if (out.size() == k)
	{
        printVector(out);
        result.push_back(out);
		return;
	}

	// start from previous element in the current combination
	// till last element
	for (int j = i; j < n; j++)
	{
		// add current element arr[j] to the solution and recur with
		// same index j (as repeated elements are allowed in combinations)
		out.push_back(arr[j]);
		recur(arr, out, result, k, j, n);

		// backtrack - remove current element from solution
		out.pop_back();

		// code to handle duplicates - skip adjacent duplicate elements
		while (j < n - 1 && arr[j] == arr[j + 1])
			j++;
	}
}


int main()
{
    // If you have a vector of integers and want k combinations with replacement, no repeats
    std::vector<int> inp {0, 1, 2, 3, 4, 5};
    int k = 2;
    int n = inp.size(); 
	// if array contains repeated elements, sort the array to handle duplicates combinations
    std::sort (inp.begin(), inp.end());
    
    vector<int> out;
    std::vector<std::vector<int>> result;
    recur(inp, out, result, k, 0, n);

    // Test: print result
    //for (int i=0;i<result.size();i++){
    //    printf(" \n ");
    //    for (int j=0; j<result[i].size(); j++){
    //        printf("%d ", result[i][j]);
    //    }
    //}
    //printf(" \n ");
	return 0;
}


