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
void recur(std::vector<int> arr, std::vector<int> &out, std::vector<std::vector<int>> &result, int k, int i, int w, int n)
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
    for (int v = w; v < n; v++)
    {
	  for (int j = i; j < n; j++)
	  {
	  	// add current element arr[j] to the solution and recur with
	  	// same index j (as repeated elements are allowed in combinations)
        // Add one element from each sublist 
	  	out.push_back(arr[j]);
	  	out.push_back(arr[v]);
	  	recur(arr, out, result, k, j, v, n);

	  	// backtrack - remove current element from solution
	  	out.pop_back();
	  	out.pop_back();

	  	// code to handle duplicates - skip adjacent duplicate elements
          //while (j < n - 1 && arr[j] == arr[j + 1])
	      //    j++;
	  }
    }
}


int main()
{
    // If you have a vector of integers and want k combinations with replacement, no repeats
    std::vector<int> inp {2, 5, 8, 11};
    //std::vector<std::vector<int>> inp {{5, 8, 11}, {5, 8, 11}};
    //std::vector<std::vector<int>> inp {{5, 8, 11}, {5, 8, 11}, {5, 8, 11}};
    int k = 2;
    //int n = inp.size(); 
    int n = 3; //inp.size(); 
    // Need to initialize starting sizes for each vector in inp

    std::sort (inp.begin(), inp.end());
	// if array contains repeated elements, sort the array to handle duplicates combinations
    //for (int i=0;i<inp.size();i++){
    //    std::sort (inp[i].begin(), inp[i].end());
    //}
    
    vector<int> out;
    std::vector<std::vector<int>> result;
    recur(inp, out, result, k, 0, 0, n);

    //for (int i=0;i<inp.size();i++){
    //    printf(" \n ");
    //    for (int j=0; j<inp[i].size(); j++){
    //        printf("%d ", inp[i][j]);
    //    }
    //}
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


