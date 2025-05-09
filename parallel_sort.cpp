#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>

using namespace std;

// Utility: Print vector
void printVector(const vector<int>& arr) {
    for (int val : arr)
        cout << val << " ";
    cout << endl;
}

// ------------------ Parallel Bubble Sort -------------------
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    cout << "\n--- Parallel Bubble Sort Steps ---\n";
    for (int i = 0; i < n; ++i) {
        #pragma omp parallel for
        for (int j = i % 2; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                #pragma omp critical
                cout << "Thread " << omp_get_thread_num() 
                     << " swapped positions " << j << " and " << j + 1 << endl;
            }
        }
    }
}

// ------------------ Parallel Merge Sort --------------------
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1, n2 = right - mid;
    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; ++i) L[i] = arr[left + i];
    for (int j = 0; j < n2; ++j) R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    #pragma omp critical
    cout << "Thread " << omp_get_thread_num() << " is merging from " 
         << left << " to " << right << endl;

    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void parallelMergeSort(vector<int>& arr, int left, int right, int depth = 0) {
    if (left < right) {
        int mid = (left + right) / 2;
        if (depth <= 3) {
            #pragma omp parallel sections
            {
                #pragma omp section
                parallelMergeSort(arr, left, mid, depth + 1);
                #pragma omp section
                parallelMergeSort(arr, mid + 1, right, depth + 1);
            }
        } else {
            parallelMergeSort(arr, left, mid, depth + 1);
            parallelMergeSort(arr, mid + 1, right, depth + 1);
        }
        merge(arr, left, mid, right);
    }
}

// ----------------------- Main -----------------------------
int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> input(n);
    cout << "Enter " << n << " elements:\n";
    for (int i = 0; i < n; ++i)
        cin >> input[i];

    vector<int> arr;

    // Parallel Bubble Sort
    arr = input;
    double start = omp_get_wtime();
    parallelBubbleSort(arr);
    double end = omp_get_wtime();
    cout << "\nSorted array (Bubble Sort): ";
    printVector(arr);
    cout << "Time taken (Parallel Bubble Sort): " << fixed << setprecision(6) << (end - start) << " seconds\n";

    // Parallel Merge Sort
    arr = input;
    start = omp_get_wtime();
    cout << "\n--- Parallel Merge Sort Steps ---\n";
    parallelMergeSort(arr, 0, n - 1);
    end = omp_get_wtime();
    cout << "\nSorted array (Merge Sort): ";
    printVector(arr);
    cout << "Time taken (Parallel Merge Sort): " << fixed << setprecision(6) << (end - start) << " seconds\n";

    return 0;
}

/*
g++ -fopenmp parallel_sort.cpp -o parallel_sort.exe

.\parallel_sort.exe

Input:

Enter number of elements: 6

Enter 6 elements:
9 2 5 1 6 3

Parallel merge sort scales much better than bubble sort.

Bubble sort (even parallelized) is inefficient for large inputs.

For real applications, use merge/quick sort, not bubble sort.

| Algorithm     | Sequential Time Complexity | Parallel Time Complexity | Scalability**                |
| ------------- | -------------------------- | ------------------------ | ---------------------------- |
| Bubble Sort   | O(n²)                      | O(n² / P) (ideal case)   | Poor (due to dependencies)   |
| Merge Sort    | O(n log n)                 | O(log n) to O(n)         | Excellent (divide & conquer) |

*/
