#include <iostream>
#include <vector>
#include <limits>
#include <omp.h>
#include <iomanip>
using namespace std;

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr(n);
    cout << "Enter " << n << " elements:\n";
    for (int i = 0; i < n; ++i)
        cin >> arr[i];

    int global_min = INT_MAX;
    int global_max = INT_MIN;
    long long sum = 0;

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(min:global_min) reduction(max:global_max) reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        // For debug info
        #pragma omp critical
        {
            if (arr[i] < global_min)
                cout << "Thread " << omp_get_thread_num() << " updated min to " << arr[i] << endl;
            if (arr[i] > global_max)
                cout << "Thread " << omp_get_thread_num() << " updated max to " << arr[i] << endl;
        }

        global_min = min(global_min, arr[i]);
        global_max = max(global_max, arr[i]);
        sum += arr[i];
    }

    double end = omp_get_wtime();
    double avg = (double)sum / n;

    cout << fixed << setprecision(6);
    cout << "\nResults using Parallel Reduction:\n";
    cout << "Minimum: " << global_min << endl;
    cout << "Maximum: " << global_max << endl;
    cout << "Sum    : " << sum << endl;
    cout << "Average: " << avg << endl;
    cout << "Execution time: " << (end - start) << " seconds\n";

    return 0;
}
/*
g++ -fopenmp parallel_reduction.cpp -o parallel_reduction.exe

.\parallel_reduction.exe

| Operation  | Complexity (Sequential) | Complexity (Parallel) | Notes                                   |
| ---------- | ----------------------- | --------------------- | --------------------------------------- |
| Finding Min| O(n)                    | O(n / P)              | Parallelized using `reduction(min:...)` |
| Finding Max| O(n)                    | O(n / P)              | Parallelized using `reduction(max:...)` |
| Sum        | O(n)                    | O(n / P)              | Parallelized using `reduction(+:...)`   |
| Average    | O(1)                    | O(1)                  | Just one division                       |

*/