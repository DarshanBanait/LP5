### Install Required Python Packages

Make sure you're using **Python 3.10** (TensorFlow is not compatible with Python 3.13+).

```bash
pip install tensorflow keras scikit-learn seaborn numpy pandas matplotlib
```
or

```bash
python -m pip install tensorflow keras scikit-learn seaborn numpy pandas matplotlib
```
### For CUDA Code - Copy-Paste the code in [LeetGPU](https://www.leetgpu.com/playground)

### For Other HPC codes
```bash
g++ -fopenmp file_name.cpp -o file_name.exe

.\file_name.exe
```
| **Directive**                        | **Used In**                 | **Purpose**                                                                |
| ------------------------------------ | --------------------------- | -------------------------------------------------------------------------- |
| `#pragma omp parallel for`           | BFS, Bubble Sort, Reduction | Parallelizes loop iterations across threads                                |
| `#pragma omp critical`               | BFS, DFS, Reduction         | Ensures only one thread accesses a critical section at a time              |
| `#pragma omp task`                   | DFS                         | Creates a new task (for parallel recursive calls)                          |
| `#pragma omp taskwait`               | DFS                         | Waits until all tasks in the current context finish                        |
| `#pragma omp parallel`               | DFS                         | Starts a parallel region                                                   |
| `#pragma omp single`                 | DFS                         | Ensures only one thread executes the block (usually to spawn tasks)        |
| `#pragma omp parallel sections`      | Merge Sort                  | Allows independent sections to run in parallel                             |
| `#pragma omp section`                | Merge Sort                  | Defines individual sections within `parallel sections`                     |
| `#pragma omp parallel for reduction` | Reduction                   | Combines loop parallelism with safe reduction of values like sum, min, max |
