#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <thread>
#include <chrono>

using namespace std;

class Graph {
public:
    int V;
    vector<vector<int>> adj;

    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // undirected graph
    }

    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;
        vector<int> bfsOrder;

        visited[start] = true;
        q.push(start);

        cout << "\nParallel BFS starting from node " << start << ":\n";

        while (!q.empty()) {
            int levelSize = q.size();

            #pragma omp parallel for
            for (int i = 0; i < levelSize; ++i) {
                int node;

                #pragma omp critical
                {
                    if (!q.empty()) {
                        node = q.front();
                        q.pop();
                    } else {
                        node = -1;
                    }
                }

                if (node != -1) {
                    #pragma omp critical
                    {
                        cout << "Visited " << node << " by thread " << omp_get_thread_num() << endl;
                        bfsOrder.push_back(node);
                    }

                    for (int neighbor : adj[node]) {
                        #pragma omp critical
                        {
                            if (!visited[neighbor]) {
                                visited[neighbor] = true;
                                q.push(neighbor);
                            }
                        }
                    }
                }
            }
        }

        cout << "BFS traversal order: ";
        for (int node : bfsOrder)
            cout << node << " ";
        cout << endl;
    }

    void parallelDFSUtil(int node, vector<bool>& visited, vector<int>& dfsOrder) {
        bool alreadyVisited;

        #pragma omp critical
        {
            alreadyVisited = visited[node];
            if (!alreadyVisited) {
                visited[node] = true;
                cout << "Visited " << node << " by thread " << omp_get_thread_num() << endl;
                dfsOrder.push_back(node);
            }
        }

        if (alreadyVisited) return;

        #pragma omp parallel for
        for (int i = 0; i < adj[node].size(); ++i) {
            int neighbor = adj[node][i];
            #pragma omp task
            parallelDFSUtil(neighbor, visited, dfsOrder);
        }

        #pragma omp taskwait
    }

    void parallelDFS(int start) {
        vector<bool> visited(V, false);
        vector<int> dfsOrder;

        cout << "\nParallel DFS starting from node " << start << ":\n";
        #pragma omp parallel
        {
            #pragma omp single
            {
                parallelDFSUtil(start, visited, dfsOrder);
            }
        }

        cout << "DFS traversal order: ";
        for (int node : dfsOrder)
            cout << node << " ";
        cout << endl;
    }
};

int main() {
    int V, E;
    cout << "Enter number of vertices: ";
    cin >> V;

    cout << "Enter number of edges: ";
    cin >> E;

    Graph g(V);

    cout << "Enter " << E << " edges (u v format, 0-based index):\n";
    for (int i = 0; i < E; ++i) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    int start;
    cout << "Enter starting node for traversal: ";
    cin >> start;

    g.parallelBFS(start);
    this_thread::sleep_for(chrono::seconds(2)); // optional delay for clarity in output
    g.parallelDFS(start);

    return 0;
}
/*
g++ -fopenmp parallel_graph.cpp -o parallel_graph.exe

.\parallel_graph.exe

Input:

Enter number of vertices: 6
Enter number of edges: 7
0 1
0 2
1 3
1 4
2 4
3 5
4 5
Enter starting node for traversal: 0

*/