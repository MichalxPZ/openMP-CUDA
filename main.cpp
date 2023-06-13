#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <omp.h>
#include <chrono>

struct Edge {
    int source;
    int destination;
    int weight;
};

Edge* generateRandomGraph(int vertices, int density, int maxEdges, int& numEdges) {
    Edge* edges = new Edge[maxEdges];

    // Generate all possible edges
    int index = 0;
    for (int i = 0; i < vertices - 1; i++) {
        for (int j = i + 1; j < vertices; j++) {
            edges[index].source = i;
            edges[index].destination = j;
            edges[index].weight = rand() % 100 + 1;
            index++;
        }
    }

    // Calculate the number of edges based on the desired density
    numEdges = (density * maxEdges) / 100;

    // Remove random edges to achieve the desired density
    int numEdgesToRemove = maxEdges - numEdges;
    while (numEdgesToRemove > 0) {
        int randomIndex = rand() % index;
        edges[randomIndex] = edges[index - 1];
        index--;
        numEdgesToRemove--;
    }

    return edges;
}

void dijkstra(const Edge* edges, int numEdges, int numVertices, int source) {
    auto startTime = std::chrono::high_resolution_clock::now();

    int* distance = new int[numVertices];
    bool* visited = new bool[numVertices];

    for (int i = 0; i < numVertices; ++i) {
        distance[i] = INT_MAX;
        visited[i] = false;

    }
    distance[source] = 0;

    for (int count = 0; count < numVertices - 1; ++count) {
        int minDistance = INT_MAX;
        int minVertex = -1;

        for (int v = 0; v < numVertices; ++v) {
            if (!visited[v] && distance[v] <= minDistance) {
                minDistance = distance[v];
                minVertex = v;
            }
        }

        visited[minVertex] = true;

        for (int i = 0; i < numEdges; ++i) {
            if (edges[i].source == minVertex) {
                int neighbor = edges[i].destination;
                int weight = edges[i].weight;

                if (!visited[neighbor] && distance[minVertex] != INT_MAX &&
                    distance[minVertex] + weight < distance[neighbor]) {
                    distance[neighbor] = distance[minVertex] + weight;
                }
            }
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

    printf("Execution time %d Serial: %lld microseconds\n", numVertices, duration);

//    for (int i = 0; i < numVertices; ++i) {
//        std::cout << "Odleglosc od wierzcholka " << source << " do wierzcholka " << i << ": ";
//        if (distance[i] == INT_MAX) {
//            std::cout << "Brak ścieżki" << std::endl;
//        } else {
//            std::cout << distance[i] << std::endl;
//        }
//    }

    delete[] distance;
    delete[] visited;
}


void dijkstraOpenMP(const Edge* edges, int numEdges, int numVertices, int source) {
    auto startTime = std::chrono::high_resolution_clock::now();

    int* distance = new int[numVertices];
    bool* visited = new bool[numVertices];

    #pragma omp parallel for
    for (int i = 0; i < numVertices; ++i) {
        distance[i] = INT_MAX;
        visited[i] = false;
    }
    #pragma omp barrier

    distance[source] = 0;

    for (int count = 0; count < numVertices - 1; ++count) {
        int minDistance = INT_MAX;
        int minVertex = -1;

        #pragma omp parallel for reduction(min:minDistance) reduction(min:minVertex)
        for (int v = 0; v < numVertices; ++v) {
            if (!visited[v] && distance[v] <= minDistance) {
                minDistance = distance[v];
                minVertex = v;
            }
        }
        visited[minVertex] = true;
        #pragma omp barrier

        #pragma omp parallel for
        for (int i = 0; i < numEdges; ++i) {
            if (edges[i].source == minVertex) {
                int neighbor = edges[i].destination;
                int weight = edges[i].weight;

                if (!visited[neighbor] && distance[minVertex] != INT_MAX &&
                    distance[minVertex] + weight < distance[neighbor]) {
                    distance[neighbor] = distance[minVertex] + weight;
                }
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

    printf("Execution time %d OPENMP: %lld microseconds\n", numVertices, duration);

//    for (int i = 0; i < numVertices; ++i) {
//        std::cout << "Odleglosc od wierzcholka " << source << " do wierzcholka " << i << ": ";
//        if (distance[i] == INT_MAX) {
//            std::cout << "Brak ścieżki" << std::endl;
//        } else {
//            std::cout << distance[i] << std::endl;
//        }
//    }
    delete[] distance;
    delete[] visited;
}


int main(int argc, char** argv){
    srand(time(nullptr));
    for (int vertices = 1000; vertices <= 15000; vertices += 1000) {
        int maxEdges = (vertices * (vertices - 1)) / 2;
        int numEdges = (50 * maxEdges) / 100;
        Edge* edges = generateRandomGraph(vertices, 50, maxEdges, numEdges);
        dijkstra(edges, numEdges, vertices, 0);
        dijkstraOpenMP(edges, numEdges, vertices, 0);
    }

    return 0;
}
