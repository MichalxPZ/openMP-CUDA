%%cuda --name dijkstra.cu

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
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

__global__
void dijkstraCUDA(const Edge* edges, int numEdges, int numVertices, int source, int* distance, bool* visited) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numVertices)
        return;

    if (tid == source)
        distance[tid] = 0;
    else
        distance[tid] = INT_MAX;

    visited[tid] = false;

    __syncthreads();

    int minDistance, minVertex;

    minDistance = INT_MAX;
    minVertex = -1;

    // Znajdowanie wierzchołka o najmniejszej odległości spośród nieodwiedzonych wierzchołków
    for (int v = 0; v < numVertices; ++v) {
        if (!visited[v] && distance[v] <= minDistance) {
            minDistance = distance[v];
            minVertex = v;
        }


        visited[minVertex] = true;

        __syncthreads();

        if (minVertex != -1) {
            for (int j = 0; j < numEdges; ++j) {
                if (edges[j].source == minVertex) {
                    int neighbor = edges[j].destination;
                    int weight = edges[j].weight;

                    if (!visited[neighbor] && distance[minVertex] != INT_MAX &&
                        distance[minVertex] + weight < distance[neighbor]) {
                        distance[neighbor] = distance[minVertex] + weight;
                    }
                }
            }
        }

        __syncthreads();
    }
}
int* dijkstraParallel(const Edge* edges, int numEdges, int numVertices, int source) {
    auto startTime = std::chrono::high_resolution_clock::now();

    int* distance = new int[numVertices];
    bool* visited = new bool[numVertices];
    int* devDistance;
    bool* devVisited;
    Edge* devEdges;

    cudaMalloc(&devDistance, sizeof(int) * numVertices);
    cudaMalloc(&devVisited, sizeof(bool) * numVertices);
    cudaMalloc(&devEdges, sizeof(Edge) * numEdges);

    // Inicjalizacja distance i visited

    for (int i = 0; i < numVertices; ++i) {
        distance[i] = INT_MAX;
        visited[i] = false;
    }

    distance[source] = 0;

    // Kopiowanie danych z hosta do urządzenia

    cudaMemcpy(devDistance, distance, numVertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devVisited, visited, numVertices * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(devEdges, edges, numEdges * sizeof(Edge), cudaMemcpyHostToDevice);

    // Wywołanie funkcji dijkstraCUDA

    dijkstraCUDA<<<(numVertices + 255) / 256, 256>>>(devEdges, numEdges, numVertices, source, devDistance, devVisited);

    // Kopiowanie wyników z urządzenia do hosta

    cudaMemcpy(distance, devDistance, numVertices * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

    printf("Execution time %d CUDA: %lld microseconds\n", numVertices, duration);

    cudaFree(devDistance);
    cudaFree(devVisited);
    delete[] visited;

    return distance;
}

void printDistances(int* distances, int numVertices, int source) {
    std::cout << "Odleglosc od wierzcholka " << source << " do innych wierzcholkow:" << std::endl;
    for (int i = 0; i < numVertices; ++i) {
        if (distances[i] == INT_MAX) {
            std::cout << "Brak ścieżki do wierzchołka " << i << std::endl;
        } else {
            std::cout << "Odległość do wierzchołka " << i << ": " << distances[i] << std::endl;
        }
    }

    delete[] distances;
}


int main(int argc, char** argv){
    srand(time(nullptr));
    for (int vertices = 25; vertices <= 500; vertices += 25) {
        int maxEdges = (vertices * (vertices - 1)) / 2;
        int numEdges = (50 * maxEdges) / 100;
        Edge* edges = generateRandomGraph(vertices, 50, maxEdges, numEdges);
        // auto startTime = std::chrono::high_resolution_clock::now();
        //for (int i = 0; i < vertices; i++) {
        //     dijkstra(edges, numEdges, vertices, i);
        //}
//auto endTime = std::chrono::high_resolution_clock::now();
        //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        // printf("Execution time %d Serial: %lld microseconds\n", vertices, duration);
        int* distances = dijkstraParallel(edges, numEdges, vertices, 0);
        //printDistances(distances, vertices, 0);
    }

    return 0;
}
