//
// Created by Michal on 01.06.2023.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <json-c/json.h>
#include "dijkstra.h"
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

void dijkstraSerial(Graph* graph, int source) {
    int vertices = graph->vertices;

    // Inicjalizacja odległości jako nieskończoność dla wszystkich wierzchołków
    std::vector<int> distances(vertices, INT_MAX);

    // Ustawienie odległości źródłowego wierzchołka na 0
    distances[source] = 0;

    // Tablica oznaczająca, czy dany wierzchołek został już odwiedzony
    std::vector<bool> visited(vertices, false);

    // Główna pętla algorytmu Dijkstry
    for (int i = 0; i < vertices - 1; ++i) {
        // Znalezienie wierzchołka o najmniejszej odległości z nieodwiedzonych wierzchołków
        int minDistance = INT_MAX;
        int minIndex = -1;
        for (int j = 0; j < vertices; ++j) {
            if (!visited[j] && distances[j] < minDistance) {
                minDistance = distances[j];
                minIndex = j;
            }
        }

        // Oznaczenie znalezionego wierzchołka jako odwiedzony
        visited[minIndex] = true;

        // Aktualizacja odległości dla sąsiadów znalezionego wierzchołka
        for (const Edge& edge : graph->edges) {
            if (edge.source == minIndex && !visited[edge.destination] &&
                distances[minIndex] != INT_MAX &&
                distances[minIndex] + edge.weight < distances[edge.destination]) {
                distances[edge.destination] = distances[minIndex] + edge.weight;
            }
        }
    }
}


void dijkstraOpenMP(Graph* graph, int source) {
    int vertices = graph->vertices;

    // Inicjalizacja odległości jako nieskończoność dla wszystkich wierzchołków
    std::vector<int> distances(vertices, INT_MAX);

    // Ustawienie odległości źródłowego wierzchołka na 0
    distances[source] = 0;

    // Tablica oznaczająca, czy dany wierzchołek został już odwiedzony
    std::vector<bool> visited(vertices, false);

    // Główna pętla algorytmu Dijkstry
    for (int i = 0; i < vertices - 1; ++i) {
        // Znalezienie wierzchołka o najmniejszej odległości z nieodwiedzonych wierzchołków
        int minDistance = INT_MAX;
        int minIndex = -1;

        // Rozpoczęcie równoległej sekcji
        #pragma omp parallel
        {
            int localMinDistance = INT_MAX;
            int localMinIndex = -1;

            // Rozpoczęcie równoległej iteracji po wierzchołkach
            #pragma omp for
            for (int j = 0; j < vertices; ++j) {
                if (!visited[j] && distances[j] < localMinDistance) {
                    localMinDistance = distances[j];
                    localMinIndex = j;
                }
            }

            // Znalezienie lokalnego minimum dla bieżącego wątku
            #pragma omp critical
            {
                if (localMinDistance < minDistance) {
                    minDistance = localMinDistance;
                    minIndex = localMinIndex;
                }
            }
        }

        // Oznaczenie znalezionego wierzchołka jako odwiedzony
        visited[minIndex] = true;

        // Aktualizacja odległości dla sąsiadów znalezionego wierzchołka
        #pragma omp parallel for
        for (int j = 0; j < graph->edges.size(); ++j) {
            const Edge& edge = graph->edges[j];
            if (edge.source == minIndex && !visited[edge.destination] &&
                distances[minIndex] != INT_MAX &&
                distances[minIndex] + edge.weight < distances[edge.destination]) {
                distances[edge.destination] = distances[minIndex] + edge.weight;
            }
        }
    }

    // Wyświetlenie odległości dla wszystkich wierzchołków
    for (int i = 0; i < vertices; ++i) {
        std::cout << "Distance from vertex " << source << " to vertex " << i << ": " << distances[i] << std::endl;
    }
}

__global__ void dijkstraCUDA(Graph* graph, int source, int* distances) {
    int vertices = graph->vertices;
    int* adjacencyMatrix = &(graph->adjacencyMatrix[0][0]);

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < vertices) {
        distances[threadId] = INF;
    }

    if (threadId == source) {
        distances[threadId] = 0;
    }

    __syncthreads();

    for (int i = 0; i < vertices - 1; ++i) {
        int minDist = INF;
        int minIndex = -1;

        for (int j = 0; j < vertices; ++j) {
            if (!visited[j] && distances[j] < minDist) {
                minDist = distances[j];
                minIndex = j;
            }
        }

        visited[minIndex] = true;

        for (int k = 0; k < vertices; ++k) {
            int edgeWeight = adjacencyMatrix[minIndex * vertices + k];
            if (!visited[k] && edgeWeight && (distances[minIndex] + edgeWeight < distances[k])) {
                distances[k] = distances[minIndex] + edgeWeight;
            }
        }
    }
}

void dijkstraCUDAWrapper(Graph* graph, int source) {
    int vertices = graph->vertices;
    int* distances = new int[vertices];
    int* cudaDistances;

    cudaMalloc((void**)&cudaDistances, vertices * sizeof(int));

    cudaMemcpy(cudaDistances, distances, vertices * sizeof(int), cudaMemcpyHostToDevice);

    dijkstraCUDA<<<(vertices + 255) / 256, 256>>>(graph, source, cudaDistances);

    cudaMemcpy(distances, cudaDistances, vertices * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(cudaDistances);

    // Wypisz wyniki
    for (int i = 0; i < vertices; ++i) {
        std::cout << "Vertex: " << i << ", Distance: " << distances[i] << std::endl;
    }

    delete[] distances;
}

Graph loadGraphFromFile(const char* filename) {
    Graph graph;
    graph.vertices = 0;
    graph.edges.clear();

    FILE* file = fopen(filename, "r");
    if (file != nullptr) {
        fseek(file, 0, SEEK_END);
        long fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);

        char* buffer = (char*)malloc(fileSize + 1);
        if (buffer != nullptr) {
            fread(buffer, 1, fileSize, file);
            buffer[fileSize] = '\0';

            json_object* jsonObj = json_tokener_parse(buffer);
            if (jsonObj != nullptr) {
                json_object* jsonVertices = nullptr;
                if (json_object_object_get_ex(jsonObj, "vertices", &jsonVertices)) {
                    graph.vertices = json_object_get_int(jsonVertices);
                }

                json_object* jsonEdges = nullptr;
                if (json_object_object_get_ex(jsonObj, "edges", &jsonEdges)) {
                    int numEdges = json_object_array_length(jsonEdges);
                    for (int i = 0; i < numEdges; i++) {
                        json_object* jsonEdge = json_object_array_get_idx(jsonEdges, i);
                        json_object* jsonSource = nullptr;
                        json_object* jsonDestination = nullptr;
                        json_object* jsonWeight = nullptr;

                        if (json_object_object_get_ex(jsonEdge, "source", &jsonSource) &&
                            json_object_object_get_ex(jsonEdge, "destination", &jsonDestination) &&
                            json_object_object_get_ex(jsonEdge, "weight", &jsonWeight)) {

                            int source = json_object_get_int(jsonSource);
                            int destination = json_object_get_int(jsonDestination);
                            int weight = json_object_get_int(jsonWeight);

                            graph.edges.push_back({source, destination, weight});
                        }
                    }
                }
            }

            free(buffer);
            json_object_put(jsonObj);
        }

        fclose(file);
    }

    return graph;
}


int dijkstra() {
    Graph graph = loadGraphFromFile("graph_40.json");

    // Wywołanie algorytmu Dijkstry w różnych wersjach
    int sourceVertex = 0;

    // Algorytm Dijkstry - wersja zwykła (sekwencyjna)
    dijkstraSerial(&graph, sourceVertex);

    // Algorytm Dijkstry - wersja równoległa
    dijkstraOpenMP(&graph, sourceVertex);

    // Algorytm Dijkstry - wersja równoległa (CUDA)
    dijkstraCUDA(&graph, sourceVertex);

    return 0;
}