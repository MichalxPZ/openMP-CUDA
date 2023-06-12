#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <json-c/json.h>
#include <iostream>

struct Edge {
    int source;
    int destination;
    int weight;
};

void generateRandomGraph(int vertices, int density, json_object* jsonArray, int maxEdges, int numEdges) {

    // Generate all possible edges
    int index = 0;
    for (int i = 0; i < vertices - 1; i++) {
        for (int j = i + 1; j < vertices; j++) {
            struct json_object* jsonEdge = json_object_new_object();
            json_object_object_add(jsonEdge, "source", json_object_new_int(i));
            json_object_object_add(jsonEdge, "destination", json_object_new_int(j));
            json_object_object_add(jsonEdge, "weight", json_object_new_int(rand() % 100 + 1));
            json_object_array_add(jsonArray, jsonEdge);
            index++;
        }
    }

    // Remove random edges to achieve the desired density
    int numEdgesToRemove = maxEdges - numEdges;
    while (numEdgesToRemove > 0) {
        int randomIndex = rand() % index;
        json_object_array_del_idx(jsonArray, randomIndex, 1);
        numEdgesToRemove--;
        index--;
    }
}

Edge* readGraphFromJson(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == nullptr) {
        std::cout << "Failed to open file " << filename << " for reading." << std::endl;
        return nullptr;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* buffer = (char*)malloc(file_size + 1);
    if (buffer == nullptr) {
        std::cout << "Failed to allocate memory." << std::endl;
        fclose(file);
        return nullptr;
    }

    if (fread(buffer, 1, file_size, file) != file_size) {
        std::cout << "Failed to read file " << filename << "." << std::endl;
        free(buffer);
        fclose(file);
        return nullptr;
    }

    fclose(file);

    buffer[file_size] = '\0';

    json_object* jsonObject = json_tokener_parse(buffer);
    free(buffer);

    int arrayLength = json_object_array_length(jsonObject);
    Edge* edges = new Edge[arrayLength];

    for (int i = 0; i < arrayLength; i++) {
        json_object* jsonEdge = json_object_array_get_idx(jsonObject, i);

        Edge* edge = &edges[i];

        json_object_object_foreach(jsonEdge, key, val) {
            if (strcmp(key, "source") == 0) {
                edge->source = json_object_get_int(val);
            } else if (strcmp(key, "destination") == 0) {
                edge->destination = json_object_get_int(val);
            } else if (strcmp(key, "weight") == 0) {
                edge->weight = json_object_get_int(val);
            }
        }
    }

    json_object_put(jsonObject);

    return edges;
}

void printEdges(const Edge* edges) {
    if (edges == nullptr) {
        std::cout << "Invalid edges array." << std::endl;
        return;
    }

    // Determine the size of the edges array
    int numEdges = 0;
    while (edges[numEdges].source != 0 || edges[numEdges].destination != 0 || edges[numEdges].weight != 0) {
        numEdges++;
    }

    for (int i = 0; i < numEdges; i++) {
        const Edge* edge = &edges[i];
        std::cout << "Edge " << i << ": "
                  << "Source: " << edge->source << ", "
                  << "Destination: " << edge->destination << ", "
                  << "Weight: " << edge->weight << std::endl;
    }
}

void dijkstra(const Edge* edges, int numEdges, int numVertices, int source) {
    // Inicjalizacja odległości dla wszystkich wierzchołków jako nieskończoność
    int* distance = new int[numVertices];
    for (int i = 0; i < numVertices; ++i) {
        distance[i] = INT_MAX;
    }

    // Inicjalizacja odległości dla źródłowego wierzchołka jako 0
    distance[source] = 0;

    // Tablica odwiedzonych wierzchołków
    bool* visited = new bool[numVertices];
    for (int i = 0; i < numVertices; ++i) {
        visited[i] = false;
    }

    // Główna pętla algorytmu
    for (int count = 0; count < numVertices - 1; ++count) {
        int minDistance = INT_MAX;
        int minVertex = -1;

        // Znajdź wierzchołek o najmniejszej odległości spośród nieodwiedzonych wierzchołków
        for (int v = 0; v < numVertices; ++v) {
            if (!visited[v] && distance[v] <= minDistance) {
                minDistance = distance[v];
                minVertex = v;
            }
        }

        // Oznacz znaleziony wierzchołek jako odwiedzony
        visited[minVertex] = true;

        // Zaktualizuj odległości dla sąsiadujących wierzchołków
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

    // Wyświetl odległości dla wszystkich wierzchołków
    for (int i = 0; i < numVertices; ++i) {
        std::cout << "Odległość od wierzchołka " << source << " do wierzchołka " << i << ": ";
        if (distance[i] == INT_MAX) {
            std::cout << "Brak ścieżki" << std::endl;
        } else {
            std::cout << distance[i] << std::endl;
        }
    }

    // Zwolnienie pamięci
    delete[] distance;
    delete[] visited;
}



int main() {
    srand(time(nullptr));
    for (int vertices = 25; vertices <= 500; vertices += 25) {
        int maxEdges = (vertices * (vertices - 1)) / 2;
        int numEdges = (50 * maxEdges) / 100;
        json_object* jsonArray = json_object_new_array();
        generateRandomGraph(vertices, 50, jsonArray, maxEdges, numEdges);
        char filename[100];
        snprintf(filename, sizeof(filename), "graph_%d.json", vertices);
        FILE* file = fopen(filename, "w");
        if (file != nullptr) {
            fprintf(file, "%s", json_object_to_json_string_ext(jsonArray, JSON_C_TO_STRING_PRETTY));
            fclose(file);
            printf("Saved graph to %s\n\n", filename);
        } else {
            printf("Failed to open file %s for writing.\n\n", filename);
        }
        json_object_put(jsonArray);
    }

    for (int vertices = 25; vertices <= 25; vertices += 25) {
        int maxEdges = (vertices * (vertices - 1)) / 2;
        int numEdges = (50 * maxEdges) / 100;
        char filename[100];
        snprintf(filename, sizeof(filename), "graph_%d.json", vertices);
        Edge* edges = readGraphFromJson(filename);
//        printEdges(edges);
        dijkstra(edges, numEdges, vertices, 0);
    }

    return 0;
}
