    #include <iostream>
    #include <vector>
    #include <random>
    #include <json-c/json.h>

    struct Graph {
        int vertices{};
        std::vector<std::vector<int>> adjacencyMatrix;
        std::vector<int> distances; // Odległości w algorytmie Dijkstry
        std::vector<bool> visited; // Oznaczenie odwiedzonych wierzchołków w algorytmie Dijkstry
    };

    void generateRandomGraph(Graph& graph, int density) {
        int vertices = graph.vertices;
        int maxEdges = vertices * (vertices - 1) / 2;
        int edgeCount = density * maxEdges / 100;

        graph.adjacencyMatrix.assign(vertices, std::vector<int>(vertices, 0));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> distrib(0, vertices - 1);

        int count = 0;
        while (count < edgeCount) {
            int row = distrib(gen);
            int col = distrib(gen);
            if (row != col && graph.adjacencyMatrix[row][col] == 0) {
                graph.adjacencyMatrix[row][col] = 1;
                graph.adjacencyMatrix[col][row] = 1;
                count++;
            }
        }
    }

    void saveGraphToFile(const Graph& graph, const char* filename) {
        json_object* jsonGraph = json_object_new_object();

        json_object* jsonVertices = json_object_new_int(graph.vertices);
        json_object_object_add(jsonGraph, "vertices", jsonVertices);

        json_object* jsonAdjacencyMatrix = json_object_new_array();
        for (int i = 0; i < graph.vertices; i++) {
            json_object* jsonRow = json_object_new_array();
            for (int j = 0; j < graph.vertices; j++) {
                json_object_array_add(jsonRow, json_object_new_int(graph.adjacencyMatrix[i][j]));
            }
            json_object_array_add(jsonAdjacencyMatrix, jsonRow);
        }
        json_object_object_add(jsonGraph, "adjacencyMatrix", jsonAdjacencyMatrix);

        if (!graph.distances.empty()) {
            json_object* jsonDistances = json_object_new_array();
            for (int i = 0; i < graph.vertices; i++) {
                json_object_array_add(jsonDistances, json_object_new_int(graph.distances[i]));
            }
            json_object_object_add(jsonGraph, "distances", jsonDistances);
        }

        if (!graph.visited.empty()) {
            json_object* jsonVisited = json_object_new_array();
            for (int i = 0; i < graph.vertices; i++) {
                json_object_array_add(jsonVisited, json_object_new_boolean(graph.visited[i]));
            }
            json_object_object_add(jsonGraph, "visited", jsonVisited);
        }

        FILE* file = fopen(filename, "w");
        if (file != nullptr) {
            fprintf(file, "%s", json_object_to_json_string_ext(jsonGraph, JSON_C_TO_STRING_PRETTY));
            fclose(file);
            printf("Macierz sąsiedztwa została zapisana do pliku: %s\n", filename);
        } else {
            printf("Błąd podczas zapisywania pliku.\n");
        }

        json_object_put(jsonGraph);
    }

    int main() {
        std::vector<int> graphSizes = {10, 20, 30, 40, 50}; // Rozmiary grafów do wygenerowania
        int density = 50;

        for (int size : graphSizes) {
            Graph graph;
            graph.vertices = size;

            generateRandomGraph(graph, density);

            std::string filename = "graph_" + std::to_string(size) + ".json";
            saveGraphToFile(graph, filename.c_str());
        }

        return 0;
    }    #include <iostream>
    #include <vector>
    #include <random>
    #include <json-c/json.h>

    struct Graph {
        int vertices{};
        std::vector<std::vector<int>> adjacencyMatrix;
        std::vector<int> distances; // Odległości w algorytmie Dijkstry
        std::vector<bool> visited; // Oznaczenie odwiedzonych wierzchołków w algorytmie Dijkstry
    };

    void generateRandomGraph(Graph& graph, int density) {
        int vertices = graph.vertices;
        int maxEdges = vertices * (vertices - 1) / 2;
        int edgeCount = density * maxEdges / 100;

        graph.adjacencyMatrix.assign(vertices, std::vector<int>(vertices, 0));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> distrib(0, vertices - 1);

        int count = 0;
        while (count < edgeCount) {
            int row = distrib(gen);
            int col = distrib(gen);
            if (row != col && graph.adjacencyMatrix[row][col] == 0) {
                graph.adjacencyMatrix[row][col] = 1;
                graph.adjacencyMatrix[col][row] = 1;
                count++;
            }
        }
    }

    void saveGraphToFile(const Graph& graph, const char* filename) {
        json_object* jsonGraph = json_object_new_object();

        json_object* jsonVertices = json_object_new_int(graph.vertices);
        json_object_object_add(jsonGraph, "vertices", jsonVertices);

        json_object* jsonAdjacencyMatrix = json_object_new_array();
        for (int i = 0; i < graph.vertices; i++) {
            json_object* jsonRow = json_object_new_array();
            for (int j = 0; j < graph.vertices; j++) {
                json_object_array_add(jsonRow, json_object_new_int(graph.adjacencyMatrix[i][j]));
            }
            json_object_array_add(jsonAdjacencyMatrix, jsonRow);
        }
        json_object_object_add(jsonGraph, "adjacencyMatrix", jsonAdjacencyMatrix);

        if (!graph.distances.empty()) {
            json_object* jsonDistances = json_object_new_array();
            for (int i = 0; i < graph.vertices; i++) {
                json_object_array_add(jsonDistances, json_object_new_int(graph.distances[i]));
            }
            json_object_object_add(jsonGraph, "distances", jsonDistances);
        }

        if (!graph.visited.empty()) {
            json_object* jsonVisited = json_object_new_array();
            for (int i = 0; i < graph.vertices; i++) {
                json_object_array_add(jsonVisited, json_object_new_boolean(graph.visited[i]));
            }
            json_object_object_add(jsonGraph, "visited", jsonVisited);
        }

        FILE* file = fopen(filename, "w");
        if (file != nullptr) {
            fprintf(file, "%s", json_object_to_json_string_ext(jsonGraph, JSON_C_TO_STRING_PRETTY));
            fclose(file);
            printf("Macierz sąsiedztwa została zapisana do pliku: %s\n", filename);
        } else {
            printf("Błąd podczas zapisywania pliku.\n");
        }

        json_object_put(jsonGraph);
    }

    int main() {
        std::vector<int> graphSizes = {10, 20, 30, 40, 50}; // Rozmiary grafów do wygenerowania
        int density = 50;

        for (int size : graphSizes) {
            Graph graph;
            graph.vertices = size;

            generateRandomGraph(graph, density);

            std::string filename = "graph_" + std::to_string(size) + ".json";
            saveGraphToFile(graph, filename.c_str());
        }

        return 0;
    }