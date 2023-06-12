#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <json-c/json.h>

void generateRandomGraph(int vertices, int density, json_object* jsonArray) {
    int maxEdges = (vertices * (vertices - 1)) / 2;
    int numEdges = (density * maxEdges) / 100;

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


int main() {
    srand(time(nullptr));

    for (int vertices = 25; vertices <= 500; vertices += 25) {
        json_object* jsonArray = json_object_new_array();
        generateRandomGraph(vertices, 50, jsonArray);
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

    return 0;
}
