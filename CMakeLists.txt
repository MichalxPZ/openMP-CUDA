cmake_minimum_required(VERSION 3.25)
project(pRow)

# Ustawienie wersji języka C++
set(CMAKE_CXX_STANDARD 11)

# Dodanie flag kompilacji dla OpenMP
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Xpreprocessor -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Xpreprocessor -fopenmp")

# Linkowanie z biblioteką OpenMP
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")

# Znalezienie biblioteki OpenMP
find_package(OpenMP REQUIRED)


add_executable(pRow main.cpp dijkstra.cpp dijkstra.h graphGenerator.cpp graphGenerator.h)

find_package(json-c CONFIG)
target_link_libraries(${PROJECT_NAME} PRIVATE json-c::json-c program PRIVATE OpenMP::OpenMP_CXX)