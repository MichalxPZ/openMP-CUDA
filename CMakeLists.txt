cmake_minimum_required(VERSION 3.25)
project(pRow)

# Ustawienie wersji języka C++
set(CMAKE_CXX_STANDARD 11)

# Dodanie flag kompilacji dla OpenMP
add_executable(pRow main.cpp)

find_package(json-c CONFIG)
target_link_libraries(${PROJECT_NAME} PRIVATE json-c::json-c)