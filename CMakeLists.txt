cmake_minimum_required(VERSION 3.25)
project(pRow)

set(CMAKE_CXX_STANDARD 17)

add_executable(pRow main.cpp)

find_package(json-c CONFIG)
target_link_libraries(${PROJECT_NAME} PRIVATE json-c::json-c)