cmake_minimum_required(VERSION 3.14)
project(cvtk_example CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
add_executable(cvtk_example src/example.cpp)
target_link_libraries(cvtk_example PRIVATE ${OpenCV_LIBS})
