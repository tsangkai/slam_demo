cmake_minimum_required(VERSION 3.18.0)
project(slam_demo)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

option(BUILD_SHARED_LIBS "build shared libraries" ON)


find_package(OpenCV REQUIRED)
message("-- OpenCV version: ${OpenCV_VERSION}")

find_package(Eigen3 REQUIRED)
message("-- Eigen3 version: ${Eigen3_VERSION}")

find_package(Ceres REQUIRED)


add_subdirectory(lib)
add_subdirectory(external)

add_subdirectory(sim)
add_subdirectory(exp)


# add_subdirectory(test)
