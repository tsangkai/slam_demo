cmake_minimum_required(VERSION 3.5.1)
project(slam_demo)


set(CMAKE_CXX_STANDARD 14)

option(BUILD_TEST "build test programs" ON)



find_package(OpenCV REQUIRED)
message("-- OpenCV version: ${OpenCV_VERSION}")

find_package(Eigen3 REQUIRED)

find_package(Ceres REQUIRED)

find_package(Boost COMPONENTS system REQUIRED)

add_subdirectory(lib)
add_subdirectory(apps)
add_subdirectory(sim)


if(BUILD_TEST)
  add_subdirectory(test)
endif(BUILD_TEST)