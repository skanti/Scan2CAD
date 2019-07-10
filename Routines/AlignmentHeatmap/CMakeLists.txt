cmake_minimum_required(VERSION 3.0)
project(alignment_heatmap)

# dependencies
find_package(Eigen3 3.3 REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build) # <-- output dir is build

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O1 -g -Wno-int-in-bool-context -fopenmp")

include_directories(../Base)
include_directories(./MatrixLib)
include_directories(./MatrixLib/matrix_structures)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(main main.cpp)
target_link_libraries(main ceres glog lapack blas jsoncpp)

