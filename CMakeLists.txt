cmake_minimum_required(VERSION 3.5)
project(cusingler)

enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ version selection")

add_subdirectory(pybind11)
pybind11_add_module(cusingler 
    src/main.cpp
    src/cusingler.cu
    src/io.cpp
    src/pipeline.cpp
)


target_link_libraries(cusingler PUBLIC hdf5 hdf5_cpp cudart)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(cusingler
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
