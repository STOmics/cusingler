CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()


set(CMAKE_CXX_FLAGS "-std=c++17 -O2 -DNDEBUG  -W -Wall -pedantic -fopenmp -lpthread -lrt -lz ")
# set(CMAKE_CXX_FLAGS "-std=c++17 -g -DEBUG  -W -Wall -pedantic -fopenmp -lpthread -lrt -lz")

include_directories(../include)

set (src
    main.cpp 
    io.cpp
    cusingler.cu
    pipeline.cpp
)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_INSTALL_PREFIX}/bin)
add_executable(app ${src})

target_link_libraries(app PUBLIC hdf5 hdf5_cpp cudart)

install(TARGETS app RUNTIME DESTINATION bin)
set_target_properties(app PROPERTIES OUTPUT_NAME "cusingler")

file(COPY ${HDF5_LIB}/libhdf5.so DESTINATION ${CMAKE_INSTALL_PREFIX}/lib FOLLOW_SYMLINK_CHAIN)
file(COPY ${HDF5_LIB}/libhdf5_cpp.so DESTINATION ${CMAKE_INSTALL_PREFIX}/lib FOLLOW_SYMLINK_CHAIN)
file(COPY ${HDF5_LIB}/libsz.so.2 DESTINATION ${CMAKE_INSTALL_PREFIX}/lib FOLLOW_SYMLINK_CHAIN)

