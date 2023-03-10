CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

set(CMAKE_CXX_FLAGS "-std=c++17 -O2 -DNDEBUG  -W -Wall -pedantic -fopenmp -lpthread -lrt -lz")

include_directories(../include)

set (src
    main.cpp 
    io.cpp
)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_INSTALL_PREFIX}/bin)
add_executable(app ${src})

target_link_libraries(app PUBLIC hdf5 hdf5_cpp)

install(TARGETS app RUNTIME DESTINATION bin)
set_target_properties(app PROPERTIES OUTPUT_NAME "cusingler")

message(STATUS ${HDF5_LIB})
file(COPY ${HDF5_LIB}/libhdf5.so DESTINATION ${CMAKE_INSTALL_PREFIX}/lib FOLLOW_SYMLINK_CHAIN)
file(COPY ${HDF5_LIB}/libhdf5_cpp.so DESTINATION ${CMAKE_INSTALL_PREFIX}/lib FOLLOW_SYMLINK_CHAIN)
file(COPY ${HDF5_LIB}/libsz.so.2 DESTINATION ${CMAKE_INSTALL_PREFIX}/lib FOLLOW_SYMLINK_CHAIN)

