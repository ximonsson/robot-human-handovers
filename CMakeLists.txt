cmake_minimum_required (VERSION 2.8)

project (thesis)

set (CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set (EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${EXECUTABLE_OUTPUT_PATH})
set (PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set (PROJECT_LIB_DIR ${CMAKE_SOURCE_DIR}/lib)

include_directories (${PROJECT_INCLUDE_DIR})

add_subdirectory (src/track)
