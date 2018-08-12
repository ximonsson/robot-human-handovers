cmake_minimum_required (VERSION 2.8)

project (thesis)

set (PROJECT_LIB_DIR ${CMAKE_SOURCE_DIR}/lib)
set (PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

set (CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set (EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_LIB_DIR})

include_directories (${PROJECT_INCLUDE_DIR})
link_directories (${PROJECT_LIB_DIR})

add_subdirectory (src/track)
add_subdirectory (src/object_pcd)
add_subdirectory (src/hand)
add_subdirectory (src/cluster)
add_subdirectory (src/rec)
add_subdirectory (src/extract)
