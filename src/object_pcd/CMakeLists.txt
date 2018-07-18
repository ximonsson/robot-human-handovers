cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

project (object_pcd)

find_package (PCL 1.5 REQUIRED)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-deprecated-declarations")
include_directories (${PCL_INCLUDE_DIRS} ${PROJECT_INCLUDE_DIR})
link_directories (${PCL_LIBRARY_DIRS} ${PROJECT_LIB_DIR})
add_definitions (${PCL_DEFINITIONS})

add_executable (object_pcd main.cpp)
target_link_libraries (object_pcd ${PCL_LIBRARIES})
