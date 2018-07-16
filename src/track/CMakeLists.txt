cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

project (track)

find_package (PCL 1.5 REQUIRED)

include_directories (${PCL_INCLUDE_DIRS} ${PROJECT_INCLUDE_DIR})
link_directories (${PCL_LIBRARY_DIRS} ${PROJECT_LIB_DIR})
add_definitions (${PCL_DEFINITIONS})

add_executable (track main.cpp object_tracker.cpp)
target_link_libraries (track ${PCL_LIBRARIES})

