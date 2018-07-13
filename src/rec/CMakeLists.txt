cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(rec)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ../../include)
link_directories(${PCL_LIBRARY_DIRS} ../../lib)
add_definitions(${PCL_DEFINITIONS})

add_executable (track main.cpp tracker.cpp)
target_link_libraries (track ${PCL_LIBRARIES})

