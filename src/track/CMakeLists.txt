cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

project (track)

find_package (PCL 1.5 REQUIRED)

#set (CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -Wno-deprecated -Wno-deprecated-declarations)
set (CMAKE_C_FLAGS -Wno-deprecated -Wno-deprecated-declarations)

include_directories (${PCL_INCLUDE_DIRS} ${PROJECT_INCLUDE_DIR})
link_directories (${PCL_LIBRARY_DIRS} ${PROJECT_LIB_DIR})
add_definitions (${PCL_DEFINITIONS})

add_executable (track object_tracker.cpp main.cpp)
target_link_libraries (track ${PCL_LIBRARIES})
