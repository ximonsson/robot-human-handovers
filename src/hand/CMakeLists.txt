cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

project (hand)

find_package (PCL REQUIRED)
find_package (OpenCV REQUIRED)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-deprecated-declarations -g3")
include_directories (${OpenCV__INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${PROJECT_INCLUDE_DIR})
link_directories (${OpenCV__LIBRARY_DIRS} ${PCL_LIBRARY_DIRS} ${PROJECT_LIB_DIR})
add_definitions (${PCL_DEFINITIONS} ${OPENCV__DEFINITIONS})

add_executable (hand main.cpp)
target_link_libraries (hand ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
