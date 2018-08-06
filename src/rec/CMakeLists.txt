cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

project (rec)

find_package (freenect2 REQUIRED)
find_package (OpenCV REQUIRED)
find_package (PCL 1.8 REQUIRED)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-deprecated-declarations")
include_directories (${freenect2_INCLUDE_DIR} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories (${PCL_LIBRARY_DIRS} ${PROJECT_LIB_DIR})
add_definitions (${PCL_DEFINITIONS} ${OPENCV__DEFINITIONS})

add_executable (rec main.cpp)
target_link_libraries (rec ${freenect2_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
