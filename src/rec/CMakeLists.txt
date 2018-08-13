cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

project (rec)

find_package (freenect2 REQUIRED)
find_package (OpenCV REQUIRED)
find_package (PkgConfig REQUIRED)

include_directories (${freenect2_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} /usr/local/include/apriltag)
link_directories (${PROJECT_LIB_DIR})
add_definitions (${OPENCV__DEFINITIONS})

add_executable (rec main.cpp)
target_link_libraries (rec ${freenect2_LIBRARIES} ${OpenCV_LIBS} ${APRILTAG_LIBRARIES})
