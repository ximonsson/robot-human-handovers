cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

project (extract)

find_package (OpenCV REQUIRED)
find_package (PkgConfig REQUIRED)

set (ENV{PKG_CONFIG_PATH} "/usr/local/lib/pkgconfig")
pkg_check_modules (APRILTAG REQUIRED apriltag)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
include_directories (${OpenCV_INCLUDE_DIRS} /usr/local/include/apriltag)
link_directories (${PROJECT_LIB_DIR})
add_definitions (${OPENCV__DEFINITIONS})

add_executable (extract main)
target_link_libraries (extract ${OpenCV_LIBS} ${APRILTAG_LIBRARIES})

