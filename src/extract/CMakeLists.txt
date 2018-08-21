cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

project (extract)

find_package (OpenCV REQUIRED)
find_package (PkgConfig REQUIRED)

set (ENV{PKG_CONFIG_PATH} "/usr/local/lib/pkgconfig")
pkg_check_modules (APRILTAG REQUIRED apriltag)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
include_directories (${OpenCV_INCLUDE_DIRS} /usr/local/include/apriltag)
link_directories (${PROJECT_LIB_DIR})
add_definitions (${OPENCV__DEFINITIONS})

add_executable (extract_handover handover_extract)
add_executable (extract_tagid tagid_extract)

target_link_libraries (extract_handover ${OpenCV_LIBS} ${APRILTAG_LIBRARIES})
target_link_libraries (extract_tagid ${OpenCV_LIBS} ${APRILTAG_LIBRARIES})

