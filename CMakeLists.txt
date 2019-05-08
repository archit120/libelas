# project
cmake_minimum_required (VERSION 2.6)
project (libelas)

# directories
set (LIBELAS_SRC_DIR src)

# include directory
include_directories("${LIBELAS_SRC_DIR}")

# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-msse3")

# sources
FILE(GLOB LIBELAS_SRC_FILES "src/*.cpp")

# make release version
set(CMAKE_BUILD_TYPE Release)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )




# build demo program
add_executable(elas ${LIBELAS_SRC_FILES})
target_link_libraries( elas ${OpenCV_LIBS} )
