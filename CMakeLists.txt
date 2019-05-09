# project
cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)

project (rectelas)

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
#add_executable(elas ${LIBELAS_SRC_FILES})
#target_link_libraries( elas ${OpenCV_LIBS} )

find_library( LIBOPENCL OpenCL )
include_directories( ${OPENCL_INCLUDE_DIRS} )

include_directories(rectdetect)

set(TARGET_CLTOHEADER "cltoheader")
add_executable(${TARGET_CLTOHEADER} rectdetect/cltoheader.c)

add_custom_command(OUTPUT oclimgutil_cl.h
  COMMENT "Generating oclimgutil_cl.h"
  COMMAND $<TARGET_FILE:${TARGET_CLTOHEADER}> ${CMAKE_CURRENT_SOURCE_DIR}/rectdetect/oclimgutil.cl source > ${CMAKE_CURRENT_BINARY_DIR}/rectdetect/oclimgutil_cl.h
  DEPENDS ${TARGET_CLTOHEADER}
  )
add_custom_target(oclimgutil_cl.h_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/oclimgutil_cl.h)

add_custom_command(OUTPUT oclpolyline_cl.h 	
  COMMENT "Generating oclpolyline_cl.ceh"
  COMMAND $<TARGET_FILE:${TARGET_CLTOHEADER}> ${CMAKE_CURRENT_SOURCE_DIR}/rectdetect/oclpolyline.cl source > ${CMAKE_CURRENT_BINARY_DIR}/rectdetect/oclpolyline_cl.h
  DEPENDS ${TARGET_CLTOHEADER}
  )
add_custom_target(oclpolyline_cl.h_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/oclpolyline_cl.h)

add_custom_command(OUTPUT oclrect_cl.h
  COMMENT "Generating oclrect_cl.h"
  COMMAND $<TARGET_FILE:${TARGET_CLTOHEADER}> ${CMAKE_CURRENT_SOURCE_DIR}/rectdetect/oclrect.cl source > ${CMAKE_CURRENT_BINARY_DIR}/rectdetect/oclrect_cl.h
  DEPENDS ${TARGET_CLTOHEADER}
  )
add_custom_target(oclrect_cl.h_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/oclrect_cl.h)

add_library(helper_obj OBJECT rectdetect/oclhelper.c rectdetect/helper.c)

add_library(oclimgutil_obj OBJECT rectdetect/oclimgutil.c rectdetect/oclimgutil.h ${CMAKE_CURRENT_BINARY_DIR}/oclimgutil_cl.h)
add_dependencies(oclimgutil_obj oclimgutil_cl.h_generated)

add_library(oclpolyline_obj OBJECT rectdetect/oclpolyline.c rectdetect/oclpolyline.h ${CMAKE_CURRENT_BINARY_DIR}/oclpolyline_cl.h)
add_dependencies(oclpolyline_obj oclpolyline_cl.h_generated)

add_library(oclrect_obj OBJECT rectdetect/oclrect.c rectdetect/oclrect.h ${CMAKE_CURRENT_BINARY_DIR}/oclrect_cl.h)
add_dependencies(oclrect_obj oclrect_cl.h_generated)

add_executable( vidrect rectdetect/vidrect.cpp $<TARGET_OBJECTS:oclrect_obj> $<TARGET_OBJECTS:oclpolyline_obj> $<TARGET_OBJECTS:oclimgutil_obj> $<TARGET_OBJECTS:helper_obj>  ${LIBELAS_SRC_FILES})
target_link_libraries( vidrect ${OpenCV_LIBS} ${LIBOPENCL} )

