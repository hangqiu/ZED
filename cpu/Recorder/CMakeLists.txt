SET(execName ZED_Recorder)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")
IF(WIN32) # Windows
SET(ZED_INCLUDE_DIRS $ENV{ZED_INCLUDE_DIRS})
	if (CMAKE_CL_64) # 64 bits
		SET(ZED_LIBRARIES $ENV{ZED_LIBRARIES_64})
		SET(Boost_LIBRARY_DIRS $ENV{BOOST_LIBRARY_DIR_64})
	else(CMAKE_CL_64) # 32 bits
		message("32bits compilation is no more available with CUDA7.0")
	endif(CMAKE_CL_64)
SET(ZED_LIBRARY_DIR $ENV{ZED_LIBRARY_DIR})

SET(Boost_INCLUDE_DIRS $ENV{BOOST_INCLUDE_DIRS})
SET(Boost_LIBRARY_DIRS $ENV{Boost_LIBRARY_DIRS})

SET(OPENCV_DIR $ENV{OPENCV_DIR})
 

ELSE()
find_package(ZED REQUIRED)
find_package(Boost 1.5 COMPONENTS program_options REQUIRED)
ENDIF(WIN32)

find_package(CUDA REQUIRED)
find_package(OpenCV 2.4 COMPONENTS core highgui imgproc REQUIRED)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

 
SET(SRC_FOLDER src)
FILE(GLOB_RECURSE SRC_FILES "${SRC_FOLDER}/*.cpp")

ADD_EXECUTABLE(${execName} ${SRC_FILES})
set_property(TARGET ${execName} PROPERTY OUTPUT_NAME "ZED Recorder")

add_definitions(-std=c++0x)

# Add the required libraries for linking:
TARGET_LINK_LIBRARIES(${execName}
                        ${ZED_LIBRARIES}
                        ${OpenCV_LIBS}
                        ${Boost_LIBRARIES}
                        ${CUDA_LIBRARIES} ${CUDA_npps_LIBRARY} ${CUDA_nppi_LIBRARY}
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3" ) # Release Perf mode
