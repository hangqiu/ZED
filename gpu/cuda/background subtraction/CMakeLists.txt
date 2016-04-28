SET(execName ZED_background_subtraction)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")
IF(WIN32) # Windows
SET(ZED_INCLUDE_DIRS $ENV{ZED_INCLUDE_DIRS})
	if (CMAKE_CL_64) # 64 bits
		SET(ZED_LIBRARIES $ENV{ZED_LIBRARIES_64})
	else(CMAKE_CL_64) # 32 bits
		message("32bits compilation is no more available with CUDA7.0")
	endif(CMAKE_CL_64)
SET(ZED_LIBRARY_DIR $ENV{ZED_LIBRARY_DIR})
ELSE() # Linux
find_package(ZED 0.9 REQUIRED)
ENDIF(WIN32)

# Mandatory
find_package(CUDA REQUIRED)
find_package(OpenCV 2.4 COMPONENTS core highgui imgproc REQUIRED)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

SET(SRC_FOLDER src)
FILE(GLOB_RECURSE SRC_FILES "${SRC_FOLDER}/*.cpp")

FILE(GLOB_RECURSE SRC_CU_FOLDER "${SRC_FOLDER}/*.cu")


list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_21)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_35)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_53)
IF(WIN32)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_52)
ENDIF(WIN32)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)

cuda_add_executable(${execName} ${SRC_FILES} ${SRC_CU_FOLDER})


set_property(TARGET ${execName} PROPERTY OUTPUT_NAME "ZED background subtraction")
add_definitions(-std=c++11)# -m64)

# Add the required libraries for linking:
TARGET_LINK_LIBRARIES(${execName}
			${ZED_LIBRARIES}
			${OpenCV_LIBRARIES}
			${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY}
			)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3" ) # Release Perf mode
