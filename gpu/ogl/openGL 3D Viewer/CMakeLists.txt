SET(execName ZED_openGL_3D_Viewer)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")
SET(LINUX_ONLY_LIBRARIES "")
IF(WIN32) # Windows
SET(ZED_INCLUDE_DIRS $ENV{ZED_INCLUDE_DIRS})
	if (CMAKE_CL_64) # 64 bits
		SET(ZED_LIBRARIES $ENV{ZED_LIBRARIES_64})
		 SET(GLUT_LIBRARY_DIRS $ENV{GLUT_LIBRARY_DIRS})
		SET(GLUT_LIBRARY $ENV{GLUT_LIBRARY})
	else(CMAKE_CL_64) # 32 bits
		 		message("32bits compilation is no more available with CUDA7.0")
	endif(CMAKE_CL_64)
SET(ZED_LIBRARY_DIR $ENV{ZED_LIBRARY_DIR})
SET(GLUT_INCLUDE_DIRS $ENV{GLUT_INCLUDE_DIRS})
ELSE() # Linux
find_package(ZED 0.9 REQUIRED)
find_package(GLUT)
ENDIF(WIN32)

# Mandatory
find_package(CUDA REQUIRED)
find_package(OpenGL)
find_package(OpenCV 2.4 COMPONENTS core highgui imgproc REQUIRED)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

SET(SRC_FOLDER src)
FILE(GLOB_RECURSE SRC_FILES "${SRC_FOLDER}/*.cpp")

ADD_EXECUTABLE(${execName} ${SRC_FILES})
set_property(TARGET ${execName} PROPERTY OUTPUT_NAME "ZED openGL 3D Viewer")
add_definitions(-std=c++0x)# -m64)

# Add the required libraries for linking:
TARGET_LINK_LIBRARIES(${execName}
			${ZED_LIBRARIES}
			${OPENGL_LIBRARIES}
            ${GLUT_LIBRARY}
			${OpenCV_LIBRARIES}
			${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY}
 )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3" ) # Release Perf mode
