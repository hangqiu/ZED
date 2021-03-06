SET(execName ZED_openGL_gpu_interop)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")

IF(WIN32) # Windows
SET(ZED_INCLUDE_DIRS $ENV{ZED_INCLUDE_DIRS})
SET(ZED_LIBRARIES $ENV{ZED_LIBRARIES_64})
SET(GLUT_LIBRARY_DIRS $ENV{GLUT_LIBRARY_DIRS})
SET(GLEW_LIBRARY_DIRS $ENV{GLEW_LIBRARY_DIRS})
SET(GLEW_LIBRARIES $ENV{GLEW_LIBRARY})

SET(ZED_LIBRARY_DIR $ENV{ZED_LIBRARY_DIR})
SET(GLUT_INCLUDE_DIRS $ENV{GLUT_INCLUDE_DIRS})
SET(GLEW_INCLUDE_DIRS $ENV{GLEW_INCLUDE_DIRS})

SET(OPENCV_DIR $ENV{OPENCV_DIR})
ELSE() # Linux
find_package(ZED REQUIRED)
find_package(GLUT)
find_package(GLEW)
ENDIF(WIN32)

# Mandatory
find_package(CUDA REQUIRED)
find_package(OpenGL)
find_package(OpenCV 2.4 COMPONENTS core highgui imgproc REQUIRED)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

SET(SRC_FOLDER src)
FILE(GLOB_RECURSE SRC_FILES "${SRC_FOLDER}/*.cpp")

ADD_EXECUTABLE(${execName} ${SRC_FILES})
set_property(TARGET ${execName} PROPERTY OUTPUT_NAME "ZED openGL gpu interop")

add_definitions(-std=c++0x)# -m64) #-Wall)

# Add the required libraries for linking:
TARGET_LINK_LIBRARIES(${execName}
                        ${ZED_LIBRARIES}
			${GLEW_LIBRARIES}
                        ${OPENGL_LIBRARIES}
						                        ${OpenCV_LIBRARIES}
                        ${GLUT_LIBRARY}
			${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY}
                    )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3" ) # Release Perf mode
