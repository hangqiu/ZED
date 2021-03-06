# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/usr/local/zed/sample/cpu/SVO Playback"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/usr/local/zed/sample/cpu/SVO Playback"

# Include any dependencies generated for this target.
include CMakeFiles/ZED_SVO_Playback.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ZED_SVO_Playback.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ZED_SVO_Playback.dir/flags.make

CMakeFiles/ZED_SVO_Playback.dir/src/main.o: CMakeFiles/ZED_SVO_Playback.dir/flags.make
CMakeFiles/ZED_SVO_Playback.dir/src/main.o: src/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report "/usr/local/zed/sample/cpu/SVO Playback/CMakeFiles" $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/ZED_SVO_Playback.dir/src/main.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/ZED_SVO_Playback.dir/src/main.o -c "/usr/local/zed/sample/cpu/SVO Playback/src/main.cpp"

CMakeFiles/ZED_SVO_Playback.dir/src/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ZED_SVO_Playback.dir/src/main.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E "/usr/local/zed/sample/cpu/SVO Playback/src/main.cpp" > CMakeFiles/ZED_SVO_Playback.dir/src/main.i

CMakeFiles/ZED_SVO_Playback.dir/src/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ZED_SVO_Playback.dir/src/main.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S "/usr/local/zed/sample/cpu/SVO Playback/src/main.cpp" -o CMakeFiles/ZED_SVO_Playback.dir/src/main.s

CMakeFiles/ZED_SVO_Playback.dir/src/main.o.requires:
.PHONY : CMakeFiles/ZED_SVO_Playback.dir/src/main.o.requires

CMakeFiles/ZED_SVO_Playback.dir/src/main.o.provides: CMakeFiles/ZED_SVO_Playback.dir/src/main.o.requires
	$(MAKE) -f CMakeFiles/ZED_SVO_Playback.dir/build.make CMakeFiles/ZED_SVO_Playback.dir/src/main.o.provides.build
.PHONY : CMakeFiles/ZED_SVO_Playback.dir/src/main.o.provides

CMakeFiles/ZED_SVO_Playback.dir/src/main.o.provides.build: CMakeFiles/ZED_SVO_Playback.dir/src/main.o

# Object files for target ZED_SVO_Playback
ZED_SVO_Playback_OBJECTS = \
"CMakeFiles/ZED_SVO_Playback.dir/src/main.o"

# External object files for target ZED_SVO_Playback
ZED_SVO_Playback_EXTERNAL_OBJECTS =

ZED\ SVO\ Playback: CMakeFiles/ZED_SVO_Playback.dir/src/main.o
ZED\ SVO\ Playback: CMakeFiles/ZED_SVO_Playback.dir/build.make
ZED\ SVO\ Playback: /usr/local/zed/lib/libsl_zed.so
ZED\ SVO\ Playback: /usr/local/zed/lib/libsl_depthcore.so
ZED\ SVO\ Playback: /usr/local/zed/lib/libsl_calibration.so
ZED\ SVO\ Playback: /usr/local/zed/lib/libcudpp.so
ZED\ SVO\ Playback: /usr/local/zed/lib/libcudpp_hash.so
ZED\ SVO\ Playback: /usr/local/lib/libopencv_core.so.2.4.10
ZED\ SVO\ Playback: /usr/local/lib/libopencv_highgui.so.2.4.10
ZED\ SVO\ Playback: /usr/local/lib/libopencv_imgproc.so.2.4.10
ZED\ SVO\ Playback: /usr/local/cuda-6.5/lib64/libcudart.so
ZED\ SVO\ Playback: /usr/local/cuda-6.5/lib64/libnppi.so
ZED\ SVO\ Playback: /usr/local/cuda-6.5/lib64/libnpps.so
ZED\ SVO\ Playback: /usr/local/lib/libopencv_core.so.2.4.10
ZED\ SVO\ Playback: CMakeFiles/ZED_SVO_Playback.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable \"ZED SVO Playback\""
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ZED_SVO_Playback.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ZED_SVO_Playback.dir/build: ZED\ SVO\ Playback
.PHONY : CMakeFiles/ZED_SVO_Playback.dir/build

CMakeFiles/ZED_SVO_Playback.dir/requires: CMakeFiles/ZED_SVO_Playback.dir/src/main.o.requires
.PHONY : CMakeFiles/ZED_SVO_Playback.dir/requires

CMakeFiles/ZED_SVO_Playback.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ZED_SVO_Playback.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ZED_SVO_Playback.dir/clean

CMakeFiles/ZED_SVO_Playback.dir/depend:
	cd "/usr/local/zed/sample/cpu/SVO Playback" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/usr/local/zed/sample/cpu/SVO Playback" "/usr/local/zed/sample/cpu/SVO Playback" "/usr/local/zed/sample/cpu/SVO Playback" "/usr/local/zed/sample/cpu/SVO Playback" "/usr/local/zed/sample/cpu/SVO Playback/CMakeFiles/ZED_SVO_Playback.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/ZED_SVO_Playback.dir/depend

