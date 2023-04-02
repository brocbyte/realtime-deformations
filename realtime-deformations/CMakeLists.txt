cmake_minimum_required (VERSION 3.8 FATAL_ERROR)
project(realtime-deformations LANGUAGES CXX CUDA)
find_package(OpenGL REQUIRED)

if( CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR )
    message( FATAL_ERROR "Please select another Build Directory ! (and give it a clever name, like bin_Visual2012_64bits/)" )
endif()
if( CMAKE_SOURCE_DIR MATCHES " " )
	message( "Your Source Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()
if( CMAKE_BINARY_DIR MATCHES " " )
	message( "Your Build Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()


add_subdirectory (external)

include_directories(
	external/glfw-3.1.2/include/
	external/glm-0.9.7.1/
	external/glew-1.13.0/include/
	external/assimp-3.0.1270/include/
	.
	include/
	external/
)

set(ALL_LIBS
	${OPENGL_LIBRARY}
	glfw
	GLEW_1130
)

add_definitions(
	-DTW_STATIC
	-DTW_NO_LIB_PRAGMA
	-DTW_NO_DIRECT3D
	-DGLEW_STATIC
	-D_CRT_SECURE_NO_WARNINGS
)
add_executable (realtime-deformations
	"main.cpp" 
	"constants.hpp"
	"user_controls.hpp"
	"user_controls.cpp"
	"camera.hpp"
	"camera.cpp"
	"fps_counter.hpp"
	"material_point_method.hpp"
	"material_point_method.cpp"
	"cudaCalc.cu"
	"utils.h"
	"mathy.hpp"
	"mathy.cpp"
	"mesh.hpp"
	"mesh.cpp"
	"logger.hpp"
	"cudaCalc.cuh"
	common/shader.cpp
	common/shader.hpp
	common/objloader.cpp
	common/objloader.hpp
	common/vboindexer.cpp
	common/vboindexer.hpp
	common/texture.cpp
	common/texture.hpp
	common/text2D.cpp
	common/text2D.hpp
)

set_property(TARGET realtime-deformations
			 PROPERTY CXX_STANDARD 17)

set_property(TARGET realtime-deformations
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET realtime-deformations
			 PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

if (MSVC)
    # warning level 4
    add_compile_options(/W4 /P)
else()
    # additional warnings
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(OpenMP REQUIRED)
set(ALL_LIBS
	${ALL_LIBS}
	OpenMP::OpenMP_CXX
)

target_link_libraries(realtime-deformations
	${ALL_LIBS}
)

set_target_properties(realtime-deformations PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/"
	RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/"
	RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/"
	RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}/"
	RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/")

