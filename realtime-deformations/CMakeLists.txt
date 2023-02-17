cmake_minimum_required (VERSION 3.0)
project(realtime-deformations)
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
	external/AntTweakBar-1.16/include/
	external/glfw-3.1.2/include/
	external/glm-0.9.7.1/
	external/glew-1.13.0/include/
	external/assimp-3.0.1270/include/
	external/bullet-2.81-rev2613/src/
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
	"utils.h"
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

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET realtime-deformations PROPERTY CXX_STANDARD 23)
endif()

if (MSVC)
    # warning level 4
    add_compile_options(/W4 /P)
else()
    # additional warnings
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

target_link_libraries(realtime-deformations
	${ALL_LIBS}
)

set_target_properties(realtime-deformations PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/"
	RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/"
)
