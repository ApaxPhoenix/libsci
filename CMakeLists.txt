cmake_minimum_required(VERSION 3.20)
project(libsci VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

add_executable(libsci ${SOURCES} ${HEADERS})

target_include_directories(libsci PRIVATE include)

if(MSVC)
    target_compile_options(libsci PRIVATE /W4 /permissive-)
else()
    target_compile_options(libsci PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()