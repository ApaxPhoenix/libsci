cmake_minimum_required(VERSION 3.20)
project(libsci VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

include(FetchContent)

if(WIN32)
    set(SLANG_ARCH "windows-x86_64")
    set(SLANG_LIB "${CMAKE_CURRENT_SOURCE_DIR}/vendors/slang/lib/slang.lib")
    set(SLANGC_EXE "${CMAKE_CURRENT_SOURCE_DIR}/vendors/slang/bin/slangc.exe")
elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        set(SLANG_ARCH "macos-aarch64")
    else()
        set(SLANG_ARCH "macos-x86_64")
    endif()
    set(SLANG_LIB "${CMAKE_CURRENT_SOURCE_DIR}/vendors/slang/lib/libslang.dylib")
    set(SLANGC_EXE "${CMAKE_CURRENT_SOURCE_DIR}/vendors/slang/bin/slangc")
else()
    set(SLANG_ARCH "linux-x86_64")
    set(SLANG_LIB "${CMAKE_CURRENT_SOURCE_DIR}/vendors/slang/lib/libslang.so")
    set(SLANGC_EXE "${CMAKE_CURRENT_SOURCE_DIR}/vendors/slang/bin/slangc")
endif()

FetchContent_Declare(
        slang
        URL "https://github.com/shader-slang/slang/releases/download/v2026.12.2/slang-2026.12.2-${SLANG_ARCH}.zip"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendors/slang"
)

FetchContent_MakeAvailable(slang)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
file(GLOB_RECURSE SLANG_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/*.slang")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    list(FILTER SOURCES EXCLUDE REGEX ".*/cpu/kernels/(avx2|avx512)\\.cpp$")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i386" OR CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
    list(FILTER SOURCES EXCLUDE REGEX ".*/cpu/kernels/neon\\.cpp$")
endif()

add_executable(libsci ${SOURCES} ${HEADERS})

foreach(SLANG_SRC ${SLANG_SOURCES})
    file(RELATIVE_PATH REL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders" "${SLANG_SRC}")
    string(REGEX REPLACE "\\.slang$" ".spv" REL_SPV "${REL_PATH}")
    set(SPV_OUT "${CMAKE_CURRENT_BINARY_DIR}/generated/shaders/${REL_SPV}")

    get_filename_component(OUT_DIR "${SPV_OUT}" DIRECTORY)

    add_custom_command(
            OUTPUT ${SPV_OUT}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
            COMMAND ${SLANGC_EXE} ${SLANG_SRC} -target spirv -entry main -stage compute -o ${SPV_OUT}
            DEPENDS ${SLANG_SRC}
    )
    target_sources(libsci PRIVATE ${SPV_OUT})
endforeach()

target_include_directories(libsci PRIVATE
        include
        "${CMAKE_CURRENT_SOURCE_DIR}/vendors/slang/include"
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

target_link_libraries(libsci PRIVATE ${SLANG_LIB})

if(MSVC)
    target_compile_options(libsci PRIVATE /W4 /permissive-)
else()
    target_compile_options(libsci PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()