cmake_minimum_required(VERSION 3.20)
project(libsci VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(SCI_NATIVE "Tune every translation unit for the build machine (-march=native / -mcpu=native)" OFF)

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

# asmjit writes the CPU JIT's machine code. Pinned to one commit and checked against its hash.
FetchContent_Declare(
        asmjit
        URL "https://github.com/asmjit/asmjit/archive/0d7f01052f2d3f6d6b9bb6714ef32bd7b80177eb.tar.gz"
        URL_HASH SHA256=955a378fac217f0e9a57e61d55d869fffa21f3217e361d9402c5251bf91e1faa
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendors/asmjit"
)
set(ASMJIT_STATIC ON CACHE BOOL "" FORCE)
set(ASMJIT_NO_FOREIGN ON CACHE BOOL "" FORCE)
set(ASMJIT_NO_UJIT ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(slang asmjit)

find_package(Vulkan REQUIRED)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
file(GLOB SLANG_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.slang")
file(GLOB SLANG_COMMON CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/*.slang")

list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

add_library(sci STATIC ${SOURCES} ${HEADERS})

# Each SIMD kernel file is compiled for its own ISA; the dispatcher picks one at runtime from cpuid,
# so the rest of the library stays at the baseline ISA.
set(SCI_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/src/core/drivers/processor/kernels")
list(LENGTH CMAKE_OSX_ARCHITECTURES SCI_ARCH_COUNT)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i[3-6]86" AND SCI_ARCH_COUNT LESS_EQUAL 1)
    if(MSVC)
        set_source_files_properties("${SCI_KERNELS}/avx2.cpp" PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
        set_source_files_properties("${SCI_KERNELS}/avx512.cpp" PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    else()
        set_source_files_properties("${SCI_KERNELS}/avx2.cpp" PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
        set_source_files_properties("${SCI_KERNELS}/avx512.cpp" PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx2;-mfma")
    endif()
endif()

# Slang compute shaders. Every shaders/*.slang compiles (all of its entry points) to one SPIR-V
# module, and the modules are embedded into the library as sci::shaders::<file name> through the
# generated header shaders/spirv.hpp, so nothing has to be found on disk at runtime.
set(SCI_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(SCI_SPIRV "")
foreach(SHADER ${SLANG_SOURCES})
    get_filename_component(NAME "${SHADER}" NAME_WE)
    set(SPIRV "${SCI_GENERATED}/shaders/${NAME}.spv")
    add_custom_command(
            OUTPUT "${SPIRV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${SCI_GENERATED}/shaders"
            COMMAND "${SLANGC_EXE}" "${SHADER}" -target spirv -fvk-use-entrypoint-name -O3 -o "${SPIRV}"
            DEPENDS ${SLANG_SOURCES} ${SLANG_COMMON}
            COMMENT "Compiling shader ${NAME}.slang"
            VERBATIM
    )
    list(APPEND SCI_SPIRV "${SPIRV}")
endforeach()

string(JOIN "|" SCI_SPIRV_LIST ${SCI_SPIRV})
add_custom_command(
        OUTPUT "${SCI_GENERATED}/shaders/spirv.hpp"
        COMMAND ${CMAKE_COMMAND} "-DOUTPUT=${SCI_GENERATED}/shaders/spirv.hpp" "-DINPUTS=${SCI_SPIRV_LIST}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake"
        DEPENDS ${SCI_SPIRV} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake"
        COMMENT "Embedding SPIR-V"
        VERBATIM
)
target_sources(sci PRIVATE "${SCI_GENERATED}/shaders/spirv.hpp")

# The shared shader code (shaders/common/*.slang) as text, through the generated header
# shaders/sources.hpp: the GPU JIT compiles it together with the code it generates for each chunk.
string(JOIN "|" SCI_COMMON_LIST ${SLANG_COMMON})
add_custom_command(
        OUTPUT "${SCI_GENERATED}/shaders/sources.hpp"
        COMMAND ${CMAKE_COMMAND} "-DOUTPUT=${SCI_GENERATED}/shaders/sources.hpp" "-DINPUTS=${SCI_COMMON_LIST}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/sources.cmake"
        DEPENDS ${SLANG_COMMON} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/sources.cmake"
        COMMENT "Embedding shader sources"
        VERBATIM
)
target_sources(sci PRIVATE "${SCI_GENERATED}/shaders/sources.hpp")

target_include_directories(sci
        PUBLIC
        include
        PRIVATE
        "${SCI_GENERATED}"
)

# Third-party headers are system headers, so their warnings stay out of libsci's build.
target_include_directories(sci SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendors/slang/include")
get_target_property(ASMJIT_INCLUDES asmjit INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(asmjit PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${ASMJIT_INCLUDES}")

find_package(Threads REQUIRED)

target_link_libraries(sci PUBLIC
        Vulkan::Vulkan
        Threads::Threads
)

# asmjit for the CPU JIT, and Slang's runtime library for the GPU JIT, which compiles shaders while
# the program runs (slangc compiles the fixed ones at build time).
target_link_libraries(sci PRIVATE
        asmjit::asmjit
        "${SLANG_LIB}"
)

# On macOS, host matrix products go through Accelerate (Apple's BLAS, on the AMX units), as NumPy's do.
if(APPLE)
    find_library(ACCELERATE Accelerate)
    if(ACCELERATE)
        target_link_libraries(sci PRIVATE ${ACCELERATE})
        target_compile_definitions(sci PRIVATE SCI_ACCELERATE ACCELERATE_NEW_LAPACK)
    endif()
endif()

if(MSVC)
    set(SCI_WARNINGS /W4 /permissive-)
    set(SCI_OPTIMIZE /O2)
else()
    set(SCI_WARNINGS -Wall -Wextra -Wpedantic)
    set(SCI_OPTIMIZE -O3)
    if(SCI_NATIVE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            list(APPEND SCI_OPTIMIZE -mcpu=native)
        else()
            list(APPEND SCI_OPTIMIZE -march=native)
        endif()
    endif()
endif()

target_compile_options(sci PRIVATE ${SCI_WARNINGS} ${SCI_OPTIMIZE})

add_executable(libsci src/main.cpp)
target_link_libraries(libsci PRIVATE sci)
target_compile_options(libsci PRIVATE ${SCI_WARNINGS} ${SCI_OPTIMIZE})
