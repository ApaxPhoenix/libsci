cmake_minimum_required(VERSION 3.25)

# The builds a preset names beside the plain release one: the sanitizers and
# clang-tidy. Each is a cache option, so an IDE's own profile can turn one on
# as well as a preset can.
set(SCI_SANITIZE "" CACHE STRING "Sanitizers to build everything with, as -fsanitize names them: address,undefined")
option(SCI_TIDY "Check libsci's own sources with clang-tidy as they build" OFF)
option(SCI_NATIVE "Tune every translation unit for the build machine (-march=native / -mcpu=native)" OFF)

# On Windows libsci is built with clang-cl -- LLVM's own, or the one Visual
# Studio installs beside cl, under VC/Tools/Llvm -- whatever compiler an IDE's
# profile names, as every project here is. A profile naming cl or a GCC, or
# naming no compiler at all, is given clang-cl instead, found through vswhere
# in the newest Visual Studio that has it, previews included; a Clang it names
# is kept. Decided here, before project(), the one place a compiler can still
# be chosen.
if (CMAKE_HOST_WIN32)
    get_filename_component(named "${CMAKE_CXX_COMPILER}" NAME_WE)
    string(TOLOWER "${named}" named)
    if (named STREQUAL "" OR named STREQUAL "cl" OR named MATCHES "^(g\\+\\+|gcc|c\\+\\+|cc)(-[0-9.]+)?$")
        set(programs "ProgramFiles(x86)")
        find_program(vswhere vswhere HINTS "$ENV{${programs}}/Microsoft Visual Studio/Installer" NO_CACHE)
        set(studio "")
        if (vswhere)
            execute_process(COMMAND "${vswhere}" -latest -prerelease -products *
                    -requires Microsoft.VisualStudio.Component.VC.Llvm.Clang -property installationPath
                    OUTPUT_VARIABLE studio OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        endif ()
        find_program(clang clang-cl HINTS "$ENV{ProgramFiles}/LLVM/bin" "${studio}/VC/Tools/Llvm/x64/bin" NO_CACHE)
        if (clang)
            if (NOT named STREQUAL "")
                message(STATUS "${named} cannot build this project (C++26); compiling with ${clang} instead")
            endif ()
            set(CMAKE_C_COMPILER "${clang}" CACHE FILEPATH "C compiler" FORCE)
            set(CMAKE_CXX_COMPILER "${clang}" CACHE FILEPATH "C++ compiler" FORCE)
        endif ()
    endif ()

    # clang-cl's own resource compiler, which links every program's manifest
    # in: the SDK's `rc` is on the path only inside a Visual Studio shell, and
    # neither a CLion profile -- Makefiles or Ninja -- nor a preset run outside
    # one is. Taken whenever no working resource compiler is named, including
    # a bare `rc` an earlier configure cached and cannot find.
    if (CMAKE_CXX_COMPILER MATCHES "clang-cl")
        set(working "")
        if (IS_ABSOLUTE "${CMAKE_RC_COMPILER}" AND EXISTS "${CMAKE_RC_COMPILER}")
            set(working "${CMAKE_RC_COMPILER}")
        elseif (CMAKE_RC_COMPILER)
            find_program(working NAMES "${CMAKE_RC_COMPILER}" NO_CACHE)
        endif ()
        if (NOT working)
            get_filename_component(tools "${CMAKE_CXX_COMPILER}" DIRECTORY)
            find_program(resources llvm-rc HINTS "${tools}" NO_CACHE)
            if (resources)
                set(CMAKE_RC_COMPILER "${resources}" CACHE FILEPATH "Resource compiler" FORCE)
            endif ()
        endif ()
    endif ()
endif ()

# The version is VERSION's first line: the packages' names and every release
# take it from there, so a new version is that one line changed. A second
# line, when there is one, is the release's title. A pre-release, like
# 0.2.0-beta.1, is its number alone to CMake, and whole everywhere it is shown.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" version LIMIT_COUNT 1)
string(STRIP "${version}" version)
string(REGEX MATCH "^[0-9]+[.][0-9]+[.][0-9]+" number "${version}")
project(libsci VERSION ${number} LANGUAGES C CXX)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS VERSION)

# C++26, with no compiler's own dialect of it. CMake does not know yet that
# some compilers accept it (AppleClang 16, for one), so for those it starts
# from C++23 and the flag is passed directly; the later -std wins.
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
if ("cxx_std_26" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 26)
else ()
    set(CMAKE_CXX_STANDARD 23)
    if (MSVC)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/std:c++latest>)
    else ()
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++26>)
    endif ()
endif ()

# A bare `cmake -B build` makes the build that ships, not CMake's default of
# no optimisation at all.
get_property(multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (NOT multi AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "The configuration to build" FORCE)
endif ()

# The C and C++ runtimes linked into the program rather than loaded beside it,
# so a .zip runs on a machine with no Visual C++ redistributable.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Defined before anything reads them. The documentation target below names
# both, and naming them before they are set silently documents nothing --
# which looks exactly like a build that worked.
set(source "${CMAKE_CURRENT_SOURCE_DIR}")
set(binary "${CMAKE_CURRENT_BINARY_DIR}")

# The program lands in the source tree's bin/, where the README says to find
# it, unless a preset keeps its own -- the sanitized and analysis builds do, so
# their programs never stand in for the release's. Under another project,
# libsci keeps to that project's.
if (PROJECT_IS_TOP_LEVEL AND NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${source}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${source}/bin")
endif ()

# Warnings are errors waiting to be read. They are libsci's own targets'
# alone: Slang's, Vulkan's and asmjit's headers are theirs, and their noise --
# none of which this project can act on -- would bury libsci's.
if (MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(warnings /W4 /permissive- -Wshadow)
elseif (MSVC)
    set(warnings /W4 /permissive-)
else ()
    set(warnings -Wall -Wextra -Wpedantic -Wshadow)
endif ()

# Arrays are numbers in bulk, so every build but a debug one is optimised
# fully, and SCI_NATIVE tunes it for the machine it runs on.
if (MSVC)
    set(optimisation $<$<NOT:$<CONFIG:Debug>>:/O2>)
else ()
    set(optimisation $<$<NOT:$<CONFIG:Debug>>:-O3>)
    if (SCI_NATIVE)
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            list(APPEND optimisation -mcpu=native)
        else ()
            list(APPEND optimisation -march=native)
        endif ()
    endif ()
endif ()

# A release build carries nothing it does not run. Every function and
# constant nothing reaches is dropped at link time, identical ones are folded
# into one, and symbols are stripped.
if (MSVC)
    add_compile_options($<$<CONFIG:Release>:/Gy> $<$<CONFIG:Release>:/Gw>)
    add_link_options($<$<CONFIG:Release>:/OPT:REF> $<$<CONFIG:Release>:/OPT:ICF>)
else ()
    add_compile_options($<$<CONFIG:Release>:-ffunction-sections> $<$<CONFIG:Release>:-fdata-sections>)
    if (APPLE)
        add_link_options($<$<CONFIG:Release>:-Wl,-dead_strip> $<$<CONFIG:Release>:-Wl,-S>)
    else ()
        add_link_options($<$<CONFIG:Release>:-Wl,--gc-sections> $<$<CONFIG:Release>:-Wl,--strip-all>)
    endif ()
endif ()

# Sanitizers over libsci, so a memory error or undefined behaviour stops the
# program where it happens rather than corrupting what comes after.
if (SCI_SANITIZE)
    if (MSVC)
        message(FATAL_ERROR "SCI_SANITIZE needs Clang on macOS or Linux")
    endif ()
    add_compile_options(-fsanitize=${SCI_SANITIZE} -fno-sanitize-recover=all -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${SCI_SANITIZE})
endif ()

find_package(Doxygen)

if (DOXYGEN_FOUND)
    set(DOXYGEN_PROJECT_NAME "libsci")
    set(DOXYGEN_PROJECT_NUMBER "${version}")
    set(DOXYGEN_PROJECT_BRIEF "numpy for C++26, on any CPU or GPU")
    set(DOXYGEN_OUTPUT_DIRECTORY "${binary}/docs")
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
    set(DOXYGEN_MARKDOWN_SUPPORT YES)
    set(DOXYGEN_SOURCE_BROWSER YES)

    # A parameter block that does not match the signature is a comment that
    # has drifted from its code, which is worse than no comment at all.
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
    set(DOXYGEN_WARN_IF_DOC_ERROR YES)
    set(DOXYGEN_QUIET YES)

    # Doxygen's own look, with a switch between light and dark in the top bar
    # and a copy button on every block of code. Headings take GitHub's
    # anchors, so the README's links to its own sections work here as they do
    # there.
    set(DOXYGEN_HTML_COLORSTYLE TOGGLE)
    set(DOXYGEN_HTML_COPY_CLIPBOARD YES)
    set(DOXYGEN_MARKDOWN_ID_STYLE GITHUB)

    # The README is the front page, and the guides -- each a Markdown file
    # beside the code it describes, never a docs folder -- and the policies
    # are pages of their own beside the reference. Everything is generated
    # into the build tree's docs/; the source tree keeps no docs folder.
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${source}/README.md")
    file(GLOB pages CONFIGURE_DEPENDS "${source}/*.md" "${source}/bindings/*/README.md" "${source}/frontend/README.md"
            "${source}/benchmarks/*.md" "${source}/tests/*.md" "${source}/assets/*/*.md")
    doxygen_add_docs(docs ${pages} "${source}/include" "${source}/src"
            COMMENT "Generating documentation")
endif ()

# Slang and asmjit, pinned: Slang by its release, asmjit by its commit and
# hash. Each is downloaded once, into vendors/, and left there as it was
# released. Slang's compiler builds the fixed shaders; its runtime library
# lets the GPU JIT compile shaders while the program runs. asmjit writes the
# CPU JIT's machine code.
set(slang "2026.12.2")
set(assembler "0d7f01052f2d3f6d6b9bb6714ef32bd7b80177eb")

if (WIN32)
    set(platform "windows-x86_64")
    set(library "${source}/vendors/slang/lib/slang.lib")
    set(compiler "${source}/vendors/slang/bin/slangc.exe")
elseif (APPLE)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        set(platform "macos-aarch64")
    else ()
        set(platform "macos-x86_64")
    endif ()
    set(library "${source}/vendors/slang/lib/libslang.dylib")
    set(compiler "${source}/vendors/slang/bin/slangc")
else ()
    set(platform "linux-x86_64")
    set(library "${source}/vendors/slang/lib/libslang.so")
    set(compiler "${source}/vendors/slang/bin/slangc")
endif ()

include(FetchContent)
FetchContent_Declare(slang
        URL "https://github.com/shader-slang/slang/releases/download/v${slang}/slang-${slang}-${platform}.zip"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_DIR "${source}/vendors/slang")
FetchContent_Declare(asmjit
        URL "https://github.com/asmjit/asmjit/archive/${assembler}.tar.gz"
        URL_HASH SHA256=955a378fac217f0e9a57e61d55d869fffa21f3217e361d9402c5251bf91e1faa
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_DIR "${source}/vendors/asmjit")
set(ASMJIT_STATIC ON CACHE BOOL "" FORCE)
set(ASMJIT_NO_FOREIGN ON CACHE BOOL "" FORCE)
set(ASMJIT_NO_UJIT ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(slang asmjit)

find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "${source}/src/*.cpp")
file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${source}/include/*.hpp")
file(GLOB shaders CONFIGURE_DEPENDS "${source}/shaders/*.slang")
file(GLOB common CONFIGURE_DEPENDS "${source}/shaders/common/*.slang")
list(REMOVE_ITEM sources "${source}/src/main.cpp")

# The library, compiled once, as the target sci: the demonstration and every
# test link this rather than each compiling all of it again.
add_library(sci STATIC ${sources} ${headers})
add_library(sci::sci ALIAS sci)

# Each SIMD kernel file is compiled for its own instruction set; the
# dispatcher picks one at run time from cpuid, so the rest of the library
# stays at the baseline.
set(kernels "${source}/src/core/drivers/processor/kernels")
list(LENGTH CMAKE_OSX_ARCHITECTURES slices)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i[3-6]86" AND slices LESS_EQUAL 1)
    if (MSVC)
        set_source_files_properties("${kernels}/avx2.cpp" PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
        set_source_files_properties("${kernels}/avx512.cpp" PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    else ()
        set_source_files_properties("${kernels}/avx2.cpp" PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
        set_source_files_properties("${kernels}/avx512.cpp" PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx2;-mfma")
    endif ()
endif ()

# The compute shaders. Every shaders/*.slang compiles, all of its entry
# points, to one SPIR-V module, and the modules are compiled into the library
# as sci::shaders::<file name> through the generated header shaders/spirv.hpp,
# so nothing has to be found on disk at run time.
set(generated "${binary}/generated")
set(modules "")
foreach (shader IN LISTS shaders)
    get_filename_component(name "${shader}" NAME_WE)
    set(module "${generated}/shaders/${name}.spv")
    add_custom_command(
            OUTPUT "${module}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${generated}/shaders"
            COMMAND "${compiler}" "${shader}" -target spirv -fvk-use-entrypoint-name -O3 -o "${module}"
            DEPENDS ${shaders} ${common}
            COMMENT "Compiling shader ${name}.slang"
            VERBATIM)
    list(APPEND modules "${module}")
endforeach ()

string(JOIN "|" listing ${modules})
add_custom_command(
        OUTPUT "${generated}/shaders/spirv.hpp"
        COMMAND ${CMAKE_COMMAND} "-DOUTPUT=${generated}/shaders/spirv.hpp" "-DINPUTS=${listing}"
                -P "${source}/cmake/embed.cmake"
        DEPENDS ${modules} "${source}/cmake/embed.cmake"
        COMMENT "Embedding SPIR-V"
        VERBATIM)
target_sources(sci PRIVATE "${generated}/shaders/spirv.hpp")

# The shared shader code, shaders/common/*.slang, as text through the
# generated header shaders/sources.hpp: the GPU JIT compiles it together with
# the code it generates for each chunk.
string(JOIN "|" listing ${common})
add_custom_command(
        OUTPUT "${generated}/shaders/sources.hpp"
        COMMAND ${CMAKE_COMMAND} "-DOUTPUT=${generated}/shaders/sources.hpp" "-DINPUTS=${listing}"
                -P "${source}/cmake/sources.cmake"
        DEPENDS ${common} "${source}/cmake/sources.cmake"
        COMMENT "Embedding shader sources"
        VERBATIM)
target_sources(sci PRIVATE "${generated}/shaders/sources.hpp")

target_include_directories(sci PUBLIC "${source}/include" PRIVATE "${generated}")

# Third-party headers are system headers, so their warnings stay out of
# libsci's build.
target_include_directories(sci SYSTEM PRIVATE "${source}/vendors/slang/include")
get_target_property(includes asmjit INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(asmjit PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${includes}")

target_link_libraries(sci PUBLIC Vulkan::Vulkan Threads::Threads)
target_link_libraries(sci PRIVATE asmjit::asmjit "${library}")
target_compile_options(sci PRIVATE ${warnings} ${optimisation})
if (WIN32)
    # The SCI_* settings are read with getenv, which the Windows C runtime marks
    # deprecated for the pointer it returns; libsci copies what it reads at once.
    target_compile_definitions(sci PRIVATE _CRT_SECURE_NO_WARNINGS)
endif ()

# The CPU's features are asked of it through __builtin_cpu_supports, which
# clang-cl answers from compiler-rt's builtins library: linked here, from
# beside the compiler, since clang-cl's driver does not add it by itself.
if (MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    get_filename_component(tools "${CMAKE_CXX_COMPILER}" DIRECTORY)
    file(GLOB builtins "${tools}/../lib/clang/*/lib/windows/clang_rt.builtins-x86_64.lib")
    if (builtins)
        list(GET builtins 0 builtins)
        target_link_libraries(sci PRIVATE "${builtins}")
    endif ()
endif ()

# On macOS, host matrix products go through Accelerate -- Apple's BLAS, on the
# AMX units -- as NumPy's do.
if (APPLE)
    find_library(accelerate Accelerate)
    if (accelerate)
        target_link_libraries(sci PRIVATE ${accelerate})
        target_compile_definitions(sci PRIVATE SCI_ACCELERATE ACCELERATE_NEW_LAPACK)
    endif ()
endif ()

# The demonstration, in one file: it is not part of the library, and nothing
# in the library depends on it.
add_executable(libsci "${source}/src/main.cpp")
target_link_libraries(libsci PRIVATE sci)
target_compile_options(libsci PRIVATE ${warnings} ${optimisation})
target_compile_definitions(libsci PRIVATE SCI_VERSION="${version}")

# clang-tidy over libsci's own sources as they build, as .clang-tidy
# configures it: the analyze preset turns it on. The tools are taken from
# beside the compiler first, then from the PATH.
get_filename_component(tools "${CMAKE_CXX_COMPILER}" DIRECTORY)
if (SCI_TIDY)
    find_program(tidy NAMES clang-tidy-22 clang-tidy HINTS "${tools}" REQUIRED)
    set_target_properties(sci libsci PROPERTIES CXX_CLANG_TIDY "${tidy};--config-file=${source}/.clang-tidy")
endif ()

# `format` lays the sources out as .clang-format says, and `format-check`
# fails on any that are not.
find_program(formatter NAMES clang-format-22 clang-format HINTS "${tools}")
if (formatter)
    set(formatted ${sources} ${headers} "${source}/src/main.cpp")
    add_custom_target(format COMMAND "${formatter}" --style=file -i ${formatted} VERBATIM)
    add_custom_target(format-check COMMAND "${formatter}" --style=file --dry-run --Werror ${formatted} VERBATIM)
endif ()

enable_testing()
if (EXISTS "${source}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif ()

# What an installed copy holds: the demonstration in bin/, the library and its
# headers for a program that links libsci, and the license and notices.
install(TARGETS libsci RUNTIME DESTINATION bin)
install(TARGETS sci ARCHIVE DESTINATION lib)
install(DIRECTORY "${source}/include/" DESTINATION include/libsci)
install(FILES "${source}/README.md" "${source}/LICENSE" "${source}/LICENSING.md" "${source}/THIRD-PARTY-NOTICES.md"
        DESTINATION share/doc/libsci)

# A release build packages itself: `cpack` in the build tree -- or the
# `package` target -- leaves the packages in the source tree's bin/ folder,
# named for the version, the system and the architecture:
# libsci-0.1.0-alpha.1-macos-arm64.tar.gz. A .zip everywhere, and a .tar.gz
# beside it on macOS and Linux.
if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_CONFIGURATION_TYPES)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" processor)
    if (APPLE AND CMAKE_OSX_ARCHITECTURES)
        string(TOLOWER "${CMAKE_OSX_ARCHITECTURES}" processor)
    endif ()
    if (processor MATCHES "^(x64|x86_64|amd64)$")
        set(architecture x64)
    elseif (processor MATCHES "^(arm64|aarch64)$")
        set(architecture arm64)
    else ()
        set(architecture "${processor}")
    endif ()
    if (WIN32)
        set(platform windows)
        set(CPACK_GENERATOR ZIP)
    elseif (APPLE)
        set(platform macos)
        set(CPACK_GENERATOR TGZ ZIP)
    else ()
        set(platform linux)
        set(CPACK_GENERATOR TGZ ZIP)
    endif ()
    set(CPACK_PACKAGE_NAME libsci)
    set(CPACK_PACKAGE_VENDOR "Andres Hernandez")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "numpy for C++26: one pass over the data, on any CPU or GPU")
    set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/ApaxPhoenix/libsci")
    set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
    set(CPACK_PACKAGE_FILE_NAME "libsci-${version}-${platform}-${architecture}")
    set(CPACK_OUTPUT_FILE_PREFIX "${source}/bin")
    set(CPACK_STRIP_FILES ON)
    include(CPack)
endif ()
