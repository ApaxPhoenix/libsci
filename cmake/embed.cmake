# Turns compiled SPIR-V modules into a C++ header of word arrays, one array per module named after
# its file (shaders/interpreter.slang -> sci::shaders::interpreter).
#
#   cmake -DOUTPUT=<header> "-DINPUTS=<a.spv|b.spv>" -P embed.cmake

string(REPLACE "|" ";" INPUTS "${INPUTS}")

set(CONTENT "// Generated from shaders/*.slang by cmake/embed.cmake. Do not edit.\n#pragma once\n\n#include <cstdint>\n\nnamespace sci::shaders {\n")

foreach(INPUT ${INPUTS})
    get_filename_component(NAME "${INPUT}" NAME_WE)
    file(READ "${INPUT}" BYTES HEX)
    # SPIR-V words are little-endian on disk.
    string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1, " WORDS "${BYTES}")
    # Eight words per line (CMake regexes have no {n} repetition).
    string(REPEAT "0x[0-9a-f]+, " 8 LINE)
    string(REGEX REPLACE "(${LINE})" "\\1\n        " WORDS "${WORDS}")
    string(STRIP "${WORDS}" WORDS)
    string(APPEND CONTENT "\n    inline constexpr std::uint32_t ${NAME}[] = {\n        ${WORDS}\n    };\n")
endforeach()

string(APPEND CONTENT "\n}\n")

# Rewrite only on change so dependents are not rebuilt for nothing.
if(EXISTS "${OUTPUT}")
    file(READ "${OUTPUT}" PREVIOUS)
endif()
if(NOT "${PREVIOUS}" STREQUAL "${CONTENT}")
    file(WRITE "${OUTPUT}" "${CONTENT}")
endif()
