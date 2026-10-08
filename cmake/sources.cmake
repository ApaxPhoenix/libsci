# Turns shader source files into a C++ header of string constants, one per file, named after it
# (shaders/common/layout.slang -> sci::shaders::sources::layout).
#
#   cmake -DOUTPUT=<header> "-DINPUTS=<a.slang|b.slang>" -P sources.cmake

string(REPLACE "|" ";" INPUTS "${INPUTS}")

set(CONTENT "// Generated from shaders/common/*.slang by cmake/sources.cmake. Do not edit.\n#pragma once\n\nnamespace sci::shaders::sources {\n")

foreach(INPUT ${INPUTS})
    get_filename_component(NAME "${INPUT}" NAME_WE)
    file(READ "${INPUT}" TEXT)
    string(APPEND CONTENT "\n    inline constexpr char ${NAME}[] = R\"slang(${TEXT})slang\";\n")
endforeach()

string(APPEND CONTENT "\n}\n")

# Rewrite only on change so dependents are not rebuilt for nothing.
if(EXISTS "${OUTPUT}")
    file(READ "${OUTPUT}" PREVIOUS)
endif()
if(NOT "${PREVIOUS}" STREQUAL "${CONTENT}")
    file(WRITE "${OUTPUT}" "${CONTENT}")
endif()
