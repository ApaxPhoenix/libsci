file(GLOB_RECURSE tests CONFIGURE_DEPENDS "*.cpp")

# The demonstration's test runs the program itself, so a build without it
# leaves that test out.
if (NOT TARGET libsci)
    list(FILTER tests EXCLUDE REGEX "/test_main\.cpp$")
endif ()

# One test per source file or header, at the same place in the tree --
# include/linear/product.hpp is tested by tests/linear/test_product.cpp --
# each standing on its own. Two may share a name, as core/drivers/processor/
# compiler.cpp and core/drivers/vulkan/compiler.cpp do, so a test is named
# for its file alone unless another shares that name, and then for its
# folders too.
set(names "")
foreach (test ${tests})
    get_filename_component(name ${test} NAME_WE)
    list(APPEND names ${name})
endforeach ()

foreach (test ${tests})
    get_filename_component(name ${test} NAME_WE)
    set(same "")
    foreach (other ${names})
        if (other STREQUAL name)
            list(APPEND same ${other})
        endif ()
    endforeach ()
    list(LENGTH same shared)
    if (shared GREATER 1)
        file(RELATIVE_PATH place ${CMAKE_CURRENT_SOURCE_DIR} ${test})
        get_filename_component(place ${place} DIRECTORY)
        string(REPLACE "/" "_" place ${place})
        string(REGEX REPLACE "^test_" "test_${place}_" name ${name})
    endif ()

    # libsci is linked, not recompiled per test. Each test stays in the build
    # tree, out of bin/.
    add_executable(${name} ${test})
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(${name} PRIVATE sci)
    target_compile_options(${name} PRIVATE ${warnings})

    # A test checks with assert, which a release build compiles out: kept on
    # here, so a test built for speed still checks everything it says.
    target_compile_options(${name} PRIVATE $<IF:$<CXX_COMPILER_FRONTEND_VARIANT:MSVC>,/UNDEBUG,-UNDEBUG>)

    # Run from the source tree, where a test finds the files it reads.
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endforeach ()

# The demonstration's test runs the executable itself, so it is told where the
# build put it, and waits for it to be built.
if (TARGET test_main)
    if (WIN32)
        set(null "NUL")
    else ()
        set(null "/dev/null")
    endif ()
    target_compile_definitions(test_main PRIVATE BINARY="$<TARGET_FILE:libsci>" NULL_DEVICE="${null}")
    add_dependencies(test_main libsci)
endif ()
