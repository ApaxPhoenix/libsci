#pragma once

/// @file
/// @brief Checks compiled into debug builds and left out of release ones.

#include <cassert>

/// @brief Checks a condition in debug builds, as assert does.
#define ASSERT(condition) assert(condition)
