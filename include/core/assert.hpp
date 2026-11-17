#pragma once

/// @file
/// @brief Checks compiled into debug builds and left out of release ones.

#include <cassert>

#define ASSERT(condition) assert(condition)
