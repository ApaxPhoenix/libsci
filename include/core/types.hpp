#pragma once

/// @file
/// @brief The integer and floating-point types every part of libsci names.

#include <cstddef>
#include <cstdint>

namespace sci::core {

    /// @brief A count or size.
    using size = std::size_t;

    /// @brief A signed index.
    using index = std::ptrdiff_t;

    /// @brief A 32-bit float.
    using float32 = float;

    /// @brief A 64-bit float.
    using float64 = double;

    /// @brief An unsigned byte.
    using uint8 = std::uint8_t;

    /// @brief An unsigned 32-bit integer.
    using uint32 = std::uint32_t;

    /// @brief An unsigned 64-bit integer.
    using uint64 = std::uint64_t;

    /// @brief A signed byte.
    using int8 = std::int8_t;

    /// @brief A signed 32-bit integer.
    using int32 = std::int32_t;

    /// @brief A signed 64-bit integer.
    using int64 = std::int64_t;

}
