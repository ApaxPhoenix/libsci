#pragma once

namespace sci::array {

    // Tag for storage whose elements are about to be overwritten, so the zero-fill can be skipped.
    struct Uninitialized {};

}
