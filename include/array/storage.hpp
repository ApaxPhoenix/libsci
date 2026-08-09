#pragma once

#include <cstdlib>
#include <utility>
#include <core/arch/alignment.hpp>
#include <core/types.hpp>

namespace sci::array {

    template <typename element>
    class storage {
        element* buffer = nullptr;
        core::size count = 0;

    public:
        storage() = default;

        explicit storage(core::size total) : count(total) {
            if (count > 0) {
                buffer = static_cast<element*>(
                    ::operator new[](count * sizeof(element), std::align_val_t{core::arch::boundary})
                );
            }
        }

        ~storage() {
            if (buffer) {
                ::operator delete[](buffer, std::align_val_t{core::arch::boundary});
            }
        }

        storage(const storage& other) : storage(other.count) {
            for (core::size index = 0; index < count; ++index) {
                buffer[index] = other.buffer[index];
            }
        }

        storage& operator=(const storage& other) {
            if (this != &other) {
                storage temp(other);
                std::swap(buffer, temp.buffer);
                std::swap(count, temp.count);
            }
            return *this;
        }

        storage(storage&& other) noexcept : buffer(other.buffer), count(other.count) {
            other.buffer = nullptr;
            other.count = 0;
        }

        storage& operator=(storage&& other) noexcept {
            if (this != &other) {
                if (buffer) {
                    ::operator delete[](buffer, std::align_val_t{core::arch::boundary});
                }
                buffer = other.buffer;
                count = other.count;
                other.buffer = nullptr;
                other.count = 0;
            }
            return *this;
        }

        element* data() noexcept { return buffer; }
        const element* data() const noexcept { return buffer; }
        core::size size() const noexcept { return count; }
    };

}