#pragma once

/// @file
/// @brief Where an array can live besides plain host memory.

#include "core/alignment.hpp"
#include "core/drivers/device.hpp"
#include "core/drivers/memory.hpp"
#include "core/memory/allocation.hpp"
#include "core/memory/arena.hpp"
#include "core/types.hpp"

#include <concepts>
#include <memory>
#include <utility>

namespace sci::array {

    using core::drivers::Device;

    /// Where an array can live besides plain host memory.
    template <typename place>
    concept location = std::same_as<place, Device> || std::same_as<place, core::memory::Arena>;

    /// The memory behind an Array: host memory it owns, memory on a Device, or a slice of an Arena.
    /// The elements are left for the Array to fill.
    template <typename element> class Storage {
    public:
        Storage() = default;

        /// @brief Storage for elements in plain host memory.
        /// @param total How many elements.
        explicit Storage(const core::size total)
            : pointer(static_cast<element *>(allocation.reserve(total * sizeof(element)))), count(total) {
            std::uninitialized_default_construct_n(pointer, count);
        }

        /// @brief Storage for elements on a device.
        /// @param device The device.
        /// @param total How many elements.
        Storage(Device &device, const core::size total) : count(total), owner(&device) {
            if (count == 0) return;
            device.allocate(count * sizeof(element), memory);
            pointer = static_cast<element *>(memory.host);
        }

        /// The arena keeps ownership.
        /// @param arena The arena.
        /// @param total How many elements.
        Storage(core::memory::Arena &arena, const core::size total)
            : pointer(arena.allocate<element>(total)), count(total) {
            std::uninitialized_default_construct_n(pointer, count);
        }

        ~Storage() {
            if (owner) {
                if (memory.bytes > 0) owner->dispose(memory);
            } else if (allocation.data()) {
                std::destroy_n(pointer, count);
            }
        }

        Storage(const Storage &) = delete;
        Storage &operator=(const Storage &) = delete;

        /// @brief Takes another storage's elements, leaving it empty.
        /// @param storage The storage moved from.
        Storage(Storage &&storage) noexcept
            : allocation(std::move(storage.allocation)), memory(std::exchange(storage.memory, {})),
              pointer(std::exchange(storage.pointer, nullptr)), count(std::exchange(storage.count, 0)),
              owner(std::exchange(storage.owner, nullptr)) {}

        /// Swaps, so the storage moved from releases what this one held.
        /// @param storage The storage moved from.
        /// @return This storage.
        Storage &operator=(Storage &&storage) noexcept {
            std::swap(allocation, storage.allocation);
            std::swap(memory, storage.memory);
            std::swap(pointer, storage.pointer);
            std::swap(count, storage.count);
            std::swap(owner, storage.owner);
            return *this;
        }

        /// @brief The first element.
        /// @return A pointer to it.
        [[nodiscard]] element *data() noexcept { return pointer; }
        /// @brief The first element, read-only.
        /// @return A pointer to it.
        [[nodiscard]] const element *data() const noexcept { return pointer; }

        /// @brief How many elements.
        /// @return The count.
        [[nodiscard]] core::size size() const noexcept { return count; }
        /// @brief The device the elements live on.
        /// @return The device, or null for host and arena memory.
        [[nodiscard]] Device *device() const noexcept { return owner; }

        /// Lets the host touch the memory: waits for queued GPU work on the storage's device.
        void synchronize() const noexcept {
            if (owner) owner->synchronize();
        }

    private:
        /// @brief The host memory, for host storage.
        core::memory::Allocation allocation{core::size{0}, core::alignment::line};
        /// @brief The device memory, for device storage.
        core::drivers::Memory memory{};
        /// @brief The first element.
        element *pointer{nullptr};
        /// @brief How many elements.
        core::size count{0};
        /// @brief The device, or null.
        Device *owner{nullptr};
    };

}
