#pragma once

#include "core/alignment.hpp"
#include "core/drivers/device.hpp"
#include "core/drivers/memory.hpp"
#include "core/drivers/target.hpp"
#include "core/memory/allocation.hpp"
#include "core/memory/arena.hpp"
#include "core/types.hpp"

#include <concepts>
#include <memory>
#include <utility>

namespace sci::array {

    using core::drivers::Device;
    using core::drivers::Target;

    // Where an array can live besides plain host memory.
    template <typename place>
    concept location = std::same_as<place, Device> || std::same_as<place, core::memory::Arena>;

    // The memory behind an Array: host memory it owns, memory on a Device, or a slice of an Arena.
    // The elements are left for the Array to fill.
    template <typename element>
    class Storage {
    public:
        Storage() = default;

        explicit Storage(const core::size total)
            : pointer(static_cast<element*>(allocation.reserve(total * sizeof(element)))), count(total) {
            std::uninitialized_default_construct_n(pointer, count);
        }

        Storage(Device& device, const core::size total)
            : count(total), owner(&device) {
            if (count == 0) return;
            device.allocate(count * sizeof(element), memory);
            pointer = static_cast<element*>(memory.host);
        }

        // The arena keeps ownership.
        Storage(core::memory::Arena& arena, const core::size total)
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

        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;

        Storage(Storage&& storage) noexcept
            : allocation(std::move(storage.allocation)),
              memory(std::exchange(storage.memory, {})),
              pointer(std::exchange(storage.pointer, nullptr)),
              count(std::exchange(storage.count, 0)),
              owner(std::exchange(storage.owner, nullptr)) {}

        // Swaps, so the storage moved from releases what this one held.
        Storage& operator=(Storage&& storage) noexcept {
            std::swap(allocation, storage.allocation);
            std::swap(memory, storage.memory);
            std::swap(pointer, storage.pointer);
            std::swap(count, storage.count);
            std::swap(owner, storage.owner);
            return *this;
        }

        [[nodiscard]] element* data() noexcept { return pointer; }
        [[nodiscard]] const element* data() const noexcept { return pointer; }

        [[nodiscard]] core::size size() const noexcept { return count; }
        [[nodiscard]] Device* device() const noexcept { return owner; }
        [[nodiscard]] Target target() const noexcept { return owner ? memory.target : Target::processor; }

        // Lets the host touch the memory: waits for queued GPU work on the storage's device.
        void synchronize() const noexcept {
            if (owner) owner->synchronize();
        }

    private:
        core::memory::Allocation allocation{core::size{0}, core::alignment::line};
        core::drivers::Memory memory{};
        element* pointer{nullptr};
        core::size count{0};
        Device* owner{nullptr};
    };

}
